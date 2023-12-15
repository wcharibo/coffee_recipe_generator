from predictResult import *
# import
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from collections import deque
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# class BERTSentenceTransform

class BERTSentenceTransform:
    r"""BERT style data transformation.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """

    def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = vocab

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
        sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 2 strings:
        text_a, text_b.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens: '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14

        For single sequences, the input is a tuple of single string:
        text_a.

        Inputs:
            text_a: 'the dog is hairy .'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a: '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 2 strings:
            (text_a, text_b). For single sequences, the input is a tuple of single
            string: (text_a,).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)

        """

        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        #vocab = self._tokenizer.vocab
        vocab = self._vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')

# class BERTDataset
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int64(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# class BERTClassifier
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)



def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def train_model():
    # 경로 설정
    PATH="./"
    #device - GPU 설정
    device = torch.device("cuda:0")

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

    df = pd.read_csv("output_without_quotes.csv", quoting=1)

    data_list = []
    for q, label in zip(df['Sentence'], df['Label'])  :
        data = []
        data.append(q)
        data.append(str(label))

        data_list.append(data)

    #train test split
    dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, random_state=0)

    # 세팅 파라미터
    max_len = 64
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 15
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  3e-5
    max_non_improving_epochs = 3

    #정의한 모델 불러오기
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # 테스트 데이터 로더
    data_test = BERTDataset(dataset_test, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    # 데이터를 X, y로 나누기
    X = [item[0] for item in dataset_train]
    y = [item[1] for item in dataset_train]

    best_val_acc = 0.0

    # StratifiedKFold를 사용하여 인덱스 분할
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\nFold {fold + 1}/{kfold.get_n_splits()}")

        # 데이터 분할
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        # 데이터 로더
        data_train = BERTDataset(train_data, 0, 1, tokenizer, vocab, max_len, True, False)
        data_val = BERTDataset(val_data, 0, 1, tokenizer, vocab, max_len, True, False)

        train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
        val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, num_workers=5)

        # 모델 및 옵티마이저 초기화
        model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        t_total = len(train_dataloader) * num_epochs
        warmup_step = int(t_total * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        eval_acc = 0.0
        train_loss_li = []
        train_acc_li = []
        val_loss_li = []
        val_acc_li = []
        non_improving_count = 0
        best_model_state = None

        for e in range(num_epochs):
            train_acc = 0.0
            test_acc = 0.0

            model.train()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length= valid_length
                label = label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)

                loss = loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                train_acc += calc_accuracy(out, label)

                if batch_id % log_interval == 0:
                    print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))

            print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
            train_loss_li.append(loss.data.cpu().numpy())
            train_acc_li.append(train_acc / (batch_id+1))

            model.eval()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(val_dataloader)):
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length= valid_length
                label = label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)
                test_acc += calc_accuracy(out, label)

            print("epoch {} validation loss {} validation acc {}".format(e+1, loss.data.cpu().numpy() ,test_acc / (batch_id+1)))

            val_loss_li.append(loss.data.cpu().numpy())
            val_acc_li.append(test_acc / (batch_id+1))

            if test_acc > eval_acc:
                eval_acc = test_acc
                non_improving_count = 0
                best_model_state = model.state_dict()
                torch.save(model, PATH + f'checkpoint_fold{fold}.pt')  # 해당 fold의 모델 저장
                torch.save(model.state_dict(), PATH + f'model_state_fold{fold}.pt')  # 해당 fold의 모델 state_dict 저장
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, PATH + f'all_fold{fold}.tar')  # 해당 fold의 체크포인트 저장
            else:
                non_improving_count += 1

            if non_improving_count >= max_non_improving_epochs:
                print(f"No improvement for {max_non_improving_epochs} epochs. Early stopping.")
                break

        # 추가된 부분: 테스트 데이터에 대한 recall, f1-score 계산
        model.load_state_dict(best_model_state)  # 저장된 최적의 모델로 복원
        model.eval()

        all_preds = []
        all_labels = []

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)

            # 모델 예측
            out = model(token_ids, valid_length, segment_ids)
            _, preds = torch.max(out, 1)

            # 정확도 계산
            test_acc += calc_accuracy(out, label)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        # recall, f1-score 계산
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        accuracy = accuracy_score(all_labels, all_preds)

        print(f"Fold {fold + 1}:Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}, Accuracy = {accuracy:.4f}")
