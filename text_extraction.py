import get_token
import json
import requests
import time
import sys
import os

def get_transcribe_id(file_path, secret_token):
    config = {}
    resp = requests.post(
        'https://openapi.vito.ai/v1/transcribe',
        headers={'Authorization': 'bearer '+ secret_token},
        data={'config': json.dumps(config)},
        files={'file': open(file_path, 'rb')}
    )
    resp.raise_for_status()
    return resp.json()['id']

def check_transcribe_status(transcribe_id, secret_token):
    msg_list = []

    while True:
        resp = requests.get(
            'https://openapi.vito.ai/v1/transcribe/'+transcribe_id,
            headers={'Authorization': 'bearer '+ secret_token},
        )
        resp.raise_for_status()
        response_json = resp.json()

        status = response_json.get('status', '')

        if status == 'completed':
            for i in resp.json()['results']['utterances']:
                msg_list.append(i['msg'])
            return msg_list
        elif status == 'transcribing':
            print('Transcribing...')
        else:
            print(f'Unexpected status: {status}')

        time.sleep(5)

def get_text(file_path):
    access_token = get_token.get_access_token()
    tr_id = get_transcribe_id(file_path, access_token)
    text = check_transcribe_status(tr_id, access_token)
    os.remove(file_path)
    return text

if __name__=="__main__":
    if len(sys.argv) !=2:
        print("Usage: python test_module.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    get_text(file_path)








