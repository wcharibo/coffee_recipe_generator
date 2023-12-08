from pytube import YouTube
import sys
import os
import text_extraction
from pydub import AudioSegment

def get_yt_text(link):
    text_list = []

    yt = YouTube(link)
    mp4_file_path = yt.streams.filter(only_audio=True).first().download()
    file_path = mp4_file_path.replace('mp4','mp3')
    os.rename(mp4_file_path, file_path)

    try:
        audio = AudioSegment.from_file(file_path, "mp3")
    except:
        audio = AudioSegment.from_file(file_path, format="mp4")

    interval = 180 * 1000
    chunks = [audio[i:i + interval] for i in range(0, len(audio), interval)]
    for i, chunk in enumerate(chunks):
        chunk.export(f"chunk_{i+1}.mp3", format="mp3")
        text_list.append(text_extraction.get_text(f"chunk_{i+1}.mp3"))

    print(text_list)
    return text_list

if __name__ =="__main__":
    if len(sys.argv) !=2:
        print("Usage: python main.py <youtube_link>")
        sys.exit(1)

    youtube_link = sys.argv[1]
    text = get_yt_text(youtube_link)
    print("result:")
    print(text)