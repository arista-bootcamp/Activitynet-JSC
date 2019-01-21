import os
import sys
import numpy as np
import requests
import json
from pytube import YouTube
import youtube_dl
import cv2
from os import listdir

with open('./data/activity_net.v1-2.min.json') as data_file:
    data = json.load(data_file)

count_training = 0
count_testing = 0
count_validation = 0
path = './data/videos'

for key, item in data["database"].items():
    count_training += item['subset'] == 'training'
    count_testing += item['subset'] == 'testing'
    count_validation += item['subset'] == 'validation'

    if count_training == 1000:
        break

    if not os.path.exists(path + '/' + item['subset']):
        os.makedirs(path)

    if key + '.mkv' in os.listdir(path + '/' + item['subset']):
        print('video already downloaded')
        continue

    options = {
        'outtmpl': path + '/' + item['subset'] + '/' + key,
        'quiet': True
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        try:
            ydl.download([item['url']])
        except:
            continue
