import os
import sys
import numpy as np
import requests
import json
import youtube_dl
from os import listdir

with open('./data/activity_net.v1-2.min.json') as data_file:
    data = json.load(data_file)

count_training = 0
count_testing = 0
count_validation = 0
path = './data/videos'

if not os.path.exists(path):
    os.makedirs(path)

for key, item in data["database"].items():
    count_training += item['subset'] == 'training'
    count_testing += item['subset'] == 'testing'
    count_validation += item['subset'] == 'validation'

    if item['subset'] == 'testing':
        continue

    if count_training == 1500:
        break

    if not os.path.exists(path + '/' + item['subset']):
        os.makedirs(path + '/' + item['subset'])

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
