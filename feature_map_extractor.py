import cv2
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.python import pywrap_tensorflow

import video_input as vi
import model as model
import utils as utils

params = utils.yaml_to_dict('config.yml')

mode = sys.argv[1] if len(sys.argv) > 1 else 'training'

if mode not in ['training', 'validation', 'testing']:
    print('Not a valid mode')
    sys.exit()

pretrain_model = model._initialize_pretrained_model()

all_data_videos = vi.all_data_videos(params, mode=mode)

videos_dict = dict()
current_video = 'not a video'
steps_videos = 0

if not os.path.exists(params['data_dir'] + '/feature_maps/'):
    os.makedirs(params['data_dir'] + '/feature_maps/')

if not os.path.exists(params['data_dir'] + '/feature_maps/' + mode):
    os.makedirs(params['data_dir'] + '/feature_maps/' + mode)


for data in all_data_videos:
    steps_videos += data[2] != current_video

    if data[2] != current_video:
        print('current video: ', data[2])

    current_video = data[2]

    image_frames = data[0]
    image_label = data[1]
    image_video_id = data[2]

    try:
        image_feature_map = pretrain_model.predict(image_frames)
    except:
        continue

    image_video_name = image_video_id + '_batch_1.npz'
    image_feature_path = os.path.join(
        params['data_dir'] + '/feature_maps/' + mode, image_video_name)
    batch_count = 1
    while os.path.isfile(image_feature_path):
        batch_count += 1
        image_video_name = image_video_id + '_batch_' + str(batch_count) + '.npz'
        image_feature_path = os.path.join(
            params['data_dir'] + '/feature_maps/training/', image_video_name)

    np.savez(image_feature_path, feature_map=image_feature_map, label=image_label)
