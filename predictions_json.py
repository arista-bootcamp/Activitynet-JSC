import os
import sys
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data
import model
import utils

params = utils.yaml_to_dict('config.yml')

with open(params['json_metadata_path']) as data_file:
    metadata_json = json.load(data_file)

label_by_idx = {
    'level_3': {},
    'level_2': {},
    'level_1': {},
    'level_0': {}
}

for key, item in metadata_json.items():
    if key != 'classes_amount':
        label_by_idx['level_3'][item['idx']] = key
        for level in ['level_2', 'level_1', 'level_0']:
            label_by_idx[level][item[level]['idx']] = item[level]['name']

data_gen_test = data.DataGenerator(params, 'validation')

estimator = tf.estimator.Estimator(
    # Custom model function
    model_fn=model.model_fn,
    params=params,
    # Model directory
    model_dir=params['model_dir'],
    # warm_start_from=cfg.PRE_TRAIN,
    config=tf.estimator.RunConfig(
        keep_checkpoint_max=params['keep_checkpoint_max'],
        save_checkpoints_steps=params['save_checkpoints_steps'],
        save_summary_steps=params['save_summary_steps'],
        log_step_count_steps=params['log_step_count_steps']
    )
)

predictions = estimator.predict(
    input_fn = lambda: data.input_fn(data_gen_test, False, params)
)

prediction_results = {
    "results": {}
}
available_formats = ['.mkv', '.webm', '.mp4']
predictions_by_video = {}

for item in predictions:
    video_id = item['metadata'].decode('utf-8')
    batch_num = int(video_id.split('batch')[-1].replace('_', ''))
    video_id = video_id.split('batch')[0].replace('_', '')
    if video_id not in prediction_results['results']:
        prediction_results['results'][video_id] = {}

    for vformat in available_formats:
        video_path = os.path.join(params['videos_folder'] + '/validation', video_id + vformat)
        if os.path.isfile(video_path):
            break

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_difference = (params['skip_frames'] / fps)
    frame_id = 0

    classes_pred = np.ones(item['probabilities'].shape[0]) * item['classes']
    classes_score = np.ones(item['probabilities'].shape[0]) * item['score']

    for frame_pred in classes_pred:
        frame_number = (params['batch_size'] * (batch_num - 1) + frame_id) * params['skip_frames']
        frame_seconds = frame_number / fps
        if frame_pred != 0:
            if frame_pred not in prediction_results['results'][video_id]:
                prediction_results['results'][video_id][frame_pred] = {
                    "seconds_array": [frame_seconds],
                    "score": item['score']
                }
            else:
                prediction_results['results'][video_id][frame_pred]["seconds_array"].append(frame_seconds)
        frame_id += 1

prediction_format = {
    "results": {}
}

for key, item in prediction_results['results'].items():
    prediction_format['results'][key] = []
    for classes_pred, classes_pred_info in item.items():
        sec_arr = np.array(classes_pred_info['seconds_array'])
        sec_arr = np.sort(sec_arr)
        ini_sec = sec_arr[0]
        sequences = [[ini_sec, ini_sec]]
        for idx in range(1, sec_arr.shape[0]):
            if sec_arr[idx] - ini_sec > frame_difference:
                sequences.append([ini_sec, sec_arr[idx]])
                to_insert = {
                    'score': classes_pred_info['score'].item(),
                    'segment': [ini_sec.item(), sec_arr[idx].item()],
                    'label': label_by_idx['level_' + str(params['taxonomy_level'])][int(classes_pred)]
                }
                if key not in prediction_format['results']:
                    prediction_format['results'][key] = [to_insert]
                else:
                    prediction_format['results'][key].append(to_insert)
            else:
                sequences[-1][1] = sec_arr[idx]
            ini_sec = sec_arr[idx]

with open('data/predicted_output.json', 'w') as outfile:
    json.dump(prediction_format, outfile)
