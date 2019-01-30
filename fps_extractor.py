import os
import cv2
import json
import sys

import utils

params = utils.yaml_to_dict('config.yml')

mode = sys.argv[1] if len(sys.argv) > 1 else 'training'

if mode not in ['training', 'validation', 'testing']:
    print('Not a valid mode')
    sys.exit()

json_format = {}
for item in os.listdir(os.path.join(params['videos_folder'], mode)):
    video_path = os.path.join(os.path.join(params['videos_folder'], mode, item))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    json_format[item.split('.')[0]] = {
        'fps': fps
    }

with open(params['fps_metadata'], 'w') as outfile:
    json.dump(json_format, outfile)
