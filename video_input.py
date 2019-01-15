import cv2
import os
import random
import numpy as np


# Load video and yield frames
def load_video(path, max_frames=0, resize=(224, 224), skip_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    count_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if skip_frames:
                if count_frames % skip_frames == 0:
                    frame = cv2.resize(frame, resize)
                    frame = frame[:, :, [2, 1, 0]]
                    frames.append(frame)
            else:
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

            if len(frames) == max_frames:
                yield np.array(frames) / 255.0
                frames = []

            count_frames += 1
    finally:
        cap.release()
    yield np.array(frames) / 255.0


def all_data_videos(params):
    
    list_videos = os.listdir(params['videos_folder'])
    a = None

    if params['shuffle']:
        random.shuffle(list_videos)

    for video in list_videos:
        a = load_video(video)

    return a
