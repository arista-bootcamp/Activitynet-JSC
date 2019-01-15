import cv2
import os
import random
import numpy as np
from IPython import display

# Load video and yield frames
def load_video(path, max_frames=15, resize=(224, 224)):

    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                yield np.array(frames) / 255.0
                frames = []
                continue
    finally:
        cap.release()
    yield np.array(frames) / 255.0


def all_data_videos(params):
    
    list_videos = os.listdir(params['videos_folder'])

    if params['shuffle']:
        random.shuffle(list_videos)

    for video in list_videos:
        a = load_video(video)

    return a


# Generate GIF from video frames
def animate(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=30)
    with open('./animation.gif', 'rb') as f:
        display.display(display.Image(data=f.read(), height=300))

