import cv2
import os
import utils
import imageio
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
                    frame = cv2.resize(frame, tuple(resize))
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

    if params['shuffle']:
        random.shuffle(list_videos)

    for video in list_videos:
        try:
            frames_video = load_video(os.path.join(params['videos_folder'], video),
                                      resize=params['resize'],
                                      skip_frames=params['skip_frames'])

            yield frames_video
            
        except StopIteration:
            print('Load next video')
            continue


def animate(images, name):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./data/' + name + '.gif', converted_images, fps=30)


if __name__ == '__main__':
    params = utils.yaml_to_dict('config.yml')
    video_gen = all_data_videos(params)
    animate(next(video_gen), 'video1')

