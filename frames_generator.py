import cv2
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


if __name__ == '__main__':
    VIDEO_PATH = 'data/videoplayback.mp4'
    frames_gen = load_video(VIDEO_PATH, max_frames=600, skip_frames=None)
    print(next(frames_gen).shape)
