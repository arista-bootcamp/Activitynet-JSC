import cv2
#import imageio
import numpy as np
from IPython import display

VIDEO_PATH = 'videoplayback.mp4'


# Load video and yield frames
def load_video(path, max_frames=0, resize=(224, 224)):
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


# Generate GIF from video frames
def animate(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=30)
    with open('./animation.gif', 'rb') as f:
        display.display(display.Image(data=f.read(), height=300))

