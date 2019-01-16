import cv2
import os
import json
import utils
import random
import imageio
import numpy as np
import tensorflow as tf


# Load video and yield frames
def load_video(path, json_data_path, json_metadata_path, classes_amount,
               max_frames=0, resize=(224, 224), skip_frames=None):
    with open(json_data_path) as data_file:
        data_json = json.load(data_file)

    with open(json_metadata_path) as data_file:
        metadata_json = json.load(data_file)

    video_id = path.split('/')[-1].split('.')[0]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frames = []
    labels = []
    try:
        while True:
            ret, frame = cap.read()
            frame_count += 1
            if not ret:
                break

            if skip_frames:
                if frame_count % skip_frames == 1:
                    label = np.zeros(classes_amount)
                    label[0] = 1
                    seconds = frame_count / fps
                    for item in data_json['database'][video_id]['annotations']:
                        if seconds < item['segment'][1] and seconds > item['segment'][0]:
                            label[metadata_json[item['label']]['idx']] = 1
                            label[0] = 0
                            break

                    frame = cv2.resize(frame, tuple(resize))
                    frame = frame[:, :, [2, 1, 0]]
                    frames.append(frame)
                    labels.append(label)
            else:
                label = np.zeros(classes_amount)
                label[0] = 1
                seconds = frame_count / fps
                for item in data_json['database'][video_id]['annotations']:
                    if seconds < item['segment'][1] and seconds > item['segment'][0]:
                        label[metadata_json[item['label']]['idx']] = 1
                        label[0] = 0
                        break

                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
                labels.append(label)

            if len(frames) == max_frames:
                yield (np.array(frames) / 255.0, np.array(labels))
                frames = []
                labels = []

    finally:
        cap.release()

    yield (np.array(frames) / 255.0, np.array(labels))


def all_data_videos(params, mode='training'):

    list_videos = os.listdir(params['videos_folder'] + '/' + mode)

    if params['shuffle']:
        random.shuffle(list_videos)

    for video in list_videos:
        try:
            frames_video = load_video(os.path.join(params['videos_folder'],
                                                   mode + '/' + video),
                                      params['json_data_path'],
                                      params['json_metadata_path'],
                                      params['classes_amount'],
                                      resize=params['resize'],
                                      skip_frames=params['skip_frames'],
                                      max_frames=1)

            batch = next(frames_video)
            batch_1 = batch[0].reshape(params['resize'][0],
                                       params['resize'][1],
                                       3)

            batch_2 = batch[1].reshape(101)

            yield (batch_1, batch_2)

        except StopIteration:
            print('Load next video')
            continue


def animate(images, name):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./data/' + name + '.gif', converted_images, fps=30)


def input_fn(data_gen, train, params):
    data_set = tf.data.Dataset.from_generator(
        generator=data_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=((params['resize'][0], params['resize'][1], 3),
                       (params['classes_amount']))
    )

    if train:
        # data_set = data_set.shuffle(buffer_size=cfg.SHUFFLE_BUFFER)
        data_set = data_set.repeat(params['num_epochs'])

    data_set = data_set.batch(params['batch_size'])

    iterator = data_set.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()

    features = dict(inputs=images_batch, labels=labels_batch)

    return features


if __name__ == '__main__':
    params = utils.yaml_to_dict('config.yml')
    video_gen = all_data_videos(params)
    a = input_fn(lambda: video_gen, True, params)
