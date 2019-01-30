import os
import tensorflow as tf
import numpy as np
import json


class MyStr(str):
    def __lt__(self, b):
        a = self.split('.')[0]
        b = b.split('.')[0]
        prefix_a = '_'.join([a.split('_')[0], a.split('_')[1]])
        prefix_b = '_'.join([b.split('_')[0], b.split('_')[1]])
        if prefix_a == prefix_b:
            return int(a.split('_')[-1]) < int(b.split('_')[-1])
        else:
            return prefix_a < prefix_b

    def __gt__(self, b):
        a = self.split('.')[0]
        b = b.split('.')[0]
        prefix_a = '_'.join([a.split('_')[0], a.split('_')[1]])
        prefix_b = '_'.join([b.split('_')[0], b.split('_')[1]])
        if prefix_a == prefix_b:
            return int(a.split('_')[-1]) < int(b.split('_')[-1])
        else:
            return prefix_a > prefix_b


def mfunc(x):
    return MyStr(x)


class DataWindowGenerator:
    """ Reads several images and returns sliding windows.
    """
    def __init__(self, params, mode='training'):

        self.mode = mode
        self.feature_map_dir = os.path.join(params['feature_map_folder'], mode)
        self.params = params
        self.feature_map_list = map(mfunc, os.listdir(self.feature_map_dir))
        self.feature_map_list = sorted(self.feature_map_list)

    def __iter__(self):
        for item in self.feature_map_list:
            try:
                for idx in range(0, self.params['batch_size']):
                    feature_map_path = os.path.join(self.feature_map_dir, item)
                    images, labels = _load_feature_map_from_npz(feature_map_path)

                    batch_num = int(item.split('.')[0].split('_')[-1])
                    video_name = item.split('.')[0].split('batch')[0][:-1]

                    available_formats = ['.mkv', '.webm', '.mp4']
                    vformat = None
                    for vformat in available_formats:
                        video_path = os.path.join(
                            self.params['videos_folder'], self.mode, video_name + vformat)
                        if os.path.isfile(video_path):
                            break

                    with open(self.params['fps_metadata']) as data_file:
                        video_metadata_json = json.load(data_file)

                    if video_name not in video_metadata_json:
                        break

                    fps = video_metadata_json[video_name]['fps']

                    metadata = {
                        'video_id': video_name + vformat
                    }
                    if idx + self.params['window_size'] > images.shape[0]:
                        next_fmap_name = video_name + '_' + 'batch_' + str(
                            batch_num + 1) + '.npz'
                        if not os.path.isfile(os.path.join(
                                self.feature_map_dir, next_fmap_name)):
                            break

                        images_next, labels_next = _load_feature_map_from_npz(
                            feature_map_path)
                        offset = self.params['window_size'] - (images.shape[0] - idx)
                        images = np.concatenate((images[idx:images.shape[0], :],
                                                 images_next[0:offset, :]), axis=0)
                        labels = np.concatenate((labels[idx:images.shape[0], :],
                                                 labels_next[0:offset, :]), axis=0)
                        frame_number_ini = (
                            self.params['batch_size'] * (batch_num - 1) + idx) * 6
                        frame_number_end = (
                            self.params['batch_size'] * batch_num + offset - 1) * 6
                        metadata['segment'] = [frame_number_ini / fps,
                                               frame_number_end / fps]

                    else:
                        images = images[idx:idx+self.params['window_size'], :]
                        labels = labels[idx:idx+self.params['window_size'], :]
                        frame_number_ini = (
                            self.params['batch_size'] * (batch_num - 1) + idx) * 6
                        frame_number_end = (
                            self.params['batch_size'] * (
                                batch_num - 1) + idx + self.params['window_size'] - 1) * 6
                        metadata['segment'] = [frame_number_ini / fps, frame_number_end / fps]

                    images = np.reshape(images, (15, -1))
                    if images.shape[1] < 38400:
                        continue
                    yield images, labels, metadata['video_id'], metadata['segment'][0], metadata['segment'][1]

            except TypeError:
                pass

    def __call__(self):
        return self


class DataGenerator:
    """Reads an image.
    Args:
    Returns:
        A tuple with the following objects (img_vol, output_volume)
    """

    def __init__(self, params, mode='training'):

        self.mode = mode
        self.feature_map_dir = os.path.join(params['feature_map_folder'], mode)
        self.params = params

        self.feature_map_list = os.listdir(self.feature_map_dir)

    def __iter__(self):
        for item in self.feature_map_list:
            images = labels = video_id = None
            try:
                feature_map_path = os.path.join(self.feature_map_dir, item)
                images, labels = _load_feature_map_from_npz(feature_map_path)
                images = np.reshape(images, (15, -1))
                video_id = item.split('.')[0]
                # data_frame_label = _concat_frames_in_volume(data_frame_label)
            except TypeError:
                pass

            if images.shape[1] < 38400:
                continue

            yield images, labels, video_id, 0, 0

    def __call__(self):
        return self


def _load_feature_map_from_npz(feature_map_path):
    feature_map = np.load(feature_map_path)
    return feature_map['feature_map'], feature_map['label']


def _concat_frames_in_volume(data_frame_label):
    feature_map = data_frame_label[0]
    label = data_frame_label[1]

    f, h, w, c = feature_map.shape
    feature_map = np.resize(feature_map, (h, w, c * f))

    f, c = label.shape
    label = np.resize(label, (c * f))

    return feature_map, label


def input_fn(data_gen, train, params):
    h, w, c = params['feature_maps_size']
    m = params['classes_amount']
    f = params['window_size']

    data_set = tf.data.Dataset.from_generator(
        generator=data_gen,
        output_types=(tf.float32, tf.float32, tf.string, tf.float32, tf.float32),
        output_shapes=((f, h * w * c), (f, m), (), (), ())
    )

    if train:
        data_set = data_set.shuffle(buffer_size=params['buffer_size'])
        data_set = data_set.repeat(params['num_epochs'])

    data_set = data_set.batch(params['batch_size'])

    iterator = data_set.make_one_shot_iterator()
    frames_batch, labels_batch, metadata, ini, end = iterator.get_next()

    features = dict(frames_batch=frames_batch, labels_batch=labels_batch,
                    metadata=metadata, ini=ini, end=end)

    return features


def serving_input_fn(params):
    inputs = {'frames_batch': tf.placeholder(tf.float32, [None,
                                                          params['max_frames'][0],
                                                          params['feature_maps_size'][0] *
                                                          params['feature_maps_size'][1] *
                                                          params['feature_maps_size'][2]]
                                             )}

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
