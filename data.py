import os
import numpy as np
import tensorflow as tf


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
            images = labels = None
            try:
                feature_map_path = os.path.join(self.feature_map_dir, item)
                images, labels = _load_feature_map_from_npz(feature_map_path)
                images = np.reshape(images, (15, -1))
                # data_frame_label = _concat_frames_in_volume(data_frame_label)
            except TypeError:
                pass
            yield images, labels

    def __call__(self):
        return self


def _load_feature_map_from_npz(feature_map_path):
    feature_map = np.load(feature_map_path)
    return feature_map['feature_map'], feature_map['label']


def _concat_frames_in_volume(data_frame_label):
    feature_map = data_frame_label[0]
    label = data_frame_label[1]

    F, H, W, C = feature_map.shape
    feature_map = np.resize(feature_map, (H, W, C * F))

    F, C = label.shape
    label = np.resize(label, (C * F))

    return feature_map, label


def input_fn(data_gen, train, params):
    H, W, C = params['feature_maps_size']
    L = params['classes_amount']
    F = params['max_frames'][0]

    data_set = tf.data.Dataset.from_generator(
        generator=data_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=((F, H * W * C), (F, L))
    )

    if train:
        data_set = data_set.shuffle(buffer_size=params['buffer_size'])
        data_set = data_set.repeat(params['num_epochs'])

    data_set = data_set.batch(params['batch_size'])

    iterator = data_set.make_one_shot_iterator()
    frames_batch, labels_batch = iterator.get_next()

    features = dict(frames_batch=frames_batch, labels_batch=labels_batch)

    return features
