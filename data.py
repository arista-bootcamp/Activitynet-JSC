import os
import tensorflow as tf
import numpy as np


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

            yield images, labels, video_id

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
    f = params['max_frames'][0]

    data_set = tf.data.Dataset.from_generator(
        generator=data_gen,
        output_types=(tf.float32, tf.float32, tf.string),
        output_shapes=((f, h * w * c), (f, m), ())
    )

    if train:
        data_set = data_set.shuffle(buffer_size=params['buffer_size'])
        data_set = data_set.repeat(params['num_epochs'])

    data_set = data_set.batch(params['batch_size'])

    iterator = data_set.make_one_shot_iterator()
    frames_batch, labels_batch, video_id = iterator.get_next()

    features = dict(frames_batch=frames_batch, labels_batch=labels_batch,
                    metadata=video_id)

    return features


def serving_input_fn(params):
    inputs = {'frames_batch': tf.placeholder(tf.float32, [None,
                                                          params['max_frames'][0],
                                                          params['feature_maps_size'][0] *
                                                          params['feature_maps_size'][1] *
                                                          params['feature_maps_size'][2]]
                                             )}

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
