import os
import tensorflow as tf

import numpy as np

class DataGenerator:
    """Reads an image.
    Args:
    Returns:
        A tuple with the following objects (img_vol, output_volume)
    """

    def __init__(self,params, mode='train'):

        self.mode = mode
        self.feature_map_dir = os.path.join(params['feature_map_folder'],mode)
        self.params = params

        self.feature_map_list = os.listdir(self.feature_map_dir)


    def __iter__(self):
        for item in self.feature_map_list:
            try:
                feature_map_path = os.path.join(self.feature_map_dir,item)
                data_frame_label = _load_feature_map_from_npz(feature_map_path)
            except TypeError:
                pass
            yield data_frame_label
        
    def __call__(self):
        return self


def _load_feature_map_from_npz(feature_map_path,):

    feature_map = np.load(feature_map_path)
    return(feature_map['feature_map'],feature_map['label'])


def input_fn(data_gen,params,train):

    frames_expected_output =  np.concatenate((params['max_frames'],params['feature_maps_size']))
    labels_expected_output =  np.concatenate((params['max_frames'],params['label_feature_maps_size']))

    data_set = tf.data.Dataset.from_generator(
        generator=data_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=(frames_expected_output,
                       labels_expected_output)
    )

    if train:
        data_set = data_set.shuffle(buffer_size=params['buffer_size'])
        data_set = data_set.repeat(params['num_epochs'])

    data_set = data_set.batch(params['batch_size'])

    iterator = data_set.make_one_shot_iterator()
    frames_batch, labels_batch = iterator.get_next()

    features = dict(frames_batch=frames_batch, labels_batch=labels_batch)

    return features