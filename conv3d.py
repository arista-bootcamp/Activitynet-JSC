import h5py
import json
import numpy as np
import argparse
import utils
import tensorflow as tf


def fps_estimator(duration, frames):
    return round(frames / duration)


def hdf5_gen(filename, params, mode):
    fid = h5py.File(filename, 'r')
    video_lst = list(fid.keys())

    with open(params['json_data_path']) as data_file:
        data_json = json.load(data_file)

    with open(params['json_metadata_path']) as data_file:
        metadata_json = json.load(data_file)

    for video in video_lst:
        try:
            frames = fid[video]['c3d_features'][:]
            duration = data_json['database'][video[2:]]['duration']
            subset = data_json['database'][video[2:]]['subset']
        except KeyError:
            continue

        if subset != mode:
            continue

        label = np.zeros(params['classes_amount'])
        label[0] = 1

        fps = fps_estimator(duration, frames.shape[0] * 8)

        frame_count = 0

        for frame in frames:
            frame_count += 1
            seconds = frame_count * 8 / fps
            for item in data_json['database'][video[2:]]['annotations']:
                if seconds < item['segment'][1] and seconds > item['segment'][0]:
                    label[metadata_json[item['label']]['idx']] = 1
                    label[0] = 0
                    break
            yield frame, label


def input_fn(data_gen, train, params):
    data_set = tf.data.Dataset.from_generator(
        generator=data_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=((500,), (params['classes_amount'],))
    )

    if train:
        # data_set = data_set.shuffle(buffer_size=cfg.SHUFFLE_BUFFER)
        data_set = data_set.repeat(params['num_epochs'])

    data_set = data_set.batch(params['batch_size'])

    return data_set


def gap_module(inputs):

    inputs = tf.convert_to_tensor(inputs)

    x = tf.layers.dense(inputs=inputs, units=101, activation=tf.nn.relu)

    return x


def model_fn(features, labels, mode, params):

    logits = gap_module(features)

    y_pred = tf.argmax(input=logits, axis=1)

    predictions = {
        "classes": y_pred,
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        tf.summary.scalar('total_loss', tf.losses.get_total_loss())

        tf.summary.scalar('accuracy', tf.metrics.accuracy(
            tf.argmax(input=labels, axis=1), y_pred)[1])

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
        labels=tf.argmax(input=labels, axis=1),
        predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(filename, params):
    data_gen = hdf5_gen(filename, params, 'training')
    data_gen_test = hdf5_gen(filename, params, 'validation')

    estimator = tf.estimator.Estimator(
        # Custom model function
        model_fn=model_fn,
        params=params,
        # Model directory
        model_dir=params['model_dir'],
        # warm_start_from=cfg.PRE_TRAIN,
        config=tf.estimator.RunConfig(
            keep_checkpoint_max=params['keep_checkpoint_max'],
            save_checkpoints_steps=params['save_checkpoints_steps'],
            save_summary_steps=params['save_summary_steps'],
            log_step_count_steps=params['log_step_count_steps']
        )
    )

    train_spec = tf.estimator.TrainSpec(
        lambda: input_fn(lambda: data_gen, True, params),
        max_steps=params['max_steps']
    )

    eval_spec = tf.estimator.EvalSpec(
        lambda: input_fn(lambda: data_gen_test, False, params),
        steps=params['eval_steps'],
        start_delay_secs=params['start_delay_secs'],
        throttle_secs=params['throttle_secs']
    )

    tf.logging.info("Start experiment....")

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')
    parser.add_argument('-v', '--verbosity', default='INFO',
                        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARM'],
                        )
    parser.add_argument('-d', '--data', help="path to hdf5 data file", default='data/C3D/sub_activitynet_v1-3.c3d.hdf5')

    args = parser.parse_args()
    tf.logging.set_verbosity(args.verbosity)

    params = utils.yaml_to_dict(args.config)
    tf.logging.info("Using parameters: {}".format(params))

    main(args.data, params)
