#!/usr/bin/env python
"""
Tensorflow YOLO Implementation Training and Evaluation
"""

import tensorflow as tf
import argparse

import model
import utils
import video_input as vi
import data as data

tf.logging.set_verbosity(tf.logging.INFO)


def main(params):

    data_gen = vi.all_data_videos(params, 'training')
    data_gen_test = vi.all_data_videos(params, 'validation')

    estimator = tf.estimator.Estimator(
        # Custom model function
        model_fn=model.model_fn,
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
        lambda: vi.input_fn(lambda: data_gen, True, params),
        max_steps=params['max_steps']
    )

    eval_spec = tf.estimator.EvalSpec(
        lambda: vi.input_fn(lambda: data_gen_test, False, params),
        steps=params['eval_steps'],
        start_delay_secs=params['start_delay_secs'],
        throttle_secs=params['throttle_secs']
    )

    tf.logging.info("Start experiment....")

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # estimator.export_savedmodel(export_dir_base=params['model_dir'],serving_input_receiver_fn=serving_input_fn)


def main_fm(params):

    data_gen_train = data.DataGenerator(params,'training')
    data_gen_test = data.DataGenerator(params, 'testing')

    estimator = tf.estimator.Estimator(
        # Custom model function
        model_fn=model.model_fn,
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
        lambda: data.input_fn(lambda: data_gen_train, True, params),
        max_steps=params['max_steps']
    )

    eval_spec = tf.estimator.EvalSpec(
        lambda: data.input_fn(lambda: data_gen_test, False, params),
        steps=params['eval_steps'],
        start_delay_secs=params['start_delay_secs'],
        throttle_secs=params['throttle_secs']
    )

    tf.logging.info("Start experiment....")

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # estimator.export_savedmodel(export_dir_base=params['model_dir'],serving_input_receiver_fn=serving_input_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')
    parser.add_argument('-v', '--verbosity', default='INFO',
                        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARM'],
                        )

    args = parser.parse_args()
    tf.logging.set_verbosity(args.verbosity)

    params = utils.yaml_to_dict(args.config)

    tf.logging.info("Using parameters: {}".format(params))

    # main(params)
    main_fm(params)
