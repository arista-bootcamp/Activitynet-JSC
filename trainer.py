#!/usr/bin/env python
"""
Tensorflow ActivityNet Implementation Training and Evaluation
"""

import json
import argparse
import numpy as np
import tensorflow as tf

import model
import utils
import data as data

tf.logging.set_verbosity(tf.logging.INFO)


def main(parameters):

def main_fm(params):

    data_gen_train = data.DataGenerator(params, 'training')
    data_gen_test = data.DataGenerator(params, 'validation')

    estimator = tf.estimator.Estimator(
        # Custom model function
        model_fn=model.model_fn,
        params=parameters,
        # Model directory
        model_dir=parameters['model_dir'],
        # warm_start_from=cfg.PRE_TRAIN,
        config=tf.estimator.RunConfig(
            keep_checkpoint_max=parameters['keep_checkpoint_max'],
            save_checkpoints_steps=parameters['save_checkpoints_steps'],
            save_summary_steps=parameters['save_summary_steps'],
            log_step_count_steps=parameters['log_step_count_steps']
        )
    )

    train_spec = tf.estimator.TrainSpec(
        lambda: data.input_fn(lambda: data_gen_train, True, parameters),
        max_steps=parameters['max_steps']
    )

    eval_spec = tf.estimator.EvalSpec(
        lambda: data.input_fn(lambda: data_gen_test, False, parameters),
        steps=parameters['eval_steps'],
        start_delay_secs=parameters['start_delay_secs'],
        throttle_secs=parameters['throttle_secs']
    )

    tf.logging.info("Start experiment....")

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    estimator.export_savedmodel(export_dir_base=parameters['model_dir'],
                                serving_input_receiver_fn=lambda: data.serving_input_fn(parameters))
    """
    result = estimator.predict(lambda: data.input_fn(lambda: data_gen_test, False, parameters))
    return result
    """


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

    main(params)
    """
    data_gen_test = data.DataGenerator(params, 'testing')

    with open('data/training_meta_data_reduced.json', 'r') as f:
        metadata = json.load(f)

    metadata = {v['idx']: k for k, v in metadata.items() if k != 'classes_amount'}

    for result_batch in results:
        print('Predicted Class: ', 
              np.argmax(result_batch['classes']), 
              '-', 
              metadata[np.argmax(result_batch['classes'])])
        
        print('Predicted Prob: ', result_batch['score'])
    """
