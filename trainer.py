#!/usr/bin/env python
"""
Tensorflow YOLO Implementation Training and Evaluation
"""

import config as cfg
import tensorflow as tf

import model
import utils

tf.logging.set_verbosity(tf.logging.INFO)

def main(params):

	#data_gen = Generator()
	#data_gen_test = Generator(mode='test')

	# Instantiate the Estimator API class in Tensoflow.
    estimator = tf.estimator.Estimator(
        # Custom model function
        model_fn=model.model_fn,
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
        lambda: input_fn(data_gen, True, params),
        max_steps=params['max_steps']
    )
    
    eval_spec = tf.estimator.EvalSpec(
        lambda: input_fn(data_gen_test, False, params),
        steps=params['eval_steps'],
        start_delay_secs=params['start_delay_secs'],
		throttle_secs=params['throttle_secs']
	)

	tf.logging.info("Start experiment....")

	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    #estimator.export_savedmodel(export_dir_base=params['model_dir'],serving_input_receiver_fn=serving_input_fn)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="path to configuration file")
    parser.add_argument('-v', '--verbosity', default='INFO',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARM'],
    )

    args = parser.parse_args()
    tf.logging.set_verbosity(args.verbosity)

    params = utils.yaml_to_dict(args.config)

    tf.logging.info("Using parameters: {}".format(params))

	main(params)