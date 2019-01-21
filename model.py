import os

import tensorflow as tf

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model


def _initialize_pretrained_model(base_model_layer='conv_7b'):
	base_model = InceptionResNetV2(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(base_model_layer).output)
	return model


def gap_module(inputs):

	x = tf.layers.conv2d(inputs=inputs, kernel_size=[5, 5], filters=101,
		padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
		bias_initializer=tf.zeros_initializer()
		)

	x = tf.layers.batch_normalization(x, training=True, momentum=0.99, epsilon=0.001, center=True, scale=True )

	x = tf.nn.sigmoid(x)

	x = tf.math.reduce_mean(x, axis=[1, 2])

	return x


def dense_module(inputs):

	inputs = tf.convert_to_tensor(inputs)

	x = tf.layers.conv2d(inputs=inputs, kernel_size=[5, 5], filters=101,
		padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
		bias_initializer=tf.zeros_initializer()
		)

	x = tf.layers.batch_normalization(x, training=True, momentum=0.99,
		epsilon=0.001, center=True,scale=True
		)

	x = tf.nn.sigmoid(x)

	x = tf.reshape(x, [-1, 5 * 5 * 101])

	x = tf.layers.dense(inputs=x, units=101, activation=tf.nn.relu)

	x = tf.layers.dense(inputs=x, units=101)

	return x


def model_fn(features, mode, params):
	
	pretrain_model = _initialize_pretrained_model()
	inputs = pretrain_model.predict(features['inputs'])

	if params['model'] == 'gap':
		logits = gap_module(inputs)
	else:
		logits = dense_module(inputs)

	y_pred = tf.argmax(input=logits, axis=1)
	predictions = {
		"classes": y_pred,
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=features['labels'],
										   logits=logits)

	if mode == tf.estimator.ModeKeys.TRAIN:

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		with tf.control_dependencies(update_ops):

			optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

			train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

		tf.summary.scalar('total_loss', tf.losses.get_total_loss())

		tf.summary.scalar('accuracy', tf.metrics.accuracy(
			tf.argmax(input=features['labels'], axis=1), y_pred)[1])

		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {"accuracy": tf.metrics.accuracy(
		labels=tf.argmax(input=features['labels'], axis=1),
		predictions=predictions["classes"])}

	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def fm_model_fn(features, labels, mode, params):

	if params['model'] == 'gap':
		logits = gap_module(features)
	else:
		logits = dense_module(features)

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
