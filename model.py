import os

import tensorflow as tf

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model


def _initialize_pretrained_model(base_model_layer='conv_7b'):
	base_model = InceptionResNetV2(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(base_model_layer).output)
	return model

def _add_fc_layer(inputs,):

	tf.layers.dense(inputs=inputs,units=4096)

	net = tf.layers.batch_normalization(net, training=training)
	net = tf.nn.elu(net)



def _add_regular_rnn_layers(inputs,params):

	rnn_layers = [tf.nn.rnn_cell.LSTMCell(node) for node in range(0,params['num_nodes'])]
	multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
	outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,inputs=[inputs[0],20,inputs[1]],dtype=tf.float32)

	return outputs;

def model_fn(features, mode, params):

	inputs = features['frames_batch']
	labels = features['labels_batch']
	
	print('************INPUTS.SHAPE*****',inputs.shape)
	print('************LOGITSS.SHAPE*****',logits.shape)

	logits = _add_regular_rnn_layers(inputs,params)

	y_pred = tf.argmax(input=logits, axis=1)
	predictions = {
		"classes": y_pred,
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

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
