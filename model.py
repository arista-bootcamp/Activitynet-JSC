import os

import tensorflow as tf

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model

def _initialize_pretrained_model(base_model_layer='conv_7b'):

	base_model = InceptionResNetV2(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(base_model_layer).output)
	return model

def _pretrained_one(inputs):

	inputs = tf.convert_to_tensor(inputs)

	x = tf.layers.conv2d(inputs=inputs, kernel_size=[5, 5], filters= 100, 
		padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
		bias_initializer=tf.zeros_initializer()
		)

	x = tf.layers.batch_normalization(x, training=True, momentum=0.99,
		epsilon=0.001, center=True,scale=True
		)

	x = tf.nn.sigmoid(x)

	x = tf.math.reduce_mean(x,axis=[1,2])	

	return x

def _pretrained_two(inputs):

	inputs = tf.convert_to_tensor(inputs)

	x = tf.layers.conv2d(inputs=inputs, kernel_size=[5, 5], filters= 100, 
		padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
		bias_initializer=tf.zeros_initializer()
		)

	x = tf.layers.batch_normalization(x, training=True, momentum=0.99,
		epsilon=0.001, center=True,scale=True
		)

	x = tf.nn.sigmoid(x)

	x = tf.reshape(x, [-1, 5 * 5 * 100])

	x = tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu)

	x = tf.layers.dense(inputs=x, units=100, activation=tf.nn.softmax)


	return x
