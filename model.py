import os

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model

def _initialize_pretrained_model(base_model_layer='conv_7b'):

	base_model = InceptionResNetV2(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(base_model_layer).output)
	return model