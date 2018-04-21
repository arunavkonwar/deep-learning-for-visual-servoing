def vgg16():
	import keras
	from keras.models import Sequential
	from keras.layers import Activation
	from keras.layers.core import Dense, Flatten
	from keras.optimizers import Adam
	from keras.metrics import categorical_crossentropy
	from keras.layers.normalization import BatchNormalization
	from keras.layers.convolutional import *
	import matplotlib.pyplot as plt
	from keras.utils import plot_model 
	from keras.applications import VGG16
	from keras import models
	from keras import layers


	#vgg16_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

	model = models.Sequential()
	model.add(conv_base)
	model.add(layers.Flatten())
	model.add(layers.Dense(4096, activation='relu'))
	model.add(layers.Dense(4096, activation='relu'))
	model.add(layers.Dense(2, activation=None))
	

	conv_base.trainable = True

	set_trainable = False
	for layer in conv_base.layers:
		if layer.name == 'block5_conv1':
			set_trainable = True
		if set_trainable:
			layer.trainable = True
		else:
			layer.trainable = False
	
	
	model.summary()
	print "length of the network:"
	print len(model.layers)
	return model
	


	



if __name__ == "__main__":
	import os
	import numpy as np
	from keras import optimizers
	from keras.models import Sequential
	from keras.models import load_model
	from keras.layers import Activation
	from keras.layers.core import Dense, Flatten
	from keras.optimizers import Adam
	from keras.metrics import categorical_crossentropy
	import matplotlib.pyplot as plt
	import h5py
	from keras.utils import plot_model
	import time
	from keras.callbacks import ModelCheckpoint
	import cv2
	
	import glob
	import skimage.io
	import skimage.exposure

	import matplotlib.pyplot as plt


	
	
	
	np.random.seed(7) # for reproducibility


	model = vgg16()
	model.load_weights('trained_model_works.h5')

	images = skimage.io.imread('/home/arunav/test/open-close/holly/1000.jpg') 
	images = skimage.exposure.rescale_intensity(images * 1.0, out_range=np.float32) 
	images = np.stack((images,)*3, -1)
	
	img_main = images [np.newaxis,...]  # dimension added to fit input size



	start = time.time()
	prediction = model.predict(img_main)
	end = time.time()
	print ("\n\n****************************\n\nModel took %0.2f seconds to predict\n\n****************************\n"%(end - start))
	print(prediction)








