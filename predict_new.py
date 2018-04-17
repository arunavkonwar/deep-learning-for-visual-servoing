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


	vgg16_model = keras.applications.vgg16.VGG16()

	#vgg16_model.summary()


	model = Sequential()
	for layer in vgg16_model.layers:
		model.add(layer)
	
	model.layers.pop()
	
	model.add(Dense(2, activation=None))
	
	for layer in model.layers:
		layer.trainable = True
	
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
	#from keras.callbacks import ModelCheckpoint
	#import utils
	#import models
	import time
	from keras.callbacks import ModelCheckpoint
	import cv2
	
	np.random.seed(7) # for reproducibility



	batch_size = 14

	#model = load_model('vgg16_edit.h5')
	model = vgg16()
	model.load_weights('trained_model_sgd_valid_40k_1-60.h5')

	img = cv2.imread('2134.jpg')
	img_1 = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
	img_main = img_1 [np.newaxis,...]  # dimension added to fit input size



	start = time.time()
	prediction = model.predict(img_main)
	end = time.time()
	print ("\n\n****************************\n\nModel took %0.2f seconds to predict\n\n****************************\n"%(end - start))
	print(prediction)








