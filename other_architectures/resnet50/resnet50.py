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


	#vgg16_model = keras.applications.vgg16.VGG16()
	inception_model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	
	'''
	model = Sequential()
	for layer in vgg16_model.layers:
		model.add(layer)
	
	#model.layers.pop()
	
	#model.add(Dense(2, activation=None))
	
	for layer in model.layers:
		layer.trainable = True
	'''
	inception_model.summary()
	return inception_model




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

	np.random.seed(7) # for reproducibility



	batch_size = 14

	#model = load_model('vgg16_edit.h5')
	model = vgg16()
	#model.load_weights('trained_model_weights.h5')

	y_filename ='../../code-new/binary/data.txt'
	y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1])

	y_data_train = y_data[:]

	#########################################

	h5f = h5py.File('images_in_h5_format.h5','r')
	x_data_train = h5f['dataset_1'][:]



	# ======================================================================                     
	# Configure the training process:
	print('Preparing training ...')
	#adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#adam = Adam(lr=0.0001)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])


	# Train:
	print('Start training ...')
	start = time.time()
	hist = model.fit(x = x_data_train, y = y_data_train,
		  epochs=1,
		  batch_size=batch_size, validation_split = 0.20, shuffle = True, verbose = 1)  
		  #By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
	end = time.time()
	print ("Model took %0.2f seconds to train"%(end - start))


	model.save_weights('trained_model_weights.h5')
	model.save('trained_model.h5')

	train_loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	train_acc = hist.history['acc']
	val_acc = hist.history['val_acc']
	
	xc = range(epochs)
	
	plt.figure(1,figsize=(7,5))
	plt.plot(xc,train_loss)
	plt.plot(xc,val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train','val'])
	print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])
	plt.savefig('train_loss vs val_loss.png')

	plt.figure(2,figsize=(7,5))
	plt.plot(xc,train_acc)
	plt.plot(xc,val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)
	plt.legend(['train','val'],loc=4)
	#print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])
	plt.savefig('train_acc vs val_acc')








