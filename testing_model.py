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

	model = Sequential()
	for layer in vgg16_model.layers:
		model.add(layer)
	
	model.layers.pop()
	#model.layers.pop()
	#model.layers.pop()

	model.add(Dense(2, activation=None))
	
	for layer in model.layers:
		layer.trainable = True

	#model.add(Dense(2, activation='linear'))
	
	model.summary()
	return model




if __name__ == "__main__":
	import os
	import numpy as np
	from keras import optimizers
	from keras.models import Sequential
	from keras.models import load_model
	from keras.layers import Activation
	from keras.layers.core import Dense, Flatten
	from keras.optimizers import Adam, SGD
	from keras.metrics import categorical_crossentropy
	#import matplotlib.pyplot as plt
	#import matplotlib
	#matplotlib.use('Agg')
	from matplotlib import pyplot as plt
	import h5py
	from keras.utils import plot_model
	#from keras.callbacks import ModelCheckpoint
	#import utils
	#import models
	import time
	from keras.callbacks import ModelCheckpoint

	np.random.seed(7) # for reproducibility

	batch_size = 56

	#model = load_model('vgg16_edit.h5')
	model = vgg16()
	model.load_weights('trained_model_weights_dense_trainable_sgd_pose11-20.h5')

	y_filename ='../../code-new/binary/data_8k.txt'
	y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1])

	y_data_train = y_data[:]

	#########################################

	h5f = h5py.File('images_in_h5_format_8k.h5','r')
	x_data_train = h5f['dataset_1'][:]



	# ======================================================================                     
	# Configure the training process:
	print('Preparing training ...')
	#adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	sgd = SGD(lr=0.00001, momentum=0.9, decay=0.00138, nesterov=False)	
	#adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
	model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
	
	#update
	'''
	filepath="best_model.hdf5"	
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint] 
	'''

	iter=3
	# Train:
	print('Start training ...')
	start = time.time()
	'''
	history = model.fit(x = x_data_train, y = y_data_train,
		  epochs=iter,
		  batch_size=batch_size, validation_split = 0.20, shuffle = True, verbose = 1)  
		  #By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
	'''	  
	end = time.time()
	print ("Model took %0.2f seconds to train"%(end - start))
	#img_main = [np.newaxis,...]  # dimension added to fit input size
	history = model.evaluate(x_data_train, y_data_train, verbose=1)
	#model.save_weights('trained_model_weights_dense_trainable_sgd_pose1-200.h5')
	#model.save('trained_model.h5')
	
	print(history)
	print(history.history.keys()) 

	plt.figure(1)  

	# summarize history for accuracy  

	plt.subplot(211)  
	plt.plot(history.history['acc'])  
	plt.plot(history.history['val_acc'])  
	plt.title('model accuracy')  
	plt.ylabel('accuracy')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'validation'], loc='upper left')  

	# summarize history for loss  

	plt.subplot(212)  
	plt.plot(history.history['loss'])  
	plt.plot(history.history['val_loss'])  
	plt.title('model loss')  
	plt.ylabel('loss')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'validation'], loc='upper left')  
	#plt.show()
	plt.savefig('visualization_dense_trainable_sgd_pose1-200.png')

	#update
	'''
	loss_history = history
	numpy_loss_history = np.array(loss_history)
	np.savetxt("history.txt", numpy_loss_history, delimiter=",")
	'''
