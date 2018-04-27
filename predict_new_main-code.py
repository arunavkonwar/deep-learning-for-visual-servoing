'''http://marubon-ds.blogspot.fr/2017/08/how-to-make-fine-tuning-model.html
'''

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
	from keras import models
	from keras import layers


	resnet = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), classes=1000)

	model = models.Sequential()
	model.add(resnet)
	model.add(layers.Flatten())
	model.add(Dense(2, activation=None))
	
	
	layer_num = len(resnet.layers)
	
	for layer in resnet.layers[:int(layer_num * 0.9)]:
        	layer.trainable = False
	model_num = len(model.layers)
	'''
	for layer in model.layers[int(model_num * 0.8):]:
        	layer.trainable = True
        '''	
	
	model.summary()
	resnet.summary()
	print len(resnet.layers)
	print(layer_num * 0.8)
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
	from sklearn.metrics import mean_squared_error




	model = vgg16()
	model.load_weights('trained_model_resnet50_90percent_1-50_adam_001.h5')
	'''
	images = skimage.io.imread('/home/arunav/test/open-close/holly/1000.jpg') 
	images = skimage.exposure.rescale_intensity(images * 1.0, out_range=np.float32) 
	images = np.stack((images,)*3, -1)
	'''
	#root_dir = "/home/arunav/code-new/binary/validation_images_8k"
	root_dir = "/home/arunav/test/open-close/holly"
	names = np.array(glob.glob(os.path.join(root_dir, "*.jpg")))
	indexes = np.array([int(os.path.splitext(os.path.basename(name))[0]) for name in names])
	argsort_indexes = np.argsort(indexes)
	names = names[argsort_indexes]
	#names = names[:10]
	print(names)

	images = [skimage.io.imread(name) for name in names]
	images = [skimage.exposure.rescale_intensity(image * 1.0, out_range=np.float32) for image in images]
	images = np.stack((images,)*3, -1)
	
	#img_main = images [np.newaxis,...]  # dimension added to fit input size
	y_filename ='/home/arunav/main-code/arunav/data_8k.txt'
	
	y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1])
	y_data_train = y_data[:]
	
	
	#print(img_main.shape)
	fig=plt.figure(figsize=(8, 8))
	for i in range(8):
		start = time.time()
		img_main=images[i] [np.newaxis,...]
		prediction = model.predict(img_main)
		end = time.time()
		print ("\n\n****************************\n\nModel took %0.2f seconds to predict\n\n****************************\n"%(end - start))
		print (names[i])
		print (prediction[0])
		print ("ground truth")
		print y_data_train[i*1000 + 999]
		print ("loss")
		print (mean_squared_error(prediction[0], y_data_train[i*1000 + 999]))
		img = images[i]	
		fig.add_subplot(3, 3, i+1)
		plt.ylabel(mean_squared_error(prediction[0], y_data_train[i*1000 + 999]))
		plt.imshow(img)
	plt.show()
		
	'''
	for i in range(1800):
		start = time.time()
		img_main=images[i] [np.newaxis,...]
		prediction = model.predict(img_main)
		end = time.time()
		print ("\n\n****************************\n\nModel took %0.2f seconds to predict\n\n****************************\n"%(end - start))
		print (names[i])
		
		text_file = open("compare.txt", "a")
		if(os.path.getsize("compare.txt") > 0):
			text_file.write("\n")
			text_file.write('%s' % prediction[0][0])
			text_file.write(" ")
			text_file.write('%s' % prediction[0][1])	
		else:
			text_file.write('%s' % prediction[0][0])
			text_file.write(" ")
			text_file.write('%s' % prediction[0][1])
			
		text_file.close()
	'''
		








