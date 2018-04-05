
def initializeNetwork():
	from keras.models import load_model
	import h5py

	initialize_model = load_model('../trained_model.h5')
	return initialize_model

	



if __name__ == '__main__':
	import os
	import numpy as np
	import keras
	from keras import backend as K
	from keras.models import Sequential
	from keras.models import load_model
	from keras.layers import Activation
	from keras.layers.core import Dense, Flatten
	from keras.optimizers import Adam
	from keras.metrics import categorical_crossentropy
	from keras.preprocessing.image import ImageDataGenerator
	from keras.layers.normalization import BatchNormalization
	from keras.layers.convolutional import *
	from sklearn.metrics import confusion_matrix
	import itertools
	import matplotlib.pyplot as plt
	import h5py
	from os import listdir
	from os.path import isfile, join
	import numpy as np
	import cv2
	import time
	import struct
	import codecs


	#mypath='./data/train' 
	model = initializeNetwork()
	print "/nModel initialized..."
	#img_1 = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

	file1Path = "/dev/shm/fifo_server"
	file2Path = "/dev/shm/fifo_client"

	print "Server Waiting for Client to connect "

	print 'Handshake...'
	f = open(file1Path, 'r')
	line = f.readline()
	#print "the line is "+ line
	
	#arraySize = struct.unpack('i', f.read(4))[0] # Reading array size
	#print 'Received int = ', arraySize
	#strLength = `arraySize`+'i' +'\n'
	
	#
	filename ='../../../code-new/binary/data1.txt'
	data = np.loadtxt(filename, delimiter=' ')
	print len(data)
	a=(data[:]).reshape(224,224)

	#mypath='./data/train' 
	#img = cv2.imread('1.jpg')
	img2 = np.zeros((224,224,3), np.uint8) #replicating the numpy array shape of another image
	img2[:,:,0] = a
	img2[:,:,1] = a
	img2[:,:,2] = a
	#img_1 = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

	print "haha"	
	print img2.shape 
	img_main = img2[np.newaxis,...]  # dimension added to fit input size
	print img_main.shape
	
	start = time.time()
	prediction = model.predict(img_main)
	end = time.time()
	print(prediction)
	print ("\n\n****************************\n\nModel took %0.2f seconds to predict\n\n****************************\n"%(end - start))
	print(prediction)
	print prediction.shape
	
	print "Returning prediction value"
	
	#file = open(file2Path,'w') 
	
	np.savetxt("result.txt",prediction,fmt = '%.6f', newline = ' ')
	#data = np.loadtxt('result.txt', delimiter=' ')
	#np.savetxt("result.txt",data, fmt="%s")
	#f2 = open('result.txt', 'w')
	#f2.write(prediction)
	#f2.close()
	
	
	'''
	array = struct.unpack(strLength, f.read(arraySize*4)) # Reading the actual array

	print 'Received array size = ', len(array)
	f.close()
	print array.shape()


	imageSize1 = 224 
	imageSize2 = 224 

	image = np.zeros((imageSize1, imageSize2, 3), float, 'C')

	imageTmp = np.reshape(array, (imageSize1, imageSize2), order='F')/255.0
	image = np.zeros((imageSize1, imageSize2, 3), float, 'C')
	image[:, :, 0] = imageTmp
	image[:, :, 1] = imageTmp
	image[:, :, 2] = imageTmp  

	arraySize = model.predict(image)

	print 'Sending data back ; ArraySize'
	wp = open(file2Path, 'w')
	wp.write(struct.pack('>i',arraySize))

	print 'Sending data back ; FLOAT_Array'
	#wp = open(file2Path, 'w')
	packed = struct.pack('<'+`arraySize`+'f', *desc)
	#print 'sending = ' + `packed`
	wp.write(packed)
	wp.close()

	print 'Array Sent'

	print 'Ending handshake.'
	'''
