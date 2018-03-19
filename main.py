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




from random import randint
from sklearn.preprocessing import MinMaxScaler

batch_size = 14

#loading pretrained and edited model
model = load_model('vgg16_edit.h5')


y_filename ='./data/data.txt'
y_data = np.loadtxt(y_filename, delimiter='  ')

y_data_train = y_data[:700]
y_data_validation = y_data[700:840]

#########################################

h5f = h5py.File('images_in_h5_format.h5','r')
x_data_train = h5f['dataset_1'][:]


h5f = h5py.File('valid_images_in_h5_format.h5','r')
x_data_validation = h5f['dataset_2'][:]


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_data_train, y_data_train,
          epochs=1,
          batch_size=batch_size,
          validation_data=(x_data_validation, y_data_validation))


model.save_weights('trained_model_weights.h5')
model.save('trained_model.h5')








