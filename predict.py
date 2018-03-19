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

model = load_model('trained_model.h5')


mypath='./data/train' 
img = cv2.imread(mypath+'/1.jpg')
img_1 = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
img_main = img_1 [np.newaxis,...]  # dimension added to fit input size



prediction = model.predict(img_main)
print(prediction)