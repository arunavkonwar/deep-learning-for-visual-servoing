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
import tensorflow as tf

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import time

from keras.models import model_from_config #new
#from tensorflow_serving.session_bundle import exporter #new
from tensorflow.contrib.session_bundle import exporter



sess = tf.Session()
tf.initialize_all_variables().run(session=sess)

K.set_learning_phase(0)  # all new operations will be in test mode from now on

model = load_model('trained_model.h5')


#mypath='./data/train' 
img = cv2.imread('819.jpg')
img_1 = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
img_main = img_1 [np.newaxis,...]  # dimension added to fit input size



start = time.time()
prediction = model.predict(img_main)
end = time.time()
print ("\n\n****************************\n\nModel took %0.2f seconds to predict\n\n****************************\n"%(end - start))
print(prediction)





#afma6 robot
