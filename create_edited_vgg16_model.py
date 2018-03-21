
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


#model.layers.pop()


for layer in model.layers:
    layer.trainable = False


model.add(Dense(6, activation=None)) #or linear

#compile model before saving
#model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

#model.compile(Adam(lr=.0001), loss='mean_squared_error', metrics=['accuracy'])

model.save('vgg16_edit.h5')
model.summary()

plot_model(model, to_file='model.png')

print('\n\n**********************\nModel saved')
