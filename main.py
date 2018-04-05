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
import utils
#import models
import time
from keras.callbacks import ModelCheckpoint

np.random.seed(7) # for reproducibility



batch_size = 14

#loading pretrained and edited model
#model = load_model('vgg16_edit.h5')
model = load_model('vgg16_edit.h5')


y_filename ='./data/data.txt'
y_data = np.loadtxt(y_filename, delimiter='  ')

y_data_train = y_data[:]

#########################################

h5f = h5py.File('images_in_h5_format.h5','r')
x_data_train = h5f['dataset_1'][:]


# ======================================================================
# Run training process:
# Set callbacks:
# => callbacks are used to define operations during training. Here fore example I use a callback called "checkpoint" to save the weights of my model after each epoch. Then I define another one to compute my custom metrics.
'''
checkpoint     = ModelCheckpoint('weights.epoch{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
metrics        = utils.Metrics()
callbacks_list = [checkpoint, metrics]
'''
# ======================================================================                     
# Configure the training process:
print('Preparing training ...')
#adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#adam = Adam(lr=0.0001)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
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


#model.save_weights('trained_model_weights.h5')
model.save('trained_model.h5')

#utils.eml_save_history(hist, metrics)








