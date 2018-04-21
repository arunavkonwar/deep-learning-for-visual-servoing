from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import h5py



#path to directory	
#images already in (224, 224, 3) format

mypath='../../code-new/binary/generated_images_8k'
#mypath='../../code-new/binary/validation_images_8k' 
img = cv2.imread(mypath+'/1.jpg')
#print(img.shape)
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

#images = np.empty([len(onlyfiles),224,224,3], dtype='uint8')
#images = np.zeros([len(onlyfiles),224,224,3], float, 'C')
images = np.zeros([len(onlyfiles),224,224,3], dtype='uint8')

for n in range(0, len(onlyfiles)):
  #images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
  #img = np.zeros((224, 224, 3), float, 'C')
  img = np.zeros((224, 224, 3), dtype='uint8')
  img = cv2.imread( join(mypath,onlyfiles[n]) )
  #img_1 = np.zeros((224, 224, 3), float, 'C')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_1 = np.zeros((224, 224, 3), dtype='uint8')
  img_1[:, :, 0] = img[:, :, 0]
  img_1[:, :, 1] = img[:, :, 0]
  img_1[:, :, 2] = img[:, :, 0]
  images[n] = img_1
  
  #images[n] = cv2.resize(loaded_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

h5f = h5py.File('images_in_h5_format_8k_uint8.h5', 'w')
#h5f = h5py.File('validation_images_8k_uint8.h5', 'w')
h5f.create_dataset('dataset_1', data=images)
print('********************\nTRAINING set images saved to hdf5')



'''

mypath='./data/train' 
img = cv2.imread(mypath+'/1.jpg')
#print(img.shape)
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

images = np.empty([len(onlyfiles),224,224,3], dtype='uint8')

for n in range(0, len(onlyfiles)):
  loaded_img = cv2.imread( join(mypath,onlyfiles[n]) )
  images[n] = cv2.resize(loaded_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

h5f = h5py.File('images_in_h5_format.h5', 'w')
h5f.create_dataset('dataset_1', data=images)
print('********************\nTRAINING set images saved to hdf5')
'''

'''

mypath_validation='./data/validation' 
img = cv2.imread(mypath_validation+'/1.jpg')
#print(img.shape)
onlyfiles = [ f for f in listdir(mypath_validation) if isfile(join(mypath_validation,f)) ]

images_valid = np.empty([len(onlyfiles),224,224,3], dtype='uint8')

for n in range(0, len(onlyfiles)):
  loaded_img = cv2.imread( join(mypath_validation,onlyfiles[n]) )
  images_valid[n] = cv2.resize(loaded_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

h5f = h5py.File('valid_images_in_h5_format.h5', 'w')
h5f.create_dataset('dataset_2', data=images_valid)
print('********************\nVALIDATION set images saved to hdf5')

'''

#hey =[1,2,3,4]
#print (len(images_valid))   #double brakets for python3
#images 
