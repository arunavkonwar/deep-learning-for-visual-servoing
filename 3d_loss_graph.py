import os
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import h5py
import time
from sklearn.metrics import mean_squared_error
import cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.axes_grid1 as axes_grid1	



prediction ='/home/arunav/main-code/arunav/compare.txt'
ground_truth = '/home/arunav/main-code/arunav/validation_data_8k.txt'

y_data = np.loadtxt(prediction, delimiter=' ', usecols=[0,1])
pred_data = y_data[:]
print(pred_data[0][0])

y_data = np.loadtxt(ground_truth, delimiter='  ', usecols=[0,1])
gt_data = y_data[:]

a = np.zeros(1800)

for i in range(1800):
	print (mean_squared_error(pred_data[i],gt_data[i]))
	a[i] = mean_squared_error(pred_data[i],gt_data[i])
lol = np.zeros((1800, 1800, 1800))

z=0	
for i in range(30):
	for j in range(60):
		lol[i][j][k] = a[z]
		z=z+1
		
		
print(lol.shape)
'''
a = np.expand_dims(a, axis=0)
im = plt.imshow(a, cmap='hot')
plt.colorbar(im, orientation='horizontal')
plt.show()
'''
	
	

