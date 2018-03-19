import numpy as np
import h5py

h5f = h5py.File('images_in_h5_format.h5','r')
b = h5f['dataset_1'][:]
print(b.shape)
print(len(b))
print(b)
h5f.close()

#np.allclose(a,b)