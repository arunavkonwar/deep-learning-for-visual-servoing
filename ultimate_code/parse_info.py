import numpy as np
import cv2

filename ='../../../code-new/binary/data1.txt'

#---------GET NUMBER OF LINES IN THE FILE----------
'''
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
a=file_len(filename)
'''
#--------------------------------------------------

data = np.loadtxt(filename, delimiter=' ')
print len(data)
a=(data[:]).reshape(224,224)

#mypath='./data/train' 
img = cv2.imread('1.jpg')
img2 = np.zeros_like(img) #replicating the numpy array shape of another image
img2[:,:,0] = a
img2[:,:,1] = a
img2[:,:,2] = a
#img_1 = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

print "haha"	
print img2.shape 
img_main = img2[np.newaxis,...]  # dimension added to fit input size
print img_main.shape
'''
start = time.time()
prediction = model.predict(img_main)
end = time.time()
print ("\n\n****************************\n\nModel took %0.2f seconds to predict\n\n****************************\n"%(end - start))
print(prediction)
'''

#file.write(a)
#print a.shape
