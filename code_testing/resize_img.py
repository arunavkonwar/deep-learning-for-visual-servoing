import cv2
import numpy as np

img = cv2.imread('1.jpg')
res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('export.jpg',res)