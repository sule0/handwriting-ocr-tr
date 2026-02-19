import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.filters import threshold_sauvola

img = cv.imread('pilot_dataset\images\img1.jpeg')
dst = cv.fastNlMeansDenoisingColored(img,None,15, 7, 21)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast = clahe.apply(gray)

thresh = threshold_sauvola(contrast, window_size=25)
binary = (contrast > thresh).astype(np.uint8) * 255

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(binary, 'gray')
plt.show()