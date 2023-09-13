import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('test_images/test4.jpg')
plt.imshow(img)
plt.show()

graysc_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(graysc_img,cmap='gray')
plt.show()