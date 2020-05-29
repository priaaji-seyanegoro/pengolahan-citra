from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
img = cv2.imread('./images/peta-tugas8.png', 0)

# structuring element or kernel which decides the nature of operation.
kernel = np.ones((5, 5), np.uint8)

# remove the noise dots
dilation = cv2.dilate(img, kernel, iterations=3)
erosion = cv2.erode(dilation, kernel, iterations=2)

# blur or smothing image after remove noise
blur = cv2.GaussianBlur(erosion, (5, 5), 0)

titles = ['Original Image', 'Removing Noise', "blured Image"]
images = [img, erosion, blur]

for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
