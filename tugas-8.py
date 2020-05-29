from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
img = cv2.imread('./images/tugas-8-crop.png', 0)

# structuring element or kernel which decides the nature of operation.
kernel = np.ones((5, 5), np.uint8)

# Opening is just another name of erosion followed by dilation. It is useful in removing noise
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# blur or smothing image after remove noise
blur = cv2.GaussianBlur(opening, (5, 5), 0)

titles = ['Original Image', 'Removing Noise', "blured Image"]
images = [img, opening, blur]

for i in range(3):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
