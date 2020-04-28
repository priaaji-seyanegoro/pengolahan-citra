from cv2 import cv2
from matplotlib import pyplot as plt

# read image
img = cv2.imread('./images/sudoku.png', 0)

result_th = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

plt.title('Adaptive Gaussian Thresholding')
plt.imshow(result_th, 'gray')
plt.show()
