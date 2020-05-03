from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

frame = cv2.imread('./images/image_uts.jpg')

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# define range of blue color in HSV
lower_blue = np.array([94, 80, 2])
upper_blue = np.array([126, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)


# define contour reference of mask
contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# find biggest_contour
biggest_contour = max(contours, key=cv2.contourArea)

# set box frame
x, y, w, h = cv2.boundingRect(biggest_contour)
cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.title("PRIA AJI SN / 1644190020")
plt.imshow(img_rgb)
plt.show()
