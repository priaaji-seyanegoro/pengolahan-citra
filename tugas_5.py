from cv2 import cv2
import numpy as np

frame = cv2.imread('./images/balls.jpg')

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([94, 80, 2])
upper_blue = np.array([126, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask=mask)

cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()
