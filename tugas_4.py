from cv2 import cv2

# read an image params(name file , type flag [0 or 1 or -1])
image = cv2.imread('./images/Koala.jpg', 1)

# read img pixel as a matrics
print(image)

# show pop up window for show img params (title of window , src file)
cv2.imshow('tugas 4', image)

# set time out to 5s and close the window
cv2.waitKey(5000)
