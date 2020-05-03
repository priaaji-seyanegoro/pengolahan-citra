from cv2 import cv2
import numpy as np

# read video
cap = cv2.VideoCapture('./videos/pedestrian_overpass.mp4')

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():

    # absoulte diffrent between frame
    diff = cv2.absdiff(frame1, frame2)

    # convert to grayscale mode
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # blur videos
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # find out the threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # FIND BETTER CONTOURS W/ DILATED
    dilated = cv2.dilate(thresh, None, iterations=3)

    # find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    # draw box
    for con in contours:
        x, y, w, h = cv2.boundingRect(con)
        # if area < 1000 dont draw box
        if cv2.contourArea(con) < 1000:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # draw text on frame
        cv2.putText(frame1, "Status : {}".format('Movement'),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('feed', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
