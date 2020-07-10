from cv2 import cv2
import numpy as np

cap = cv2.VideoCapture('./videos/vtest.avi')

# data kernel 
kernel_dil = np.ones((20,20), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# GAUSSIAN MIXTURE BASED BACKGROUND AND FOREGROUND
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while True :
    ret, frame = cap.read();

    if ret == True:
        # apply fgbg 
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # find countours
        dilation = cv2.dilate(fgmask, kernel_dil, iterations=1)
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # loop and draw box 
        for con in contours:
            x, y, w, h = cv2.boundingRect(con)
            # if area countours < 1350 dont draw box
            if cv2.contourArea(con) < 1350:
                continue
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # draw text on frame
            cv2.putText(frame, "Status : {}".format('Movement'),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow('feed', frame)
        cv2.imshow('fgmask', fgmask)


    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()