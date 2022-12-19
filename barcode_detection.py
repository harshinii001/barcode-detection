#!/usr/bin/env python3

from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np
import os
import pytesseract 

a = []

j = 0

for filename in os.listdir("images"):
    img = cv2.imread(os.path.join("images",filename))

    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(img)
    gradX = cv2.Sobel(grey_image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(grey_image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    gradient = cv2.subtract(gradX, gradY) 

    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
    closed1 = cv2.erode(closed, None, iterations = 4) 
    closed2 = cv2.dilate(closed1, None, iterations = 4)
    
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print(cnts)

    for c in cnts:
        area = cv2.contourArea(c)
    if area > 0:
        cv2.drawContours(img, [c], -1, (0,255,0), 5)


    #c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    #compute the rotated bounding box of the largest contour
    #rect = cv2.minAreaRect(c)
    #box = np.int0(cv2.boxPoints(rect))

    #cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    #cv2.imshow("Image", img)

    #scharrx = cv2.Scharr(grey_image,cv2.CV_64F,1,0,-1)
    #scharry = cv2.Scharr(grey_image,cv2.CV_64F,0,1,-1)
    #combined = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

    resized_img = cv2.resize(closed2, (0, 0), fx = 0.3, fy = 0.3)
    resized_img1 = cv2.resize(gradient, (0, 0), fx = 0.3, fy = 0.3)
    cv2.imshow("output image 1 : " + str(j),resized_img)
    cv2.imshow("output image 2 : " + str(j),resized_img1)

    j = j + 1

cv2.waitKey(0)
cv2.destroyAllWindows()



"""
https://opencv.org/recognizing-one-dimensional-barcode-using-opencv/  


https://gist.github.com/gyurisc/aa692ad27c2699ec3700    

"""