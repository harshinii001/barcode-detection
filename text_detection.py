#!/usr/bin/env python3

from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np
import os


a = []

j = 0

for filename in os.listdir("missing and partial barcode"):
    img = cv2.imread(os.path.join("missing and partial barcode",filename))
    
    image = cv2.imread("missing and partial barcode output/output_IMG_20220303_174744.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
    img_floodfill = thresh1
    h,w = img_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill,mask,(0,0), (255,255,255))
    denoise = cv2.blur(img_floodfill, (25, 25))
    ret, denoise_1 = cv2.threshold(denoise, 120, 255, cv2.THRESH_BINARY)

    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
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

    dest_and = cv2.bitwise_and(denoise_1, closed2, mask = None)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    detector = cv2.SimpleBlobDetector_create(params)
    blank = np.zeros((1, 1))
    keypoints = detector.detect(dest_and) 
    blobs = cv2.drawKeypoints(dest_and, keypoints, blank, (0,0,255), 
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    #c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    #compute the rotated bounding box of the largest contour
    #rect = cv2.minAreaRect(c)
    #box = np.int0(cv2.boxPoints(rect))

    #cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    #cv2.imshow("Image", img)

    #scharrx = cv2.Scharr(grey_image,cv2.CV_64F,1,0,-1)
    #scharry = cv2.Scharr(grey_image,cv2.CV_64F,0,1,-1)
    #combined = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

    resized_img = cv2.resize(img, (0, 0), fx = 0.3, fy = 0.3)
    resized_img1 = cv2.resize(dest_and, (0, 0), fx = 0.3, fy = 0.3)
    cv2.imshow("output image 1 : " + str(j),resized_img)
    cv2.imshow("output image 2 : " + str(j),resized_img1)

    j = j + 1

cv2.waitKey(0)
cv2.destroyAllWindows()


### https://colab.research.google.com/github/bhattbhavesh91/keras-ocr-demo/blob/main/keras-ocr-notebook.ipynb#scrollTo=bL4Dy8kKbsFr  

###  