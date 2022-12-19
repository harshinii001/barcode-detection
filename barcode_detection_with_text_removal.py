#!/usr/bin/env python3

from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np
import os
 

a = []

img = cv2.imread("partial barcode/IMG_20220303_175317.jpg")
    
image = cv2.imread("partial barcode output/output_IMG_20220303_175317.jpg",0)
ret, thresh1 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
img_floodfill = thresh1.copy()
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
dest_and1 = cv2.bitwise_and( gradient, denoise_1, mask = None)
blurred = cv2.blur(dest_and1, (9, 9))  ## gradient
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
closed1 = cv2.erode(closed, None, iterations = 4) 
closed2 = cv2.dilate(closed1, None, iterations = 4)
    

dest_and = cv2.bitwise_and( closed2, denoise_1, mask = None)

cv2.imwrite('partial barcode output/final_IMG_20220303_175317.jpg', dest_and)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 10
detector = cv2.SimpleBlobDetector_create(params)
blank = np.zeros((1, 1))
keypoints = detector.detect(dest_and) 
blobs = cv2.drawKeypoints(dest_and, keypoints, blank, (0,0,255), 
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(keypoints)

    #c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    #compute the rotated bounding box of the largest contour
    #rect = cv2.minAreaRect(c)
    #box = np.int0(cv2.boxPoints(rect))

    #cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    #cv2.imshow("Image", img)

    #scharrx = cv2.Scharr(grey_image,cv2.CV_64F,1,0,-1)
    #scharry = cv2.Scharr(grey_image,cv2.CV_64F,0,1,-1)
    #combined = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

resized_img = cv2.resize(dest_and1, (0, 0), fx = 0.2, fy = 0.2)
resized_img1 = cv2.resize(gradient, (0, 0), fx = 0.2, fy = 0.2)
cv2.imshow("output image 1 : " + str(1),resized_img)
cv2.imshow("output image 2 : " + str(2),resized_img1)

cv2.waitKey(0)
cv2.destroyAllWindows()