#!/usr/bin/env python3

from PIL import Image
import cv2
import numpy as np
import os

a = []
j = 0
for filename in os.listdir("output1_text_detection"):
    
    image = cv2.imread(os.path.join("output1_text_detection",filename), 0)
    ret, thresh1 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
    img_floodfill = thresh1.copy()
    h,w = img_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill,mask,(0,0), (255,255,255))
    denoise = cv2.blur(img_floodfill, (25, 25))
    ret, denoise_1 = cv2.threshold(denoise, 120, 255, cv2.THRESH_BINARY)


    path = "/home/harshini/Desktop/sample/output2_text_detection"
    cv2.imwrite(os.path.join(path , "fill" + filename), denoise_1)
    #info = decode(Image.open(os.path.join("all barcode",filename)))

    j = j + 1


    

cv2.waitKey(0)
cv2.destroyAllWindows()