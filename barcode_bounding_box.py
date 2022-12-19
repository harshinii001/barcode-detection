#!/usr/bin/env python3

from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np
import os

a = []

j = 0
for filename in os.listdir("images"):
    img = cv2.imread(os.path.join("images",filename))


    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grey_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
    info = decode(thresh)
    
    for i in info:
        p = []
        q = []
        print("only id")
        print(i.data.decode("utf-8"))
        (x, y, w, h) = i.rect

        for n in range(0,len(i.polygon)):
            p.append(i.polygon[n][0])
            q.append(i.polygon[n][1])
        x_min = min(p)
        x_max = max(p)
        y_min = min(q)
        y_max = max(q)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)

    half1 = cv2.resize(img, (0, 0), fx = 0.3, fy = 0.3)
    #cv2.imshow("image_bounding_box" + str(j), half1)
    #cv2.imshow('input image' + str(j), thresh)

    path = "/home/harshini/Desktop/sample/output5_barcode_detection_and_decode_with_pyzbar"
    cv2.imwrite(os.path.join(path , "barcode_"+filename), img)

    print("all information")
    print("image number " + str(j) + "  :",info)
    j = j + 1
    



cv2.waitKey(0)
cv2.destroyAllWindows()


"""

        for n in range(0,len(i.polygon)):
            for m in range(0,2):
                p.append(i.polygon[n][m])
                #print(p)
            q.append(p)
            p = []
        #print(q)
        print(zip(q))
        q = []  


        print(i.orientation)
        if(i.orientation == "UP"):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
        elif(i.orientation == "DOWN"):
            cv2.rectangle(img, (x, y), (x + w, y - h), (255, 0, 0), 5)
        elif(i.orientation == "LEFT"):
            cv2.rectangle(img, (x, y), (x - w, y + h), (255, 0, 0), 5)
        elif(i.orientation == "RIGHT"):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)

        

"""