#!/usr/bin/env python3

from pyzbar.pyzbar import decode
from PIL import Image
import cv2
import numpy as np
import os

a = []

j = 0
for filename in os.listdir("all barcode"):
    img = cv2.imread(os.path.join("all barcode",filename))
  
#### for increasing accuracy 
    half = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
    path = "/home/harshini/Desktop/sample/all barcode output"
    cv2.imwrite(os.path.join(path , str(j) + '.jpg'), half)
####     -----   #### 

    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    info = decode(grey_image)
    
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
        cv2.imshow("image_bounding_box" + str(j), half1)


    print("all information")
    print("image number " + str(j) + "  :",info)
    j = j + 1

    if img is not None:
        a.append(img)
print("testing")
info1 = decode(Image.open(os.path.join("all barcode/IMG_20220303_175539.jpg")))
for d in info1:
    print(d.data.decode("utf-8"))
#for i in range(0,len(a)):
    #cv2.imshow('input image' + str(i), a[i])
    

cv2.waitKey(0)
cv2.destroyAllWindows()
