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
    half = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
    path = "/home/harshini/Desktop/sample/all barcode output"
    cv2.imwrite(os.path.join(path , str(j) + '.jpg'), half)
    #info = decode(Image.open(os.path.join("all barcode",filename)))
    info = decode(img)
    for i in info:
        print("only id")
        print(i.data.decode("utf-8"))

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
    

#cv2.waitKey(0)
#cv2.destroyAllWindows()