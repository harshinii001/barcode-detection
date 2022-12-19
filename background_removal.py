#!/usr/bin/env python3
import numpy as np
import imutils
import cv2
import numpy as np
import os

a = []
j = 0

for filename in os.listdir("images"):
    img = cv2.imread(os.path.join("images",filename))
    
    image = cv2.imread(os.path.join("images",filename), -1)

    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rgb_planes = cv2.split(image)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)   ### removed shadows and plane colours

    path = "/home/harshini/Desktop/sample/output3_shadow_removal"
    cv2.imwrite(os.path.join(path , "rgb_no_background_" + filename), result_norm)

    resize1 = cv2.resize(result, (0, 0), fx = 0.3, fy = 0.3)
    resize2 = cv2.resize(result_norm, (0, 0), fx = 0.3, fy = 0.3)
    #cv2.imshow("image_bounding_box - 1 " + str(j), resize1)
    #cv2.imshow("image_bounding_box - 2 " + str(j), resize2)
    j = j + 1

cv2.waitKey(0)
cv2.destroyAllWindows()


'''


   # thresh = cv2.adaptiveThreshold(grey_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(grey_image,(5,5),0)
    #ret3,th3 = cv2.threshold(grey_image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    ret, th1 = cv2.threshold(grey_image,140,255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((1,1), dtype = "uint8")/15
    bilateral = cv2.bilateralFilter(th1, 9 , 75, 75)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(bilateral, kernel, iterations = 2)
    dilation = cv2.dilate(erosion, kernel, iterations=4)


    height, width = grey_image.shape
    white_padding = np.zeros((50, width, 3))
    white_padding[:, :] = [255, 255, 255]
    rgb_img = np.row_stack((white_padding, rgb_img))

    grey_image = 255 - grey_image
    grey_image[grey_image > 100] = 255
    grey_image[grey_image <= 100] = 0
    black_padding = np.zeros((50, width))
    grey_image = np.row_stack((black_padding, grey_image))

https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv  



'''
