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

    
    rgb_img = result_norm
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,230,255,0)
    
    path = "/home/harshini/Desktop/sample/output4_edge_contour"
    cv2.imwrite(os.path.join(path , "thresh_" + filename), thresh)

    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

# Find the convex hull for all the contours
    for cnt in contours:
        #mhull = cv2.convexHull(cnt)
        thresh = cv2.drawContours(thresh,[cnt],0,(0,0,0),5)
        #img = cv2.drawContours(img,[hull],0,(0,0,255),3)
    

    path = "/home/harshini/Desktop/sample/output4_edge_contour"
    cv2.imwrite(os.path.join(path , "thresh_" + filename), thresh)

    img_floodfill1 = thresh
    h,w = img_floodfill1.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill1,mask,(255,255), (0,0,0))
    denoise = cv2.blur(img_floodfill1, (25, 25))
    ret, denoise_1 = cv2.threshold(denoise, 120, 255, cv2.THRESH_BINARY)


    gray_img = denoise_1  ## gray
    height, width = gray_img.shape
    white_padding = np.zeros((50, width, 3))
    white_padding[:, :] = [255, 255, 255]
    rgb_img = np.row_stack((white_padding, rgb_img))
    gray_img = 255 - gray_img
    gray_img[gray_img > 100] = 255
    gray_img[gray_img <= 100] = 0
    black_padding = np.zeros((50, width))
    gray_img = np.row_stack((black_padding, gray_img))


    kernel = np.ones((30, 30), np.uint8)
    closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    #edges = cv2.Canny(closing, 100, 200)
    ret,thresh2 = cv2.threshold(closing,230,255,0)
 
    
    resize1 = cv2.resize(thresh, (0, 0), fx = 0.3, fy = 0.3)
    resize2 = cv2.resize(closing, (0, 0), fx = 0.3, fy = 0.3)
    cv2.imshow("image_bounding_box - 1 " + str(j), resize1)
    cv2.imshow("image_bounding_box - 2 " + str(j), resize2)
    j = j + 1

cv2.waitKey(0)
cv2.destroyAllWindows()


"""
https://stackoverflow.com/questions/47777585/detecting-outer-most-edge-of-image-and-plotting-based-on-it  


h,w = img_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill,mask,(255,255), (0,0,0))
    denoise = cv2.blur(img_floodfill, (25, 25))
    ret, denoise_1 = cv2.threshold(denoise, 120, 255, cv2.THRESH_BINARY)


"""
