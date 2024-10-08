"""   Created on Sun May 28 16:48:54 2023

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""
# First we get the necessary packages.
import cv2
import imutils
import numpy as np
cv2.destroyAllWindows()

image = cv2.imread("segman.png")
image=np.max(image)-image
cv2.imshow("Shapes_image", image)
cv2.waitKey(500)
gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray_scaled, 30,150)


thresh = cv2.threshold(gray_scaled, 225,225, cv2.THRESH_BINARY_INV)[1]

contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

area=[];
for contour in contours:# loop over each contour found
    output = 0*image.copy()
    # img, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED
    cv2.drawContours(output, [contour], -1,(255,255,255),thickness=cv2.FILLED) # outline and display them, one by one.
    area.append(np.sum(output))
    # cv2.imshow('Contours area ='+str(np.log (np.sum(output))), output)
    # cv2.waitKey(300)

index = np.where (area==np.max(area))
output = 0*image.copy()
cv2.drawContours(output, [contours[index[0][0]]], -1,(255,255,255),thickness=cv2.FILLED) # outline and display them, one by one.
# area.append(np.sum(output))
cv2.imshow('Contours area max  ='+str(np.sum(output)), output)
cv2.waitKey(300)
    
    