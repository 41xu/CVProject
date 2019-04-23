import cv2
import numpy as np

img1=cv2.imread('1.JPG')
img2=cv2.imread('2.JPG')
gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift=cv2.xfeatures2d.SIFT_create()

# kp=sift.detect(gray,None)
# kp,des=sift.compute(gray,kp)
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)

cv2.FlannBasedMatcher()

img_1=np.zeros(img1.shape,np.uint8)
img_2=np.zeros(img2.shape,np.uint8)

img1=cv2.drawKeypoints(gray1,kp1,img_1)
img2=cv2.drawKeypoints(gray2,kp2,img_2)

cv2.imshow('sp1',img1)
cv2.imshow('sp2',img2)
merge=np.hstack((img1,img2))
cv2.imshow('gray',merge)
cv2.waitKey(0)