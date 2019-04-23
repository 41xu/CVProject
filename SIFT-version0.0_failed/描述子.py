import cv2
import numpy as np
from matplotlib import pyplot as plt

img1=cv2.imread('1.JPG')
img2=cv2.imread('2.JPG')
gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift=cv2.xfeatures2d.SIFT_create()

# kp=sift.detect(gray,None)
# kp,des=sift.compute(gray,kp)
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)
matches=flann.knnMatch(des1,des2,k=2)

matchesMask=[[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance<0.7*n.distance:
        matchesMask[i]=[1,0]

drawParams=dict(matchColor=(0,0,255),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0)
resultimage=cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**drawParams)

plt.imshow(resultimage,),plt.show()


