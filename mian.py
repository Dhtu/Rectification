# /use/bin/env python
# Rectification
# use UTF-8
# Python 3.6.3

import cv2 as cv
import numpy as np

# load the calibration data

camtxL = np.load('camera_matrixL.npy')
camtxR = np.load('camera_matrixR.npy')
distL = np.load('distortion_cofficientsL.npy')
distR = np.load('distortion_cofficientsR.npy')
rmtx = np.load('rotation_mat.npy')
tvec = np.load('translation_vectors.npy')

Limg = cv.imread('.\data\left01.jpg', cv.IMREAD_GRAYSCALE)
Rimg = cv.imread('.\data\\right01.jpg', cv.IMREAD_GRAYSCALE)

size=(640,480)
# Recfitication
RL, RR, ncamtxL, ncamtxR, Q, validPixROI1, validPixROI2 = cv.stereoRectify(camtxL, distL, camtxR, distR, size,
                                                                           rmtx, tvec)
Lmap1,Lmap2=cv.initUndistortRectifyMap(camtxL,distL,RL,ncamtxL,size,cv.CV_16SC2)
# cv.imshow('',Limg)
# cv.waitKey()
Rmap1,Rmap2=cv.initUndistortRectifyMap(camtxR,distR,RR,ncamtxR,size,cv.CV_16SC2)
Limg_Rec=cv.remap(Limg,Lmap1,Lmap2,cv.INTER_LINEAR)
Rimg_Rec=cv.remap(Rimg,Rmap1,Rmap2,cv.INTER_LINEAR)
cv.imwrite('.\data\left_Rec.jpg',Limg_Rec)
cv.imwrite('.\data\\right_Rec.jpg',Rimg_Rec)
