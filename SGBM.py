# /use/bin/env python
# Rectification
# use UTF-8
# Python 3.6.3

import cv2 as cv
import numpy as np

# import camera_configs
# load the calibration data

camtxL = np.load('camera_matrixL.npy')
camtxR = np.load('camera_matrixR.npy')
distL = np.load('distortion_cofficientsL.npy')
distR = np.load('distortion_cofficientsR.npy')
rmtx = np.load('rotation_mat.npy')
tvec = np.load('translation_vectors.npy')

Limg = cv.imread('.\data\left03.jpg', cv.IMREAD_GRAYSCALE)
Rimg = cv.imread('.\data\\right03.jpg', cv.IMREAD_GRAYSCALE)

size = (640, 480)
# Recfitication
RL, RR, ncamtxL, ncamtxR, Q, validPixROI1, validPixROI2 = cv.stereoRectify(camtxL, distL, camtxR, distR, size,
                                                                           rmtx, tvec)
Lmap1, Lmap2 = cv.initUndistortRectifyMap(camtxL, distL, RL, ncamtxL, size, cv.CV_16SC2)
# cv.imshow('',Limg)
# cv.waitKey()
Rmap1, Rmap2 = cv.initUndistortRectifyMap(camtxR, distR, RR, ncamtxR, size, cv.CV_16SC2)
Limg_Rec = cv.remap(Limg, Lmap1, Lmap2, cv.INTER_LINEAR)
Rimg_Rec = cv.remap(Rimg, Rmap1, Rmap2, cv.INTER_LINEAR)
cv.imwrite('.\data\left_Rec.jpg', Limg_Rec)
cv.imwrite('.\data\\right_Rec.jpg', Rimg_Rec)

cv.namedWindow("left")
cv.namedWindow("right")
cv.namedWindow("depth")
cv.moveWindow("left", 0, 0)
cv.moveWindow("right", 800, 0)
cv.createTrackbar("blockSize", "depth", 3, 11, lambda x: None)

# def callbackFunc(e, x, y, f, p):
#     if e == cv.EVENT_LBUTTONDOWN:
#         print("threeD[", y, "][", x, "]")
#
#
# cv.setMouseCallback("depth", callbackFunc, None)
while True:

    blockSize = cv.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5

    # StereoSGBM setting
    window_size = 5
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=blockSize,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    disparity = stereo.compute(Limg_Rec, Rimg_Rec)
    disp = cv.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    cv.imshow("left", Limg_Rec)
    cv.imshow("right", Rimg_Rec)
    cv.imshow("depth", disp)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv.imwrite("./data/SGBM_left.jpg", Limg_Rec)
        cv.imwrite("./data/SGBM_right.jpg", Rimg_Rec)
        cv.imwrite("./data/SGBM_depth.jpg", disp)

cv.destroyAllWindows()
