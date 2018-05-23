#! /usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt

frame_idx = 0;

def next_frame_pair():
    global frame_idx
    fl = '{:08d}'.format(frame_idx)
    fr = '{:08d}'.format(frame_idx+1)
    print(fl, fr)
    frame_idx +=2
    return (cv2.imread('frames/'+fl+'.jpeg'),
            cv2.imread('frames/'+fr+'.jpeg'))

stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=25)

img = None
dx = 0

while True:
    imgLc, imgRc = next_frame_pair()
    imgL = cv2.cvtColor(imgLc, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgRc, cv2.COLOR_BGR2GRAY)
    #print(imgL, imgR)
    if img is None:
        img = np.zeros((imgL.shape[0],2*imgL.shape[1],3), np.float32)
        dx = imgL.shape[1]

    disparity=stereo.compute(imgL, imgR)
    da = np.array(disparity,dtype=np.float32)
    da /= np.max(da)
    da = np.dstack([da]*3)
    da[disparity<0] = (0,0,255)

    print(np.min(disparity), np.max(disparity))

    img[:imgL.shape[0], :dx] = imgLc.copy()/255.0
    img[:imgL.shape[0], dx:] = da.copy()

    cv2.imshow('disparity', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('stereoD.png',img)
