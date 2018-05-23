#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# frame_idx = 0;
#
# def next_frame_pair():
#     global frame_idx
#     fl = '{:08d}'.format(frame_idx)
#     fr = '{:08d}'.format(frame_idx+1)
#     print(fl, fr)
#     frame_idx +=2
#     return (cv2.imread('frames/'+fl+'.jpeg'),
#             cv2.imread('frames/'+fr+'.jpeg'))
#
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=25)
#
# img = None
# dx = 0
#
# while True:
#     imgLc, imgRc = next_frame_pair()
#     imgL = cv2.cvtColor(imgLc, cv2.COLOR_BGR2GRAY)
#     imgR = cv2.cvtColor(imgRc, cv2.COLOR_BGR2GRAY)
#     #print(imgL, imgR)
#     if img is None:
#         img = np.zeros((imgL.shape[0],2*imgL.shape[1],3), np.float32)
#         dx = imgL.shape[1]
#
#     disparity=stereo.compute(imgL, imgR)
#     da = np.array(disparity,dtype=np.float32)
#     da /= np.max(da)
#     da = np.dstack([da]*3)
#     da[disparity<0] = (0,0,255)
#     #d = cv2.cvtColor(da, cv2.COLOR_GRAY2BGR)
#
#     print(np.min(disparity), np.max(disparity))
#
#     img[:imgL.shape[0], :dx] = imgLc.copy()/255.0
#     img[:imgL.shape[0], dx:] = da.copy()
#
#     cv2.imshow('disparity', img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break






#! /usr/bin/env python3

import numpy as np
import cv2

frame_idx = 0;
def next_frame():
    global frame_idx
    f = '{:08d}'.format(frame_idx)
    print(f)
    frame_idx +=2
    return cv2.imread('frames/'+f+'.jpeg')

#cap = cv2.VideoCapture('slow.flv')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
#ret, old_frame = cap.read()
old_frame = next_frame()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = []
# Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)
while(1):
    if len(p0)==0:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        mask = np.zeros_like(old_frame)
#    ret,frame = cap.read()
    frame = next_frame()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # print(p0, '***', p1)
    if not p1 is None:
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)
            cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
         # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()
