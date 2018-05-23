import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
cap = cv2.VideoCapture('video.avi')
orb = cv2.ORB_create()
k1, d1 = orb.detectAndCompute(post, None)
kn1.train(np.float32(d1), cv2.ml.ROW_SAMPLE, np.arange(len(d1)))
while(True):
    ret, frame = cap.read()
    kk, dd = orb.detectAndCompute(frame, None)
    ret, res, nei, dis = kn1.findNearest(np.float32(d2), k=1)
    cp = np.array(img, copy=True)
    for i in range (len(res)):
        (a1, b1) = k1[int(res[i])].pt
        (a2, b2) = k2[i].pt
        cv2.circle(cp, (np.int32(a2), np.int32(b2)), 63, (0,0,255), -1)
    plt.imshow(cp)

cap.release()
cv2.destroyAllWindows()
