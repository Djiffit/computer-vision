import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import math

cap = cv2.VideoCapture('video.avi')
orb = cv2.ORB_create()
kn1 = cv2.ml.KNearest_create()
post = cv2.imread('poster.jpeg')
k1, d1 = orb.detectAndCompute(post, None)
kn1.train(np.float32(d1), cv2.ml.ROW_SAMPLE, np.arange(len(d1)))
while(True):
    ret, frame = cap.read()
    cp = np.array(frame, copy=True)
    a = datetime.now()

    kk, dd = orb.detectAndCompute(frame, None)
    if dd is None: break
    ret, res, nei, dis = kn1.findNearest(np.float32(dd), k=1)
    for i in range(0, post.shape[1]):
        for j in range(0, post.shape[0]):
            if j < post.shape[0] and i < post.shape[1]:
                cp[j][i] = (post[j][i])
    for i in range (len(res)):
        (a1, b1) = k1[int(res[i])].pt
        (a2, b2) = kk[i].pt
        cv2.line(cp, (np.int32(a1), np.int32(b1)), (np.int32(a2), np.int32(b2)), (0, 255, 0), 1)



    # index_params = dict(algorithm = 6,
    #                    table_number = 6, # 12
    #                    key_size = 12,     # 20
    #                    multi_probe_level = 1) #2
    # search_params = dict(checks=50)   # or pass empty dictionary
    #
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(d1, dd, k=1)
    #
    #
    # matchesMask = [[0,0] for i in range(len(matches))]
    #
    # for i in range(0, (len(matches) - 2)):
    #     m = matches[i]
    #     n = matches[i + 1]
    #     if len(n) > 0 and len(m) > 0:
    #         n = n[0]
    #         m = m[0]
    #         if m.distance < 0.7*n.distance:
    #             matchesMask[i]=[1,0]
    #
    # draw_params = dict(matchColor = (0,255,0),
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = 0)
    #
    # img3 = cv2.drawMatchesKnn(post, k1, frame, kk ,matches,None,**draw_params)

    cv2.imshow('title', cp)

    cv2.imwrite('frameslow.jpeg', cp)
    b = datetime.now()
    c = b - a
    print(c.total_seconds())
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()


# import cv2
# import sys
#
# faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#
# video_capture = cv2.VideoCapture(0)
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )
#
#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# cv2.destroyAllWindows()
