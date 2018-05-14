#! /usr/bin/env python3

import cv2
import numpy as np


p = cv2.imread('strawberries.tiff')
# m = cv2.imread('messi-gray.tiff', cv2.IMREAD_GRAYSCALE)/255
# print(p.shape, m.shape)
#
# n = random_gauss(m, 0.15)
#
# m1 = clip(m+random_gauss(m, 0.05))
# m2 = clip(m+random_gauss(m, 0.15))

cv2.imwrite('mansikka.jpg', p)
# cv2.imwrite('m.jpg',  255*m)
# cv2.imwrite('m1.jpg', 255*m1)
# cv2.imwrite('m2.jpg', 255*m2)

cv2.imshow('BGR', p)
#cv2.imshow('', i)
cv2.waitKey()

