#! /usr/bin/env python3

import pickle
import gzip
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import cv2

with gzip.open('mnist.pkl.gz', 'rb') as fs:
    train_set, valid_set, test_set = pickle.load(fs, encoding='latin1')

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x,  test_y  = test_set

plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()

print(train_x.shape, valid_x.shape, test_x.shape)
print(train_y[0:20])

