# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import os
from PIL import Image
from copy import deepcopy
import tifffile


def RGB2GRAY(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def GRAY2BLACK(img):
    res = deepcopy(img)
    res[:] = 0
    res[img > 128] = 255
    return res


if __name__ == '__main__':
    path = '/Users/rhc/PICS/%d.jpg'
    savePath = '/Users/rhc/PICS/1/%d.jpg'

    for i in range(9):
        pathName = path % i
        savePathName = savePath % i
        img = Image.open(pathName).convert("RGB")
        img = np.array(img)
        img = RGB2GRAY(img)
        #cv2.imwrite(savePath, img)
        img = GRAY2BLACK(img)
        cv2.imwrite(savePathName, img)
