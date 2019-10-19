# -*- coding: utf-8 -*-
import tifffile
import numpy as np
import cv2
import glob


def Tiff2PNG(path, savePath):
    img = tifffile.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(savePath, img)


if __name__ == '__main__':
    path = '/Users/rhc/WorkPlace/Programs/JackNetworkFramework/ResultImg/'
    savePath = '/Users/rhc/PNG_2/'
    name = '*.tif'

    files = glob.glob(path + name)

    for i in range(len(files)):
        readPath = files[i]
        pos = readPath.find('.tif')
        fileName = files[i]
        # print fileName[17:pos]
        fileName = savePath + fileName[len(path):pos] + '.png'
        print fileName
        Tiff2PNG(readPath, fileName)

    #path = path + name + 'tif'
