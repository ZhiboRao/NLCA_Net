# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import os
from PIL import Image


def Mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    # check the file pat
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

    return


def ResizeImg(path, savePath):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    height, width, _ = img.shape
    # print height, width
    height = int(height / 4)
    width = int(width / 4)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(savePath, img)


def ResizeImgFolder(path, savePath, name, filetype):
    Mkdir(savePath)
    files = glob.glob(path + name)
    print 'Find the number of file in %s : %d' % (path, len(files))

    for i in range(len(files)):
        readPath = files[i]
        pos = readPath.find(filetype)
        fileName = files[i]
        # print fileName[17:pos]
        fileName = savePath + fileName[len(path):pos] + '.png'
        print fileName
        ResizeImg(readPath, fileName)


if __name__ == '__main__':
    root = '/Users/rhc/Documents/Scholar/Ph.D./MyPaper/BGA_Net/BGA_NET_TGRSS_ORIGIN/img/'

    path = '/Users/rhc/Documents/Scholar/Ph.D./MyPaper/BGA_Net/BGA_NET_TGRSS_ORIGIN/img/Example/'
    savePath = '/Users/rhc/Documents/Scholar/Ph.D./MyPaper' +\
        '/BGA_Net/BGA_NET_TGRSS_ORIGIN/img/Example/SMALL/'

    subfolders = ['Example/', 'Result/US3D/', 'Result/CityScape/', 'Result/KITTI/',
                  'Result/Attention/dsp_attention/', 'Result/Attention/cls_attention/']

    name = '*.png'
    filetype = '.png'

    for i in range(len(subfolders)):
        path = root + subfolders[i]
        ResizeImgFolder(path, path, name, filetype)

    # ResizeImgFolder(path, savePath, name, filetype)
    print "Finish!"
