# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import os
from PIL import Image
from copy import deepcopy
import tifffile


def sequential_to_las_labels(seq_labels):
    labels = deepcopy(seq_labels)
    labels[:] = 65
    labels[seq_labels == 0] = 2     # ground
    labels[seq_labels == 1] = 5     # trees
    labels[seq_labels == 2] = 6     # building roof
    labels[seq_labels == 3] = 9     # water
    labels[seq_labels == 4] = 17    # bridge / elevated road
    return labels


# convert category value image to RGB color image
def category_to_color(category_image):
    # define colors
    # color table is here: https://www.rapidtables.com/web/color/RGB_Color.html
    colors = []
    colors.append((165, 42, 42))      # 0  brown (ground)
    colors.append((0, 128, 0))        # 1  green (trees)
    colors.append((255, 0, 0))        # 2  red (buildings)
    colors.append((0, 0, 255))        # 3  blue (water)
    colors.append((128, 128, 128))    # 4  gray (elevated road)
    colors.append((0, 0, 0))          # 6  black (other)

    # convert categories to color image
    rows = category_image.shape[0]
    cols = category_image.shape[1]
    categories = category_image.astype(np.uint8)
    categories = np.reshape(categories, [rows, cols])
    rgb_image = cv2.cvtColor(categories, cv2.COLOR_GRAY2RGB)
    for i in range(cols):
        for j in range(rows):
            rgb_image[j, i, :] = colors[categories[j, i]]
    return rgb_image


def SaveDFCTestCLSImg(savePath, resImg, name):
    resImg = np.array(resImg)

    cls_name = savePath + name + 'LEFT_CLS.tif'
    viz_name = savePath + name + 'SEGMENTATION_RGB.tif'
    tifffile.imsave(viz_name, category_to_color(resImg))
    resImg = sequential_to_las_labels(resImg)
    resImg = resImg.astype(np.uint8)
    tifffile.imsave(cls_name, resImg, compress=6)


def FindName(path):
    name = os.path.basename(path)
    pos = name.find('SEGMENTATION_RGB')
    name = name[0:pos]
    return name


def color_to_category(image):
    labels = deepcopy(image)
    labels = labels.astype(np.float32)
    labels[:, :, 2] = labels[:, :, 2] * 5
    img = labels[:, :, 0] + labels[:, :, 1] + labels[:, :, 2]
    print img.shape
    res = deepcopy(img)
    res[:] = 0
    res[img == 417] = 0
    res[img == 128] = 1
    res[img == 255] = 2
    res[img == 1275] = 3
    res[img == 896] = 4
    return res


if __name__ == '__main__':
    root = '/Users/rhc/WorkPlace/Programs/JackNetworkFramework/Submission/'
    path = 'Result/'
    savePath = 'Result_2/'

    name = '*.tif'
    filetype = '.tif'

    files = glob.glob(root + path + name)
    print 'Find the number of file in %s : %d' % (root + path, len(files))

    for i in range(len(files)):
        path = files[i]
        name = FindName(path)
        img = tifffile.imread(path)
        res = color_to_category(img)
        SaveDFCTestCLSImg(root+savePath, res, name)
    print "Finish!"
