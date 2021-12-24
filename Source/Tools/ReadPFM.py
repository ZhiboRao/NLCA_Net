# -*- coding: utf-8 -*-
import re
import numpy as np
import sys
from PIL import Image
import cv2


def readPFM(file):
    with open(file, 'rb') as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data, scale


def writePfm(file, image, scale=1):
    with open(file, mode='wb') as file:
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        file.write('PF\n' if color else 'Pf\n')
        file.write('%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        file.write('%f\n' % scale)

        image_string = image.tostring()
        file.write(image_string)


if __name__ == '__main__':
    #f = '/Users/rhc/scan9_train/depth_map_0000.pfm'
    f = '/Users/rhc/Documents/depths_mvsnet_rao/%08d_prob.pfm'
    fileName = f % 0
    img, _ = readPFM(fileName)
    print img
    # for i in xrange(49):
    #    fileName = f % i
    #    img, _ = readPFM(fileName)
    #    print i
    #    print img.shape
    #    img = np.ones((128, 160), np.float32)
    #    writePfm(fileName, img)
    # print img.shape
    # print img.shape
    #imgArray = (img * float(256.0)).astype(np.uint16)
    #writePfm("/Users/rhc/1.pfm", img)
    #cv2.imwrite("/Users/rhc/scan9_train/depth_map_0000.png", imgArray)
    #img = Image.fromarray(img)
    # print np.array(img)
    # if img.mode != 'RGB':
    #    img = img.convert('RGB')
    # img.save('/Users/rhc/Downloads/0000/left/0006.png')
