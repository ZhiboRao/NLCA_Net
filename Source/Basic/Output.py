# -*- coding: utf-8 -*-
from Define import *
from PIL import Image
import cv2
import tifffile
from copy import deepcopy
import math


# new folder
def Mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    # check the file pat
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

    return


# Create the result file
def CreateResultFile(args):
    # create the dir
    Mkdir(args.outputDir)
    Mkdir(args.resultImgDir)

    if args.phase == 'train':
        # four file
        train_loss = args.outputDir + TRAIN_LOSS_FILE
        train_acc = args.outputDir + TRAIN_ACC_FILE
        val_acc = args.outputDir + VAL_ACC_FILE
        val_loss = args.outputDir + VAL_LOSS_FILE

        # if it is a new train model
        if args.pretrain:
            if os.path.exists(val_acc):
                os.remove(val_acc)
            if os.path.exists(train_loss):
                os.remove(train_loss)
            if os.path.exists(train_acc):
                os.remove(train_acc)
            if os.path.exists(val_loss):
                os.remove(val_loss)

        fd_train_acc = open(train_acc, 'a')
        fd_train_loss = open(train_loss, 'a')
        fd_val_acc = open(val_acc, 'a')
        fd_val_loss = open(val_loss, 'a')

        return fd_train_acc, fd_train_loss, fd_val_acc, fd_val_loss
    else:
        test_acc = args.outputDir + TEST_ACC_FILE
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        return fd_test_acc


# create the output file
def GenerateOutImgPath(dirPath, filenameFormat, imgType, num):
    path = dirPath + filenameFormat % num + imgType
    return path


# save the data
def SaveTestData(args, resImg, num):
    path = GenerateOutImgPath(args.resultImgDir, args.saveFormat, args.imgType, num)
    imgArray = DepthToImgArray(resImg)
    SavePngImg(imgArray, path)


# write file
def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


# change the data
def DepthToImgArray(mat):
    mat = np.array(mat)
    # mat = mat.reshape(mat.shape[1], mat.shape[2])
    imgArray = (mat * float(DEPTH_DIVIDING)).astype(np.uint16)
    return imgArray


# save the png file
def SavePngImg(img, path):
    cv2.imwrite(path, img)


# save image
def SaveImg(img, path):
    img.save(path)


# save from the img
def SaveArray2Img(mat, path):
    img = Image.fromarray(mat)
    SaveImg(img, path)


def Normal(att_map):
    att_map = (att_map - np.min(att_map)) / (np.max(att_map) - np.min(att_map))
    return att_map


def DrawAttentionMap(att_map, rgb_map, rate=0.4):
    # print rgb_map.shape
    # print att_map.shape
    height, width, _ = rgb_map.shape
    #att_map = (att_map - np.min(att_map)) / (np.max(att_map) - np.min(att_map))
    att_map = 1 - att_map
    att_map = 255 * att_map
    att_map = att_map.astype(np.uint8)
    # att_map =
    # print att_map.shape
    att_map = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)
    att_map = cv2.resize(att_map, (height, width), interpolation=cv2.INTER_CUBIC)
    # print att_map.shape
    heatmap = att_map * rate + rgb_map * (1 - rate)
    heatmap = heatmap.astype(np.uint8)
    return heatmap


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


def SaveDFCTestCLSImg(args, resImg, name):
    resImg = np.array(resImg)

    cls_name = args.resultImgDir + name + 'LEFT_CLS.tif'
    viz_name = args.resultImgDir + name + 'SEGMENTATION_RGB.tif'
    tifffile.imsave(viz_name, category_to_color(resImg))
    resImg = sequential_to_las_labels(resImg)
    resImg = resImg.astype(np.uint8)
    tifffile.imsave(cls_name, resImg, compress=6)


def SaveDFCTestDispImg(args, resImg, name):
    resImg = np.array(resImg)

    dsp_name = args.resultImgDir + name + 'LEFT_DSP.tif'
    viz_name = args.resultImgDir + name + 'STEREO_GRAY.tif'

    tifffile.imsave(dsp_name, resImg, compress=6)

    # save grayscale version of image for visual inspection
    resImg = resImg - resImg.min()
    resImg = ((resImg / resImg.max()) * 255.0).astype(np.uint8)
    resImg = cv2.cvtColor(resImg, cv2.COLOR_GRAY2RGB)
    tifffile.imsave(viz_name,  resImg, compress=6)


def SaveDFCAttentionImg(args, resImg, rgbImg, name):
    resImg = np.array(resImg)
    att_name = args.resultImgDir + name + 'Attention_Map.png'
    #resImg = Normal(resImg)
    heatmap = DrawAttentionMap(resImg, rgbImg)
    SavePngImg(heatmap, att_name)


def Sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def Softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis)


def SaveDspAttentionImg(args, resImg, rgbImg, name):
    resImg = np.array(resImg)
    folder_name = args.resultImgDir + name + 'Dsp_Attention_Map/'
    Mkdir(folder_name)

    print resImg.shape
    #resImg = Normal(resImg)
    # print resImg.shape[2]
    #resImg = softmax(resImg, axis=2)

    img_num = resImg.shape[2]
    for i in range(img_num):
        temImg = resImg[:, :, i]
        temImg = Normal(temImg)
        temImg = np.expand_dims(temImg, axis=2)
        print temImg.shape
        att_name = folder_name + name + 'Attention_Map_%d.png' % i
        heatmap = DrawAttentionMap(temImg, rgbImg)
        SavePngImg(heatmap, att_name)


def SaveDspAttentionImg_1(args, resImg, rgbImg, name):
    resImg = np.array(resImg)
    folder_name = args.resultImgDir + name + 'Dsp_Attention_Map/'
    Mkdir(folder_name)

    print resImg.shape
    res = np.zeros((resImg.shape[0], resImg.shape[1], 1))

    # print resImg.shape[2]
    res = softmax(res, axis=2)
    img_num = resImg.shape[2]
    for i in range(img_num):
        temImg = resImg[:, :, i]
        temImg = np.expand_dims(temImg, axis=2)
        res = res + temImg * (i + 1)

    # print temImg.shape
    att_name = folder_name + name + 'Attention_Map.png'
    res = Normal(res)
    heatmap = DrawAttentionMap(res, rgbImg)
    SavePngImg(heatmap, att_name)


if __name__ == "__main__":
    a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 2, 3], [9, 5, 6]]])
    print a.shape
    print a
    b = softmax(a, axis=0)
    print b

    print a
    b = Softmax(a)
    print b
