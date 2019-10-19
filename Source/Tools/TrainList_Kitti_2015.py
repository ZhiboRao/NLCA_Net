# -*- coding: utf-8 -*-
import os


# define sone struct
RootPath = '/home1/Documents/Database/'  # root path
RawDataFolder = 'Kitti/training/%s/'
LeftFolder = 'image_2'
RightFolder = 'image_3'
LableFolder = 'disp_occ_0'
FileName = '%06d_10'
RawDataType = '.png'
LabelType = '.png'
TrainListPath = './Dataset/trainlist_kitti_2015.txt'
LabelListPath = './Dataset/labellist_kitti_2015.txt'
ValTrainListPath = './Dataset/val_trainlist_kitti_2015.txt'
ValLabelListPath = './Dataset/val_labellist_kitti_2015.txt'
ImgNum = 200
Times = 5


def GenRawPath(fileFolder, num):
    path = RootPath + RawDataFolder % fileFolder + FileName % num + \
        RawDataType
    return path


def OpenFile():
    if os.path.exists(TrainListPath):
        os.remove(TrainListPath)
    if os.path.exists(LabelListPath):
        os.remove(LabelListPath)
    if os.path.exists(ValTrainListPath):
        os.remove(ValTrainListPath)
    if os.path.exists(ValLabelListPath):
        os.remove(ValLabelListPath)

    fd_train_list = open(TrainListPath, 'a')
    fd_label_list = open(LabelListPath, 'a')
    fd_val_train_list = open(ValTrainListPath, 'a')
    fd_val_label_list = open(ValLabelListPath, 'a')

    return fd_train_list, fd_label_list, fd_val_train_list, fd_val_label_list


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def GenList(fd_train_list, fd_label_list, fd_val_train_list, fd_val_label_list):

    total = 0
    for num in xrange(ImgNum):

        rawLeftPath = GenRawPath(LeftFolder, num)
        rawRightPath = GenRawPath(RightFolder, num)
        lablePath = GenRawPath(LableFolder, num)

        rawLeftPathisExists = os.path.exists(rawLeftPath)
        rawRightPathisExists = os.path.exists(rawRightPath)
        lablePathisExists = os.path.exists(lablePath)

        if (not rawLeftPathisExists) and \
                (not lablePathisExists) and (not rawRightPathisExists):
            break

        if num % Times == 0:
            OutputData(fd_val_train_list, rawLeftPath)
            OutputData(fd_val_train_list, rawRightPath)
            OutputData(fd_val_label_list, lablePath)
        else:
            OutputData(fd_train_list, rawLeftPath)
            OutputData(fd_train_list, rawRightPath)
            OutputData(fd_label_list, lablePath)

        total = total + 1

    return total


if __name__ == '__main__':
    fd_train_list, fd_label_list, fd_val_train_list, fd_val_label_list = OpenFile()
    total = GenList(fd_train_list, fd_label_list, fd_val_train_list, fd_val_label_list)
    print total
    #folderId = ConvertNumToChar(0)
    #folderNum = 0
    #num = 6
    #rawLeftPath = GenRawPath(folderId, folderNum, LeftFolder, num)
    # print rawLeftPath
