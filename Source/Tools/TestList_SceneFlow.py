# -*- coding: utf-8 -*-
import os


# define sone struct
RootPath = '/home1/Documents/Database/'  # root path
FolderNameFormat = '%04d/'
RawDataFolder = 'frames_cleanpass/TEST/%s/'
LableFolder = 'disparity/TEST/%s/'
LeftFolder = 'left/'
RightFolder = 'right/'
FileName = '%s%04d'
RawDataType = '.png'
LabelType = '.pfm'
TrainListPath = './Dataset/testlist_scene_flow.txt'
LabelListPath = './Dataset/test_label_scene_flow.txt'
FolderNum = 437
IDNum = 3


def ConvertNumToChar(folderId):
    res = 'None'
    if folderId == 0:
        res = 'A'
    elif folderId == 1:
        res = 'B'
    elif folderId == 2:
        res = 'C'
    return res


def GenRawPath(folderId, folderNum, fileFolder, num):
    path = RootPath + RawDataFolder % folderId + FolderNameFormat % folderNum + \
        FileName % (fileFolder, num) + RawDataType
    return path


def GenLabelPath(folderId, folderNum, fileFolder, num):
    path = RootPath + LableFolder % folderId + FolderNameFormat % folderNum + \
        FileName % (fileFolder, num) + LabelType
    return path


def OpenFile():
    if os.path.exists(TrainListPath):
        os.remove(TrainListPath)
    if os.path.exists(LabelListPath):
        os.remove(LabelListPath)

    fd_train_list = open(TrainListPath, 'a')
    fd_label_list = open(LabelListPath, 'a')

    return fd_train_list, fd_label_list


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def GenList(fd_train_list, fd_label_list):
    total = 0
    for Id in xrange(IDNum):
        for folderNum in xrange(FolderNum):
            num = 6
            while True:
                folderId = ConvertNumToChar(Id)
                rawLeftPath = GenRawPath(folderId, folderNum, LeftFolder, num)
                rawRightPath = GenRawPath(folderId, folderNum, RightFolder, num)
                lablePath = GenLabelPath(folderId, folderNum, LeftFolder, num)

                rawLeftPathisExists = os.path.exists(rawLeftPath)
                rawRightPathisExists = os.path.exists(rawRightPath)
                lablePathisExists = os.path.exists(lablePath)

                if (not rawLeftPathisExists) and \
                        (not lablePathisExists) and (not rawRightPathisExists):
                    break

                OutputData(fd_train_list, rawLeftPath)
                OutputData(fd_train_list, rawRightPath)
                OutputData(fd_label_list, lablePath)
                num = num + 1
                total = total + 1
    return total


if __name__ == '__main__':
    fd_train_list, fd_label_list = OpenFile()
    total = GenList(fd_train_list, fd_label_list)
    print total
    #folderId = ConvertNumToChar(0)
    #folderNum = 0
    #num = 6
    #rawLeftPath = GenRawPath(folderId, folderNum, LeftFolder, num)
    # print rawLeftPath
