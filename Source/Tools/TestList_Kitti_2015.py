# -*- coding: utf-8 -*-
import os


# define sone struct
RootPath = '/home1/Documents/Database/'  # root path
RawDataFolder = 'Kitti/testing/%s/'
LeftFolder = 'image_2'
RightFolder = 'image_3'

FileName = '%06d_10'
RawDataType = '.png'

TestListPath = './Dataset/testlist_kitti_2015.txt'
ImgNum = 200


def GenRawPath(fileFolder, num):
    path = RootPath + RawDataFolder % fileFolder + FileName % num + \
        RawDataType
    return path


def OpenFile():
    if os.path.exists(TestListPath):
        os.remove(TestListPath)

    fd_test_list = open(TestListPath, 'a')

    return fd_test_list


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def GenList(fd_test_list):

    total = 0
    for num in xrange(ImgNum):

        rawLeftPath = GenRawPath(LeftFolder, num)
        rawRightPath = GenRawPath(RightFolder, num)

        rawLeftPathisExists = os.path.exists(rawLeftPath)
        rawRightPathisExists = os.path.exists(rawRightPath)

        if (not rawLeftPathisExists) and \
                (not rawRightPathisExists):
            break

        OutputData(fd_test_list, rawLeftPath)
        OutputData(fd_test_list, rawRightPath)

        total = total + 1

    return total


if __name__ == '__main__':
    fd_test_list = OpenFile()
    total = GenList(fd_test_list)
    print(total)
    #folderId = ConvertNumToChar(0)
    #folderNum = 0
    #num = 6
    #rawLeftPath = GenRawPath(folderId, folderNum, LeftFolder, num)
    # print rawLeftPath
