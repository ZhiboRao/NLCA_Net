# -*- coding: utf-8 -*-
import os
import glob


# define sone struct
RootPath = '/home1/Documents/Database/'  # root path
FolderNameFormat = '%04d/'
RawDataFolder = 'frames_cleanpass/TRAIN/%s/'
LableFolder = 'disparity/TRAIN/%s/'
LeftFolder = 'left/'
RightFolder = 'right/'
FileName = '%s%04d'
RawDataType = '.png'
LabelType = '.pfm'
TrainListPath = './Dataset/trainlist_scene_flow.txt'
LabelListPath = './Dataset/label_scene_flow.txt'
FolderNum = 750
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
    return RootPath + RawDataFolder % folderId + FolderNameFormat % folderNum + \
        FileName % (fileFolder, num) + RawDataType


def GenLabelPath(folderId, folderNum, fileFolder, num):
    return RootPath + LableFolder % folderId + FolderNameFormat % folderNum + \
        FileName % (fileFolder, num) + LabelType


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
                num += 1
                total += 1
    return total


def ProduceList(folder_list, fd_train_list, fd_label_list):
    total = 0
    for i in range(len(folder_list)):
        img_folder_path = RootPath + RawDataFolder % folder_list[i]
        gt_foler_path = RootPath + LableFolder % folder_list[i]

        # print img_folder_path

        left_files = glob.glob(img_folder_path + LeftFolder + '*' + RawDataType)

        for j in range(len(left_files)):
            name = os.path.basename(left_files[j])
            pos = name.find('.png')
            name = name[0:pos]
            # print name

            left_img_path = left_files[j]
            right_img_path = img_folder_path + RightFolder + name + RawDataType
            gt_img_path = gt_foler_path + LeftFolder + name + LabelType

            rawLeftPathisExists = os.path.exists(left_img_path)
            rawRightPathisExists = os.path.exists(right_img_path)
            lablePathisExists = os.path.exists(gt_img_path)

            if (not rawLeftPathisExists) and \
                    (not lablePathisExists) and (not rawRightPathisExists):
                print "\"" + left_img_path + "\"" + "is not exist!!!"
                break

            OutputData(fd_train_list, left_img_path)
            OutputData(fd_train_list, right_img_path)
            OutputData(fd_label_list, gt_img_path)
            total = total + 1

    return total


def GenList_2(fd_train_list, fd_label_list):
    folder_list = ['15mm_focallength/scene_backwards/fast',
                   '15mm_focallength/scene_backwards/slow',
                   '15mm_focallength/scene_forwards/fast',
                   '15mm_focallength/scene_forwards/slow',
                   '35mm_focallength/scene_backwards/fast',
                   '35mm_focallength/scene_backwards/slow',
                   '35mm_focallength/scene_forwards/fast',
                   '35mm_focallength/scene_forwards/slow']
    return ProduceList(folder_list, fd_train_list, fd_label_list)


def GenList_3(fd_train_list, fd_label_list):
    folder_list = ['a_rain_of_stones_x2',
                   'eating_camera2_x2',
                   'eating_naked_camera2_x2',
                   'eating_x2',
                   'family_x2',
                   'flower_storm_augmented0_x2',
                   'flower_storm_augmented1_x2',
                   'flower_storm_x2',
                   'funnyworld_augmented0_x2',
                   'funnyworld_augmented1_x2',
                   'funnyworld_camera2_augmented0_x2',
                   'funnyworld_camera2_augmented1_x2',
                   'funnyworld_camera2_x2',
                   'funnyworld_x2',
                   'lonetree_augmented0_x2',
                   'lonetree_augmented1_x2',
                   'lonetree_difftex2_x2',
                   'lonetree_difftex_x2',
                   'lonetree_winter_x2',
                   'lonetree_x2',
                   'top_view_x2',
                   'treeflight_augmented0_x2',
                   'treeflight_augmented1_x2',
                   'treeflight_x2']
    return ProduceList(folder_list, fd_train_list, fd_label_list)


if __name__ == '__main__':
    fd_train_list, fd_label_list = OpenFile()
    flying_num = GenList(fd_train_list, fd_label_list)
    print flying_num
    driving_num = GenList_2(fd_train_list, fd_label_list)
    print driving_num
    monkey_num = GenList_3(fd_train_list, fd_label_list)
    print monkey_num
    #monkey_num = 0
    total = flying_num + driving_num + monkey_num

    print total
    #folderId = ConvertNumToChar(0)
    #folderNum = 0
    #num = 6
    #rawLeftPath = GenRawPath(folderId, folderNum, LeftFolder, num)
    # print rawLeftPath
