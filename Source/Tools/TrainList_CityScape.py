# -*- coding: utf-8 -*-

import os
import glob


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


TrainListPath = './Dataset/trainlist_CityScape.txt'
CLSLabelListPath = './Dataset/labellist_cls_CityScape.txt'
DispLabelListPath = './Dataset/labellist_disp_CityScape.txt'

ValTrainListPath = './Dataset/val_trainlist_CityScape.txt'
ValDispLabelListPath = './Dataset/val_label_disp_list_CityScape.txt'
ValClsLabelListPath = './Dataset/val_label_cls_list_CityScape.txt'


RootPath = '/home2/Documents/CityScape/'


cls_folder_list = ['leftImg8bit/', 'rightImg8bit/', 'gtCoarse/', 'disparity/']
train_folder_list = ['train/aachen', 'train/bochum', 'train/bremen', 'train/cologne',
                     'train/darmstadt', 'train/dusseldorf', 'train/erfurt', 'train/hamburg',
                     'train/hanover', 'train/jena', 'train/krefeld', 'train/monchengladbach',
                     'train/strasbourg', 'train/stuttgart', 'train/tubingen', 'train/ulm',
                     'train/weimar', 'train/zurich',
                     'train_extra/augsburg', 'train_extra/bad-honnef', 'train_extra/bamberg',
                     'train_extra/bayreuth', 'train_extra/dortmund', 'train_extra/dresden',
                     'train_extra/duisburg', 'train_extra/erlangen', 'train_extra/freiburg',
                     'train_extra/heidelberg', 'train_extra/heilbronn', 'train_extra/karlsruhe',
                     'train_extra/konigswinter', 'train_extra/konstanz', 'train_extra/mannheim',
                     'train_extra/muhlheim-ruhr', 'train_extra/nuremberg', 'train_extra/oberhausen',
                     'train_extra/saarbrucken', 'train_extra/schweinfurt', 'train_extra/troisdorf',
                     'train_extra/wuppertal', 'train_extra/wurzburg']

val_folder_list = ['val/frankfurt', 'val/lindau', 'val/munster']


if os.path.exists(TrainListPath):
    os.remove(TrainListPath)

if os.path.exists(CLSLabelListPath):
    os.remove(CLSLabelListPath)

if os.path.exists(DispLabelListPath):
    os.remove(DispLabelListPath)

if os.path.exists(ValTrainListPath):
    os.remove(ValTrainListPath)

if os.path.exists(ValDispLabelListPath):
    os.remove(ValDispLabelListPath)

if os.path.exists(ValClsLabelListPath):
    os.remove(ValClsLabelListPath)


fd_train_list = open(TrainListPath, 'a')
fd_cls_label_list = open(CLSLabelListPath, 'a')
fd_disp_label_list = open(DispLabelListPath, 'a')

fd_val_train_list = open(ValTrainListPath, 'a')
fd_val_cls_label_list = open(ValClsLabelListPath, 'a')
fd_val_disp_label_list = open(ValDispLabelListPath, 'a')


for i in range(len(train_folder_list)):
    path = RootPath + cls_folder_list[0] + train_folder_list[i]
    files = glob.glob(path + '/*.png')
    for j in range(len(files)):
        filename = files[j]
        pos = filename.find(train_folder_list[i])
        filename = filename[pos + len(train_folder_list[i])+1:-15]
        # print filename
        # break

        path_0 = RootPath + cls_folder_list[0] + \
            train_folder_list[i] + '/' + filename + 'leftImg8bit.png'
        path_1 = RootPath + cls_folder_list[1] + \
            train_folder_list[i] + '/' + filename + 'rightImg8bit.png'
        path_2 = RootPath + cls_folder_list[2] + \
            train_folder_list[i] + '/' + filename + 'gtCoarse_labelIds.png'
        path_3 = RootPath + cls_folder_list[3] + \
            train_folder_list[i] + '/' + filename + 'disparity.png'

        exist_0 = os.path.exists(path_0)
        exist_1 = os.path.exists(path_1)
        exist_2 = os.path.exists(path_2)
        exist_3 = os.path.exists(path_3)

        if (not exist_0) or \
            (not exist_1) or \
            (not exist_2) or \
                (not exist_3):
            print "'" + path_0 + "' : is not existed!"
            print "'" + path_1 + "' : is not existed!"
            print "'" + path_2 + "' : is not existed!"
            print "'" + path_3 + "' : is not existed!"
            print '***************'
            break

        OutputData(fd_train_list, path_0)
        OutputData(fd_train_list, path_1)
        OutputData(fd_cls_label_list, path_2)
        OutputData(fd_disp_label_list, path_3)
    print "Finish: " + train_folder_list[i]


for i in range(len(val_folder_list)):
    path = RootPath + cls_folder_list[0] + val_folder_list[i]
    files = glob.glob(path + '/*.png')
    for j in range(len(files)):
        filename = files[j]
        pos = filename.find(val_folder_list[i])
        filename = filename[pos + len(val_folder_list[i])+1:-15]
        # print filename
        # break

        path_0 = RootPath + cls_folder_list[0] + \
            val_folder_list[i] + '/' + filename + 'leftImg8bit.png'
        path_1 = RootPath + cls_folder_list[1] + \
            val_folder_list[i] + '/' + filename + 'rightImg8bit.png'
        path_2 = RootPath + cls_folder_list[2] + \
            val_folder_list[i] + '/' + filename + 'gtCoarse_labelIds.png'
        path_3 = RootPath + cls_folder_list[3] + \
            val_folder_list[i] + '/' + filename + 'disparity.png'

        exist_0 = os.path.exists(path_0)
        exist_1 = os.path.exists(path_1)
        exist_2 = os.path.exists(path_2)
        exist_3 = os.path.exists(path_3)

        if (not exist_0) or \
            (not exist_1) or \
            (not exist_2) or \
                (not exist_3):
            print "'" + path_0 + "' : is not existed!"
            print "'" + path_1 + "' : is not existed!"
            print "'" + path_2 + "' : is not existed!"
            print "'" + path_3 + "' : is not existed!"
            print '***************'
            break

        OutputData(fd_val_train_list, path_0)
        OutputData(fd_val_train_list, path_1)
        OutputData(fd_val_cls_label_list, path_2)
        OutputData(fd_val_disp_label_list, path_3)

    print "Finish: " + val_folder_list[i]

# if __name__ == '__main__':
