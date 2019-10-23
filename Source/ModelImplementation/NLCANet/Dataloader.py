# -*- coding: utf-8 -*-
from Basic.LogHandler import *
from JackBasicStructLib.Model.Template.DataHandlerTemplate import DataHandlerTemplate
from JackBasicStructLib.ImgProc.ImgHandler import *
from JackBasicStructLib.ImgProc.DataAugmentation import *
from JackBasicStructLib.FileProc.FileHandler import *

import cv2

# output file setting
DEPTH_DIVIDING = 256.0

TRAIN_ACC_FILE = 'train_acc.csv'                        # acc file's name
TRAIN_LOSS_FILE = 'train_loss.csv'                      # loss file's name
VAL_LOSS_FILE = 'val_loss.csv'                          # val file's name
VAL_ACC_FILE = 'val_acc.csv'                            # val file's name
TEST_ACC_FILE = 'test_acc.csv'                          # test file's name


class DataHandler(DataHandlerTemplate):
    """docstring for DataHandler"""

    def __init__(self, args):
        super(DataHandler, self).__init__()
        self.__args = args
        self.fd_train_acc, self.fd_train_loss, self.fd_val_acc,\
            self.fd_val_loss, self.fd_test_acc = self.__CreateResultFile(args)

    def GetTrainingData(self, paras, trainList, num):
        imgLs, imgRs, imgGrounds = self.__GetBatchImage(self.__args, trainList, num)
        input, label = self.__CreateRes(imgLs, imgRs, imgGrounds)
        return input, label

    def GetValData(self, paras, valList, num):
        imgLs, imgRs, imgGrounds = self.__GetBatchImage(self.__args, valList, num, True)
        input, label = self.__CreateRes(imgLs, imgRs, imgGrounds)
        return input, label

    def GetTestingData(self, paras, testList, num):
        imgLs, imgRs, top_pads, left_pads = self.__GetBatchTestImage(
            self.__args, testList, num, True)
        input, _ = self.__CreateRes(imgLs, imgRs, None)
        supplement = self.__CreateSupplement(top_pads, left_pads)
        return input, supplement

    def ShowTrainingResult(self, epoch, loss, acc, duration):
        format_str = ('[TrainProcess] epochs = %d ,loss = %.6f, ' +
                      'coarse_disp_loss = %.6f, refine_disp_loss = %.6f, ' +
                      'coarse_acc = %.6f, refine_acc = %.6f (%.3f sec/batch)')
        Info(format_str % (epoch, loss[0], loss[1], loss[2],
                           acc[0], acc[1], duration))
        OutputData(self.fd_train_acc, loss[0])
        OutputData(self.fd_train_loss, acc[1])

    def ShowValResult(self, epoch, loss, acc, duration):
        format_str = ('[ValProcess] epochs = %d ,loss = %.6f, ' +
                      'coarse_disp_loss = %.6f, refine_disp_loss = %.6f, ' +
                      'coarse_acc = %.6f, refine_acc = %.6f (%.3f sec/batch)')
        Info(format_str % (epoch, loss[0], loss[1], loss[2],
                           acc[0], acc[1], duration))
        OutputData(self.fd_val_acc, loss[0])
        OutputData(self.fd_val_loss, acc[1])

    def SaveResult(self, output, supplement, imgID, testNum):
        args = self.__args
        res = np.array(output)
        top_pads = supplement[0]
        left_pads = supplement[1]

        for i in range(args.gpu):
            for j in range(args.batchSize):
                temRes = res[i, 1, j, :, :]
                temRes = temRes[top_pads[i*args.batchSize+j]:, :-left_pads[i*args.batchSize+j]]
                self.__SaveTestData(args, temRes, args.gpu*args.batchSize *
                                    imgID + i*args.batchSize + j)
                Info('[TestProcess] Finish ' +
                     str(args.gpu * args.batchSize*imgID + i*args.batchSize + j) + ' image.')

    def __CreateRes(self, imgLs, imgRs, imgGrounds):
        input = []
        label = []
        input.append(imgLs)
        input.append(imgRs)
        label.append(imgGrounds)
        return input, label

    def __CreateSupplement(self, top_pads, left_pads):
        supplement = []
        supplement.append(top_pads)
        supplement.append(left_pads)
        return supplement

    def __CreateResultFile(self, args):
        # create the dir
        Info("Begin create the result folder")
        Mkdir(args.outputDir)
        Mkdir(args.resultImgDir)

        fd_train_acc = OpenLogFile(args.outputDir + TRAIN_LOSS_FILE, args.pretrain)
        fd_train_loss = OpenLogFile(args.outputDir + TRAIN_ACC_FILE, args.pretrain)
        fd_val_acc = OpenLogFile(args.outputDir + VAL_ACC_FILE, args.pretrain)
        fd_val_loss = OpenLogFile(args.outputDir + VAL_LOSS_FILE, args.pretrain)
        fd_test_acc = OpenLogFile(args.outputDir + TEST_ACC_FILE, args.pretrain)

        Info("Finish create the result folder")
        return fd_train_acc, fd_train_loss, fd_val_acc, fd_val_loss, fd_test_acc

    def __GenerateOutImgPath(self, dirPath, filenameFormat, imgType, num):
        path = dirPath + filenameFormat % num + imgType
        return path

    def __DepthToImgArray(self, img):
        img = np.array(img)
        img = (img * float(DEPTH_DIVIDING)).astype(np.uint16)
        return img

        # save the png file
    def __SavePngImg(self, path, img):
        cv2.imwrite(path, img)

    def __SaveTestData(self, args, img, num):
        path = self.__GenerateOutImgPath(args.resultImgDir, args.saveFormat, args.imgType, num)
        img = self.__DepthToImgArray(img)
        self.__SavePngImg(path, img)

    def __GetBatchImage(self, args, randomlist, num, isVal=False):
        for i in xrange(args.batchSize * args.gpu):
            idNum = randomlist[args.batchSize * args.gpu * num + i]

            if isVal == False:
                imgL, imgR, imgGround = self.__RandomCropRawImage(args, idNum)       # get img
            else:
                imgL, imgR, imgGround = self.__ValRandomCropRawImage(args, idNum)       # get img

            if i == 0:
                imgLs = imgL
                imgRs = imgR
                imgGrounds = imgGround
            else:
                imgLs = np.concatenate((imgLs, imgL), axis=0)
                imgRs = np.concatenate((imgRs, imgR), axis=0)
                imgGrounds = np.concatenate((imgGrounds, imgGround), axis=0)

        return imgLs, imgRs, imgGrounds

    def __GetBatchTestImage(self, args, randomlist, num, isVal=False):
        top_pads = []
        left_pads = []
        for i in xrange(args.batchSize * args.gpu):
            idNum = randomlist[args.batchSize * args.gpu * num + i]
            imgL, imgR, top_pad, left_pad = self.__GetPadingTestData(args, idNum)       # get img

            top_pads.append(top_pad)
            left_pads.append(left_pad)
            if i == 0:
                imgLs = imgL
                imgRs = imgR
            else:
                imgLs = np.concatenate((imgLs, imgL), axis=0)
                imgRs = np.concatenate((imgRs, imgR), axis=0)

        return imgLs, imgRs, top_pads, left_pads

        # flying thing groundtrue

    def __ReadRandomPfmGroundTrue(self, path, x, y, w, h):
        imgGround, _ = ReadPFM(path)
        imgGround = ImgGroundSlice(imgGround, x, y, w, h)
        imgGround = np.expand_dims(imgGround, axis=0)
        return imgGround

    def __ReadRandomGroundTrue(self, path, x, y, w, h):
        # kitti groundtrue
        img = Image.open(path)
        imgGround = np.ascontiguousarray(img, dtype=np.float32)/float(DEPTH_DIVIDING)
        imgGround = ImgGroundSlice(imgGround, x, y, w, h)
        imgGround = np.expand_dims(imgGround, axis=0)
        return imgGround

    def __ReadData(self, args, pathL, pathR, pathGround):
        # Flying Things and Kitti
        w = args.corpedImgWidth
        h = args.corpedImgHeight

        # get the img, the random crop
        imgL = ReadImg(pathL)
        imgR = ReadImg(pathR)

        # random crop
        x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w, h)
        imgL = ImgSlice(imgL, x, y, w, h)
        imgL = Standardization(imgL)
        imgL = np.expand_dims(imgL, axis=0)

        # the right img
        imgR = ImgSlice(imgR, x, y, w, h)
        imgR = Standardization(imgR)
        imgR = np.expand_dims(imgR, axis=0)

        # get groundtrue
        if args.dataset == "KITTI":
            imgGround = self.__ReadRandomGroundTrue(pathGround, x, y, w, h)
        else:
            imgGround = self.__ReadRandomPfmGroundTrue(pathGround, x, y, w, h)

        return imgL, imgR, imgGround

    def __RandomCropRawImage(self, args, num):
        # Get path
        pathL = GetPath(args.trainListPath, 2*num+1)
        pathR = GetPath(args.trainListPath, 2*(num + 1))
        pathGround = GetPath(args.trainLabelListPath, num + 1)

        imgL, imgR, imgGround = self.__ReadData(args, pathL, pathR, pathGround)

        return imgL, imgR, imgGround

    # Val Flying Things and Kitti
    def __ValRandomCropRawImage(self, args, num):
        # Get path
        pathL = GetPath(args.valListPath, 2*num+1)
        pathR = GetPath(args.valListPath, 2*(num + 1))
        pathGround = GetPath(args.valLabelListPath, num + 1)
        imgL, imgR, imgGround = self.__ReadData(args, pathL, pathR, pathGround)

        return imgL, imgR, imgGround

    # Padding Img, used in testing
    def __GetPadingTestData(self, args, num):
        pathL = GetPath(args.testListPath, 2*num+1)
        pathR = GetPath(args.testListPath, 2*(num + 1))

        imgL = ReadImg(pathL)
        imgL = Standardization(imgL)
        imgL = np.expand_dims(imgL, axis=0)
        imgR = ReadImg(pathR)
        imgR = Standardization(imgR)
        imgR = np.expand_dims(imgR, axis=0)

        # pading size
        top_pad = args.padedImgHeight - imgL.shape[1]
        left_pad = args.padedImgWidth - imgL.shape[2]

        # pading
        imgL = np.lib.pad(imgL, ((0, 0), (top_pad, 0), (0, left_pad),
                                 (0, 0)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (top_pad, 0), (0, left_pad),
                                 (0, 0)), mode='constant', constant_values=0)

        return imgL, imgR, top_pad, left_pad
