# -*- coding: utf-8 -*-
from Basic.LogHandler import *
from JackBasicStructLib.Model.Template.DataHandlerTemplate import DataHandlerTemplate
from JackBasicStructLib.FileProc.FileHandler import *
from JackBasicStructLib.Dataloader.KittiFlyingDataloader import KittiFlyingDataloader as kfd

import cv2
import numpy as np

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
        self.kfd = kfd()

    def GetTrainingData(self, paras, trainList, num):
        imgLs, imgRs, imgGrounds = self.kfd.GetBatchImage(self.__args, trainList, num)
        input, label = self.__CreateRes(imgLs, imgRs, imgGrounds)
        return input, label

    def GetValData(self, paras, valList, num):
        imgLs, imgRs, imgGrounds = self.kfd.GetBatchImage(self.__args, valList, num, True)
        input, label = self.__CreateRes(imgLs, imgRs, imgGrounds)
        return input, label

    def GetTestingData(self, paras, testList, num):
        imgLs, imgRs, top_pads, left_pads = self.kfd.GetBatchTestImage(
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

    def ShowIntermediateResult(self, epoch, loss, acc):
        format_str = ('e: %d, loss: %.3f, ' +
                      'loss_0: %.3f, loss_1: %.3f, ' +
                      'acc_0: %.3f, acc_1: %.3f')
        return format_str % (epoch, loss[0], loss[1], loss[2],
                                 acc[0], acc[1])

    def SaveResult(self, output, supplement, imgID, testNum):
        args = self.__args
        res = np.array(output)
        top_pads = supplement[0]
        left_pads = supplement[1]

        for i in range(args.gpu):
            for j in range(args.batchSize):
                temRes = res[i, 1, j, :, :]
                temRes = temRes[top_pads[i*args.batchSize+j]:, :-left_pads[i*args.batchSize+j]]
                self.kfd.SaveTestData(args, temRes, args.gpu*args.batchSize *
                                      imgID + i*args.batchSize + j)
                # Info('[TestProcess] Finish ' +
                #     str(args.gpu * args.batchSize*imgID + i*args.batchSize + j) + ' image.')

    def __CreateRes(self, imgLs, imgRs, imgGrounds):
        input = []
        label = []
        input.append(imgLs)
        input.append(imgRs)
        label.append(imgGrounds)
        return input, label

    def __CreateSupplement(self, top_pads, left_pads):
        return [top_pads, left_pads]

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
