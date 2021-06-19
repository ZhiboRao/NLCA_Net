# -*- coding: utf-8 -*-
from JackBasicStructLib.Basic.Define import *
from JackBasicStructLib.Basic.Paras import *
from Basic.LogHandler import *

import Queue
import thread


class LoadWoker(object):
    """docstring for LoadWoker"""
    __metaclass__ = ABCMeta

    def __init__(self, paras, dataloader):
        super(LoadWoker, self).__init__()
        self._paras = paras
        self._randomTrainingList = range(paras.imgNum)
        self._randomValList = range(paras.valImgNum)
        self._trainQueue = Queue.Queue(maxsize=30)
        self._valQueue = Queue.Queue(maxsize=30)
        self._exitFlag = 0
        self._num_tr_batch = paras.imgNum / paras.batchsize / paras.gpu
        self._num_val_batch = paras.valImgNum / paras.batchsize / paras.gpu
        if paras.training == False:
            self._num_val_batch = 0

        self._dataloader = dataloader

        Info("The total img num : %d , The max batch : %d" %
             (paras.imgNum, self._num_tr_batch))
        Info("The total val img num : %d , The max batch : %d" %
             (paras.valImgNum, self._num_val_batch))

        if self._num_tr_batch > 0:
            thread.start_new_thread(self.__TrainDataGenQueue_Thread, ())
            Info("__TrainDataGenQueue_Thread start work!")
        if self._num_val_batch > 0:
            thread.start_new_thread(self.__ValDataGenQueue_Thread, ())
            Info("_valDataGenQueue_Thread start work!")

    def __del__(self):
        if self._num_tr_batch > 0:
            self._trainQueue.get()
        if self._num_val_batch > 0:
            self._valQueue.get()

        self._exitFlag = 1
        time.sleep(5)
        Info("Dataloader has been deleted")

    def __RandomList(self, randomTrainingList):
        random.shuffle(randomTrainingList)

    def __TrainDataGenQueue_Thread(self):
        i = 0
        while True:
            if self._exitFlag:
                Info("_trainDataGenQueue_Thread safe exited!")
                thread.exit()

            if i >= self._num_tr_batch:
                i = 0
                self.__RandomList(self._randomTrainingList)

            dataList = []
            if self._paras.training == True:
                input, label = self._dataloader.GetTrainingData(self._paras,
                                                                self._randomTrainingList, i)
                dataList.append(input)
                dataList.append(label)
            else:
                input, supplement = self._dataloader.GetTestingData(self._paras,
                                                                    self._randomTrainingList, i)
                dataList.append(input)
                dataList.append(supplement)

            self._trainQueue.put(dataList)

            i += 1

    def __ValDataGenQueue_Thread(self):
        i = 0
        while True:
            if self._exitFlag:
                Info("_valDataGenQueue_Thread safe exited!")
                thread.exit()

            if i >= self._num_val_batch:
                i = 0

            input, label = self._dataloader.GetValData(self._paras,
                                                       self._randomValList, i)

            dataList = [input, label]
            self._valQueue.put(dataList)
            i += 1

    def GetTrainData(self):
        dataList = self._trainQueue.get()
        return dataList[0], dataList[1]

    def GetValData(self):
        dataList = self._valQueue.get()
        return dataList[0], dataList[1]

    def GetMaxBatch(self):
        return self._num_tr_batch, self._num_val_batch
