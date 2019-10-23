# -*- coding: utf-8 -*-
from JackBasicStructLib.Basic.Define import *
from JackBasicStructLib.Basic.Paras import *


class DataHandlerTemplate(object):
    """docstring for DataHandlerTemplate"""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(DataHandlerTemplate, self).__init__()
        pass

    @abstractmethod
    def GetTrainingData(self, paras, trainList, num):
        return input, label

    @abstractmethod
    def GetValData(self, paras, valList, num):
        return input, label

    @abstractmethod
    def GetTestingData(self, paras, testList, num):
        return input, supplement

    @abstractmethod
    def ShowTrainingResult(self, epoch, loss, acc, duration):
        pass

    @abstractmethod
    def ShowValResult(self, epoch, loss, acc, duration):
        pass

    @abstractmethod
    def SaveResult(self, output, supplement, imgID, testNum):
        pass
