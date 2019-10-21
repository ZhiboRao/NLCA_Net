# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ModelTemplate(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def GenInputInterface(self):
        return input

    @abstractmethod
    def GenLabelInterface(self):
        return label

    @abstractmethod
    def Inference(self, input, training=True):
        return output

    @abstractmethod
    def Optimizer(self, lr):
        return opt

    @abstractmethod
    def Accuary(self, output, label):
        return acc

    @abstractmethod
    def Loss(self, output, label):
        return loss
