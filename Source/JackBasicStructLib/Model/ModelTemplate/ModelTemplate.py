# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ModelTemplate(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def Inference_1(self, args, training=True):
        pass
