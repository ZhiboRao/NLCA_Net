# -*- coding: utf-8 -*-
from Basic.Switch import Switch
from Basic.LogHandler import *
from NLCANet.Model import NLCANet as nlca


class StereoMatchingNetWorks(object):
    def __init__(self):
        pass

    def Inference(self, name, args, is_training=True):
        for case in Switch(name):
            if case('NLCANet'):
                res = nlca(args, is_training)
                break
            if case('GCNet'):
                Info("This is GCNet")
                #res = ResNet().NetWork(x, is_training)
                break
            if case():
                Error('NetWork Type Error!!!')

        return res
