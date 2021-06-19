# -*- coding: utf-8 -*-
from Basic.Switch import Switch
from Basic.LogHandler import *
from Basic.Define import *
from JackBasicStructLib.Basic.Paras import *
from NLCANet.Model import NLCANet as nlca
from CCANET.Model import CCANet as cca
from NLCANet.Dataloader import *


def Args2Paras(args, is_training):
    return Paras(args.learningRate, args.batchSize,
                  args.gpu, args.imgNum,
                  args.valImgNum, args.maxEpochs,
                  args.log, args.modelDir,
                  MODEL_NAME, args.auto_save_num,
                  10, args.pretrain,
                  1, is_training)


class NetWorkInference(object):
    def __init__(self):
        pass

    def Inference(self, args, is_training=True):
        name = args.modelName
        for case in Switch(name):
            if case('NLCANet'):
                Info("Begin loading NLCA_Net Model")
                paras = Args2Paras(args, is_training)
                model = nlca(args, is_training)
                dataHandler = DataHandler(args)
                break
            if case('CCANet'):
                Info("Begin loading CCA_Net Model")
                paras = Args2Paras(args, is_training)
                model = cca(args, is_training)
                dataHandler = DataHandler(args)
                break
            if case():
                Error('NetWork Type Error!!!')

        return paras, model, dataHandler
