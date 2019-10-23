# -*- coding: utf-8 -*-
# The JackLIb
from JackBasicStructLib.Basic.Define import *


class Paras(object):
    def __init__(self, lr, batchsize, gpu, imgNum, valImgNum, maxEpoch,
                 log_path, save_dir, save_name, save_epoch,
                 max_save_num=10, pretrain=False, test_times=1, training=True):
        self.lr = lr
        self.batchsize = batchsize
        self.gpu = gpu
        self.imgNum = imgNum
        self.valImgNum = valImgNum
        self.maxEpoch = maxEpoch
        self.log_path = log_path
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, save_name)
        self.save_epoch = save_epoch
        self.max_save_num = max_save_num
        self.pretrain = pretrain
        self.test_times = test_times
        self.training = training

        Info("The hyperparameters are set as follows:")
        Info('├── lr: ' + str(self.lr))
        Info('├── batchsize: ' + str(self.batchsize))
        Info('├── gpu: ' + str(self.gpu))
        Info('├── imgNum: ' + str(self.imgNum))
        Info('├── valImgNum: ' + str(self.valImgNum))
        Info('├── maxEpoch: ' + str(self.maxEpoch))
        Info('├── log_path: ' + self.log_path)
        Info('├── save_dir: ' + self.save_dir)
        Info('├── save_path: ' + self.save_path)
        Info('├── save_epoch: ' + str(self.save_epoch))
        Info('├── max_save_num: ' + str(self.max_save_num))
        Info('├── pretrain: ' + str(self.pretrain))
        Info('├── test_times: ' + str(self.test_times))
        Info('└── training: ' + str(self.training))
