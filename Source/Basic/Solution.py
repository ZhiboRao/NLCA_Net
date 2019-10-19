# -*- coding: utf-8 -*-
from Define import *
from ArgParser import *
from LogHandler import *
from Output import *

from TrainProcess.TestModel import TestModel
from TrainProcess.TrainModel import TrainModel  # The TrainProcess


# build the folder
def InitPro(args):
    # check the space
    Mkdir(args.outputDir)
    InitLog(args.outputDir + LOG_FILE, args.pretrain)
    Info("Finish init work")


# train
def Train(args):
    Info("Start train work")
    TrainModel(args).Train()
    Info("Finish train work")


# test
def Test(args):
    Info("Start test work")
    Mkdir(args.resultImgDir)
    TestModel(args).Test()
    Info("Finish test work")
