# -*- coding: utf-8 -*-
from Define import *
from ArgParser import *
from LogHandler import *
from JackBasicStructLib.FileProc.FileHandler import *


# build the folder
def InitPro(args):
    # check the space
    Mkdir(args.outputDir)
    InitLog(args.outputDir + LOG_FILE, args.pretrain)
    Info("Finish logHandler init work")
