# -*coding: utf-8 -*-
#
# import the library
from Basic.ArgParser import *
from Basic.Solution import *
from JackBasicStructLib.Proc.Executor import *


# main function
def main(args):
    InitPro(args)

    if args.phase == 'train':
        Executor(args, True).Train()
    else:
        Executor(args, False).Test()


# execute the main function
if __name__ == "__main__":
    args = ParseArgs()      # parse args
    main(args)              # main function
