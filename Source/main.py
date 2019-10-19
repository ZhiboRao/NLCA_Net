# -*coding: utf-8 -*-
#
# import the library
from Basic.Define import *
from Basic.ArgParser import *
from Basic.Solution import *


# main function
def main(args):
    InitPro(args)

    if args.phase == 'train':
        Train(args)
    else:
        Test(args)


# execute the main function
if __name__ == "__main__":
    args = ParseArgs()      # parse args
    main(args)              # main function
