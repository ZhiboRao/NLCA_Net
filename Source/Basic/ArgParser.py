# -*- coding: utf-8 -*-
#
# import the library
from Define import *
#from LogHandler import *
import argparse


def Str2Bool(arg):
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parse the train model's para
def ParseArgs():
    parser = argparse.ArgumentParser(
        description="Tune stereo matching network")
    # the path setting
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--dataset', default='KITTI', help='KITTI or FlyingThing')
    parser.add_argument('--datasetNum', default=DATASET_NUM, type=int, help="The number of dataset")
    parser.add_argument('--trainListPath', default=TRAIN_LIST_PATH, help='Train list path')
    parser.add_argument('--trainLabelListPath', default=LABEL_LIST_PATH, help='Label list Path')
    parser.add_argument('--valListPath', default=TRAIN_LIST_PATH, help='val list path')
    parser.add_argument('--valLabelListPath', default=LABEL_LIST_PATH, help='val list Path')
    parser.add_argument('--testListPath', default=TRAIN_LIST_PATH, help='val list path')
    parser.add_argument('--testLabelListPath', default=LABEL_LIST_PATH, help='val list Path')

    # output path
    parser.add_argument('--outputDir', default=OUTPUT_PATH,
                        help="The output's path. e.g. './log/'")
    parser.add_argument('--modelDir', default=MODEL_PATH,
                        help="The model's path. e.g. ./model/")
    parser.add_argument('--resultImgDir', default=RESULT_IMG_PATH,
                        help="The test result img's path. e.g. ./ResultImg/")
    parser.add_argument('--saveFormat', default=SAVE_FORMAT, help="The save format")
    parser.add_argument('--imgType', default=IMAGE_TYPE, help="The image's type")
    parser.add_argument('--log', default=LOG_PATH, help="The log file")
    parser.add_argument('--auto_save_num', type=int, default=AUTO_SAVE_NUM, help='AUTO_SAVE_NUM')

    # program setting
    parser.add_argument('--imgNum', default=IMAGE_NUM,
                        type=int, help="The img's num, training and testing")
    parser.add_argument('--valImgNum', default=IMAGE_NUM,
                        type=int, help="The img's num, val num")
    parser.add_argument('--maxEpochs', default=MAX_STEPS,
                        type=int, help="Max step. e.g. 500")
    parser.add_argument('--gpu', type=int, default=GPU_NUM,
                        help='state the num of gpu: 0, 1, 2 or 3 ...')

    # training setting
    parser.add_argument('--batchSize', type=int, default=BATCH_SIZE, help='Batch Size')
    parser.add_argument('--learningRate', default=LEARNING_RATE,
                        type=float,
                        help="Learning rate. e.g. 0.01, 0.001, 0.0001")
    parser.add_argument('--pretrain', default=False,
                        type=Str2Bool, help='true or false')
    parser.add_argument('--modelName', default='NLCANet', help='model name')

    # the image's size
    parser.add_argument('--corpedImgWidth', default=IMAGE_WIDTH, type=int,
                        help="Image's width. e.g. 512, In the training process is Clipped size")
    parser.add_argument('--corpedImgHeight', default=IMAGE_HEIGHT, type=int,
                        help="Image's width. e.g. 256, In the training process is Clipped size")
    parser.add_argument('--padedImgWidth', default=IMAGE_ORG_WIDTH, type=int,
                        help="Image's width. e.g. 1280,"
                        + "In the testing process is the Expanded size ")
    parser.add_argument('--padedImgHeight', default=IMAGE_ORG_HEIGHT, type=int,
                        help="Image's width. e.g. 384," +
                        "In the testing process is the Expanded size ")
    return parser.parse_args()
