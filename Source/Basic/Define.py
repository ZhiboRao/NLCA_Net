# -*- coding: utf-8 -*-
#
# The program's define
# Date: 2018-05-04
# import some lib
import os
import sys
import time
import tensorflow as tf     # tensorflow
import numpy as np          # mat

# the path's define, you can change the content
# output's path
OUTPUT_PATH = './Result/'
# model's path
MODEL_PATH = './Model/'
# result img's path
RESULT_IMG_PATH = './ResultImg/'
# Training list
TRAIN_LIST_PATH = './dataset/trainlist.txt'
# Label list
LABEL_LIST_PATH = './dataset/label.txt'
# log path
LOG_PATH = './log/'
DATASET_NUM = 22


# DateBase define
# image name's format
LEFT_IMAGE_NAME_FORMAT = "training/image_2/%06d_10"
# image name's format
RIGHT_IMAGE_NAME_FORMAT = "training/image_3/%06d_10"
# Save's format
SAVE_FORMAT = "%06d_10"
# image's type
IMAGE_TYPE = ".png"
# image's width
IMAGE_WIDTH = 512
# image's height
IMAGE_HEIGHT = 256
# image's width
IMAGE_ORG_WIDTH = 1280
# image's height
IMAGE_ORG_HEIGHT = 384
# image's depth
IMG_DEPTH = 3
# image's num
IMAGE_NUM = 20000

# program setting
IMG_DISPARITY = 192                                     # the image's disparity
MAX_STEPS = 500000                                      # the max step
LEARNING_RATE = 0.001                                   # learin rate
AUTO_SAVE_NUM = 20                                      # save
MODEL_NAME = "model.ckpt"                               # model name
VAL_SET = 20                                            # every 20 is validation
SHOW_SET = 40                                           # show the res
BATCH_SIZE = 2                                          # BATCH_SIZE
VAL_TIMES = 1                                           # the val times
FILE_NUM = 5
CIFAR_10_LABLE_NUM = 10
GPU_NUM = 1
LABLE_NUM = 6

# network para
# const para
CONV_BLOCK_NUM = 4
DENSE_BLOCK_NUM = 4
RES_3D_BLOCK = 4
FEATURE_NUM = 32
FEATURE_SCALE_NUM = 3
NON_LOCAL_BLOCK_NUM = 4
