# -*- coding: utf-8 -*-
# The JackLIb
import tensorflow as tf
import numpy as np
import random
import time
import cv2
import sys
import re
import os

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from abc import ABCMeta, abstractmethod
from PIL import Image

from Basic.LogHandler import *


# img setting
IMG_DEPTH = 3

# the basic para
CONV_WEIGHT_DECAY = 0.00001
GC_VARIABLES = 'gc_variables'
MOVING_AVERAGE_DECAY = 0.99
BN_DECAY = MOVING_AVERAGE_DECAY
UPDATE_OPS_COLLECTION = 'gc_update_ops'  # training ops
BN_EPSILON = 1e-9
GN_EPSILON = 1e-9
EPSILON = 1e-9
