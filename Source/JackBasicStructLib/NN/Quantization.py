# -*- coding: utf-8 -*-


class quantization(object):
    """docstring for quantization"""

    def __init__(self, nbit=8):
        super(quantization, self).__init__()
        self.__nbit = nbit

    def quantify(self, w):
        with tf.variable_scope("Conv2D"):
