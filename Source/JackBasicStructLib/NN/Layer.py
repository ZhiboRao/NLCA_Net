# -*- coding: utf-8 -*-
from Ops import *


def AvgPooling2D(x, ksize, stride):
    with tf.variable_scope('AvgPooling2D'):
        x = tf.nn.avg_pool(x, [1, ksize, ksize, 1],
                           [1, stride, stride, 1], padding='SAME')
    return x


def AvgPooling3D(x, ksize, stride):
    with tf.variable_scope('AvgPooling3D'):
        x = tf.nn.avg_pool3d(x, [1, ksize, ksize, ksize, 1], [
            1, stride, stride, stride, 1], padding='SAME')
    return x


def MaxPooling2D(x, ksize, stride):
    with tf.variable_scope('MaxPooling2D'):
        x = tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], padding='SAME')
    return x


def MaxPooling3D(x, ksize, stride):
    with tf.variable_scope('MaxPooling3D'):
        x = tf.nn.max_pool3d(x, [1, ksize, ksize, ksize, 1], [
            1, stride, stride, stride, 1], padding='SAME')
    return x


def BiasedBnActiveLayer(x, biased=False, bn=True, relu=True, training=True):
    with tf.variable_scope("BiasedBnActiveLayer"):
        if biased == True:
            x = Biased(x)

        if bn == True:
            x = BN(x, training)

        if relu == True:
            #x = tf.nn.relu(x)
            x = MISH(x)

    return x


def Conv2DLayer(x, ksize, stride, filters_out, name,
                biased=False, bn=True, relu=True, training=True):

    with tf.variable_scope(name):
        x = Conv2D(x, ksize, stride, filters_out)
        x = BiasedBnActiveLayer(x, biased, bn, relu, training)

    return x


def AtrousConv2DLayer(x, ksize, rate, filters_out, name,
                      biased=False, bn=True, relu=True, training=True):

    with tf.variable_scope(name):
        x = AtrousConv2D(x, ksize, rate, filters_out)
        x = BiasedBnActiveLayer(x, biased, bn, relu, training)

    return x


def DeConv2DLayer(x, ksize, stride, filters_out, name,
                  biased=False, bn=True, relu=True, training=True):

    with tf.variable_scope(name):
        x = DeConv2D(x, ksize, stride, filters_out)
        x = BiasedBnActiveLayer(x, biased, bn, relu, training)

    return x


def Conv3DLayer(x, ksize, stride, filters_out, name,
                biased=False, bn=True, relu=True, training=True):

    with tf.variable_scope(name):
        x = Conv3D(x, ksize, stride, filters_out)
        x = BiasedBnActiveLayer(x, biased, bn, relu, training)

    return x


def DeConv3DLayer(x, ksize, stride, filters_out, name,
                  biased=False, bn=True, relu=True, training=True):

    with tf.variable_scope(name):
        x = DeConv3D(x, ksize, stride, filters_out)
        x = BiasedBnActiveLayer(x, biased, bn, relu, training)

    return x


def FullConnectLayer(x, filters_out, name,
                     biased=True, relu=False, training=True):
    with tf.variable_scope(name):
        x = FullConnect(x, filters_out)
        x = BiasedBnActiveLayer(x, biased, False, relu, training)
    return x


def AvgPoolingConv2DUpsampleLayer(x, ksize, stride, filters_out, name, training=True):
    with tf.variable_scope(name + "/AvgPoolingConv2DUpsampleLayer"):
        _, height, width, _ = x.get_shape()

        x = AvgPooling2D(x, ksize, stride)
        x = Conv2DLayer(x, 1, 1, filters_out, "ConvA", training=training)
        x = tf.image.resize_images(x, [height, width])

    return x
