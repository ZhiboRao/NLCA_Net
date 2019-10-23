# -*- coding: utf-8 -*-
from JackBasicStructLib.Basic.Define import *


def Conv2D(x, ksize, stride, filters_out):
    with tf.variable_scope("Conv2D"):
        # get the para
        filters_in = x.get_shape()[-1]          # in
        shape = [ksize, ksize, filters_in, filters_out]

        layers = tf.contrib.layers
        weights = tf.get_variable('weights',
                                  shape=shape,
                                  dtype='float32',
                                  initializer=layers.xavier_initializer(),
                                  regularizer=layers.l2_regularizer(
                                      CONV_WEIGHT_DECAY),
                                  collections=[
                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                      GC_VARIABLES],
                                  trainable=True)

        x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

    return x


def DeConv2D(x, ksize, stride, filters_out):
    with tf.variable_scope("DeConv2D"):
        filters_in = x.get_shape()[-1]

        # must have as_list to get a python list!
        x_shape = x.get_shape().as_list()
        batchsize = x_shape[0]
        height = x_shape[1] * stride
        width = x_shape[2] * stride
        strides = [1, stride, stride, 1]
        shape = [ksize, ksize, filters_out, filters_in]

        layers = tf.contrib.layers
        weights = tf.get_variable('weights',
                                  shape=shape,
                                  dtype='float32',
                                  initializer=layers.xavier_initializer(),
                                  regularizer=layers.l2_regularizer(
                                      CONV_WEIGHT_DECAY),
                                  collections=[
                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                      GC_VARIABLES],
                                  trainable=True)

        x = tf.nn.conv2d_transpose(
            x, weights, output_shape=[batchsize, height, width, filters_out],
            strides=strides, padding='SAME')

    return x


def AtrousConv2D(x, ksize, rate, filters_out):
    with tf.variable_scope('AtrousConv2D'):
        filters_in = x.get_shape()[-1]

        shape = [ksize, ksize, filters_in, filters_out]
        layers = tf.contrib.layers

        weights = tf.get_variable('weights',
                                  shape=shape,
                                  dtype='float32',
                                  initializer=layers.xavier_initializer(),
                                  regularizer=layers.l2_regularizer(
                                      CONV_WEIGHT_DECAY),
                                  collections=[
                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                      GC_VARIABLES],
                                  trainable=True)

        x = tf.nn.atrous_conv2d(x, filters=weights, rate=rate, padding='SAME')
        return x


def Conv3D(x, ksize, stride, filters_out):
    with tf.variable_scope("Conv3D"):
        filters_in = x.get_shape()[-1]
        shape = [ksize, ksize, ksize, filters_in, filters_out]

        layers = tf.contrib.layers
        weights = tf.get_variable('weights',
                                  shape=shape,
                                  dtype='float32',
                                  initializer=layers.xavier_initializer(),
                                  regularizer=layers.l2_regularizer(
                                      CONV_WEIGHT_DECAY),
                                  collections=[
                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                      GC_VARIABLES],
                                  trainable=True)

        x = tf.nn.conv3d(
            x, weights, [1, stride, stride, stride, 1], padding='SAME')

    return x


def DeConv3D(x, ksize, stride, filters_out):
    with tf.variable_scope("DeConv3D"):
        filters_in = x.get_shape()[-1]

        # must have as_list to get a python list!
        x_shape = x.get_shape().as_list()
        batchsize = x_shape[0]
        depth = x_shape[1] * stride
        height = x_shape[2] * stride
        width = x_shape[3] * stride
        strides = [1, stride, stride, stride, 1]
        shape = [ksize, ksize, ksize, filters_out, filters_in]

        layers = tf.contrib.layers
        weights = tf.get_variable('weights',
                                  shape=shape,
                                  dtype='float32',
                                  initializer=layers.xavier_initializer(),
                                  regularizer=layers.l2_regularizer(
                                      CONV_WEIGHT_DECAY),
                                  collections=[
                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                      GC_VARIABLES],
                                  trainable=True)

        x = tf.nn.conv3d_transpose(
            x, weights, output_shape=[batchsize, depth, height, width, filters_out],
            strides=strides, padding='SAME')

    return x


def BN(x, training=True):
    with tf.variable_scope("BN"):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]
        axis = list(range(len(x_shape) - 1))

        beta = tf.get_variable('beta',
                               shape=params_shape,
                               initializer=tf.zeros_initializer(),
                               dtype='float32',
                               collections=[
                                   tf.GraphKeys.GLOBAL_VARIABLES,
                                   GC_VARIABLES],
                               trainable=True)
        gamma = tf.get_variable('gamma',
                                shape=params_shape,
                                initializer=tf.ones_initializer(),
                                dtype='float32',
                                collections=[
                                    tf.GraphKeys.GLOBAL_VARIABLES,
                                    GC_VARIABLES],
                                trainable=True)

        moving_mean = tf.get_variable('moving_mean',
                                      shape=params_shape,
                                      initializer=tf.zeros_initializer(),
                                      dtype='float32',
                                      collections=[
                                          tf.GraphKeys.GLOBAL_VARIABLES,
                                          GC_VARIABLES],
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance',
                                          shape=params_shape,
                                          initializer=tf.ones_initializer(),
                                          dtype='float32',
                                          collections=[
                                              tf.GraphKeys.GLOBAL_VARIABLES,
                                              GC_VARIABLES],
                                          trainable=False)

        # These ops will only be performed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean,
                                                                   BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        training = tf.convert_to_tensor(training,
                                        dtype='bool',
                                        name='is_training')

        mean, variance = control_flow_ops.cond(
            training, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance,
                                      beta, gamma, BN_EPSILON)
    return x


def Biased(x):
    with tf.variable_scope("Biased"):
        filters_out = x.get_shape()[-1]
        bias = tf.get_variable(
            'bias',
            shape=[filters_out],
            dtype='float32',
            initializer=tf.constant_initializer(0.05, dtype='float32'),
            trainable=True)
        x = tf.nn.bias_add(x, bias)

    return x


def FullConnect(x, filters_out):
    with tf.variable_scope("FullConnect"):
        filters_in = x.get_shape()[-1]
        shape = [filters_in, filters_out]

        layers = tf.contrib.layers
        weights = tf.get_variable('weights',
                                  shape=shape,
                                  dtype='float32',
                                  initializer=layers.xavier_initializer(),
                                  regularizer=layers.l2_regularizer(
                                      CONV_WEIGHT_DECAY),
                                  collections=[
                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                      GC_VARIABLES],
                                  trainable=True)

        x = tf.matmul(x, weights)
    return x


def GN(x, group_num):
    with tf.variable_scope('GN'):
        G = group_num
        # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
        x = tf.transpose(x, [0, 3, 1, 2])
        _, C, H, W = x.get_shape().as_list()

        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + GN_EPSILON)

        # per channel gamma and beta
        beta = tf.get_variable('beta',
                               shape=[1, C, 1, 1],
                               initializer=tf.zeros_initializer(),
                               dtype='float32',
                               collections=[
                                   tf.GraphKeys.GLOBAL_VARIABLES,
                                   GC_VARIABLES],
                               trainable=True)
        gamma = tf.get_variable('gamma',
                                shape=[1, C, 1, 1],
                                initializer=tf.ones_initializer(),
                                dtype='float32',
                                collections=[
                                    tf.GraphKeys.GLOBAL_VARIABLES,
                                    GC_VARIABLES],
                                trainable=True)
        #gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
        #beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
        #gamma = tf.reshape(gamma, [1, C, 1, 1])
        #beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta

        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
    return output


def MISH(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))
