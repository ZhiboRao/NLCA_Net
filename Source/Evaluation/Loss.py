# -*- coding: utf-8 -*-
from Basic.Define import *

# Test the code of loss
#import tensorflow as tf
#import numpy as np
#IMG_DISPARITY = 192

LOSS_EPSILON = 1e-7
NO_DATA = 0


def MAE_Loss(result, labels):
    loss_ = tf.abs(tf.subtract(result, labels))

    # check the right map
    mask1 = tf.cast(labels <= IMG_DISPARITY, dtype=tf.bool)
    loss_ = tf.where(mask1, loss_, tf.zeros_like(loss_))
    mask2 = tf.cast(labels > NO_DATA, dtype=tf.bool)
    loss_ = tf.where(mask2, loss_, tf.zeros_like(loss_))

    # calu the num
    mask = tf.logical_and(mask1, mask2)
    loss_sum = tf.reduce_sum(loss_)
    mask = tf.cast(mask, tf.float32)
    # get the l2
    # regularization_losses = tf.get_collection(
    #    tf.GraphKeys.REGULARIZATION_LOSSES)

    # get the loss
    #loss_final = tf.add_n([loss_mean] + regularization_losses)
    #loss_final = loss_mean
    return tf.div(loss_sum, tf.reduce_sum(mask) + LOSS_EPSILON)


def L2_loss(loss, alphi=1):

    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)

    # get the loss
    # print len(regularization_losses)

    regularization_losses = [x*alphi for x in regularization_losses]

    return tf.add_n([loss] + regularization_losses)


def Cross_Entropy(res, labels, cls_num):

    raw_prediction = tf.reshape(res, [-1, cls_num])
    # print raw_prediction.get_shape()
    raw_gt = tf.reshape(labels, [-1, ])
    # print raw_gt.get_shape()
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, cls_num - 1)), 1)
    # print indices.get_shape()
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    # print gt.get_shape()
    prediction = tf.gather(raw_prediction, indices)
    # print prediction.get_shape()
    gt = tf.one_hot(gt, depth=cls_num)
    # regularization_losses = tf.get_collection(
    #    tf.GraphKeys.REGULARIZATION_LOSSES)

    #loss_final = tf.add_n([cross_entropy] + regularization_losses)
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt, logits=prediction))


if __name__ == "__main__":
    a = tf.placeholder(tf.float32, shape=(2, 3, 3))
    b = tf.placeholder(tf.float32, shape=(2, 3, 3))
    c = MAE_Loss(a, b)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    m = np.array([[1, 3, 3], [3, 1, 3], [3, 3, 3]])
    m = np.expand_dims(m, axis=0)
    m = np.concatenate((m, m), axis=0)
    n = np.array([[3, 0, 0], [0, 3, 3], [3, 3, 3]])
    n = np.expand_dims(n, axis=0)
    n = np.concatenate((n, n), axis=0)

    x = sess.run([c], feed_dict={a: m, b: n})

    print x
