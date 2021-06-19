# -*- coding: utf-8 -*-
#from Basic.Define import *


# Test the code of acc
import tensorflow as tf
import numpy as np
IMG_DISPARITY = 4

ACC_PIXEL_START = 2
ACC_PIXEL_NUM = 4  # 2,3,4,5
ACC_PIXEL = 3
RELATE_ERR = 0.05
ACC_EPSILON = 1e-7

NO_DATA = 0
# get the acc from 2 piexl to 5 piexl
VOILD_PIXEL = 255


def Mask(labels):
    mask = tf.cast(labels < VOILD_PIXEL, dtype=tf.bool)
    mask = tf.cast(mask, tf.int32)
    labels = tf.multiply(mask, labels)
    return labels


def MatchingAcc(result, labels):
    # get the err mat
    err = tf.abs(tf.subtract(result, labels))
    mask = tf.cast(labels > NO_DATA, dtype=tf.bool)  # get the goundtrue's vaild points
    # mask2 = tf.cast(labels <= IMG_DISPARITY, dtype=tf.bool)  # get the goundtrue's vaild points

    # get the relative_err
    relative_err = tf.div(err, labels)
    mask_relative = tf.cast(relative_err >= RELATE_ERR, dtype=tf.bool)

    totalAcc = []
    # mask = tf.logical_and(mask, mask2)
    mask_point = tf.cast(mask, tf.float32)
    total_num = tf.reduce_sum(mask_point) + ACC_EPSILON
    for i in xrange(ACC_PIXEL_NUM):
        # get the err point
        # get the point: > ACC_PIXEL_START + i
        mask_point = tf.cast(err >= (ACC_PIXEL_START + i), dtype=tf.bool)
        # get the point: ACC_PIXEL_START + i and > 5%
        mask_point = tf.logical_and(mask_point, mask_relative)
        # get the vaild point: ACC_PIXEL_START + i and > 5% and label > 0
        mask_point = tf.logical_and(mask_point, mask)
        mask_point = tf.cast(mask_point, tf.float32)
        # get the acc and append to totalAcc
        acc = tf.div(tf.reduce_sum(mask_point), total_num)
        totalAcc.append(acc)

    return totalAcc


def MatchingAcc_v2(result, labels):
    # get the err mat
    err = tf.abs(tf.subtract(result, labels))
    mask = tf.cast(labels > NO_DATA, dtype=tf.bool)  # get the goundtrue's vaild points

    totalAcc = []
    # mask = tf.logical_and(mask, mask2)
    mask_point = tf.cast(mask, tf.float32)
    total_num = tf.reduce_sum(mask_point) + ACC_EPSILON
    for i in xrange(ACC_PIXEL_NUM):
        # get the err point
        # get the point: > ACC_PIXEL_START + i
        mask_point = tf.cast(err >= (ACC_PIXEL_START + i), dtype=tf.bool)
        # get the point: ACC_PIXEL_START + i and > 5%
        #mask_point = tf.logical_and(mask_point, mask_relative)
        # get the vaild point: ACC_PIXEL_START + i and > 5% and label > 0
        mask_point = tf.logical_and(mask_point, mask)
        mask_point = tf.cast(mask_point, tf.float32)
        # get the acc and append to totalAcc
        acc = tf.div(tf.reduce_sum(mask_point), total_num)
        totalAcc.append(acc)

    return totalAcc


def IoU(predictions, labels, cls_num):
    weights = tf.cast(tf.less_equal(labels, cls_num - 1), tf.int32)
    acc, acc_op = tf.metrics.accuracy(labels, predictions)
    mIoU, mIoU_op = tf.metrics.mean_iou(labels, predictions, cls_num, weights=weights)
    return acc, acc_op, mIoU, mIoU_op


def Acc(result, labels):
    res = np.array(result)
    # res = res.reshape(res.shape[0], res.shape[1])

    groundTrue = np.array(labels)
    # groundTrue = groundTrue.reshape(groundTrue.shape[0], groundTrue.shape[1])

    err = abs(res - groundTrue)
    num = 0
    total = 0
    for k in xrange(err.shape[0]):
        for i in xrange(err.shape[1]):
            for j in xrange(err.shape[2]):
                if groundTrue[k, i, j] != 0:     # this point is effect
                    point = err[k, i, j]         # get the error
                    if point > ACC_PIXEL and point / groundTrue[k, i, j] > RELATE_ERR:
                        num += 1
                    total += 1

    acc = 0
    if total != 0:
        acc = num / float(str(total))
    # diff = result - labels
    # num = np.sum(diff == True)
    # total = diff.shape[0] * diff.shape[1]
    # acc = num / float(str(total))
    return acc, num


def AccClassification(res, labels):
    correct_prediction = tf.equal(tf.argmax(res, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


if __name__ == "__main__":
    a = tf.placeholder(tf.float32, shape=(5, 512, 256))
    b = tf.placeholder(tf.float32, shape=(5, 512, 256))
    c = MatchingAcc(a, b)
    h = c[2]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #m = np.array([[1, 2, 4], [8, 3, 3], [3, 3, 3]])
    #m = np.expand_dims(m, axis=0)
    #m = np.concatenate((m, m), axis=0)
    #n = np.array([[3, 3, 0], [2, 5, 3], [3, 3, 3]])
    #n = np.expand_dims(n, axis=0)
    #n = np.concatenate((n, n), axis=0)
    #
    m = np.random.randint(0, 192, size=(5, 512, 256))
    m = m.astype(np.float32)
    n = np.random.randint(0, 192, size=(5, 512, 256))
    n = n.astype(np.float32)

    x, y = sess.run([c, h], feed_dict={a: m, b: n})

    print x
    print y

    x = Acc(m, n)

    print x
