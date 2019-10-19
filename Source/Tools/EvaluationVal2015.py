# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
# Test the code of acc
import tensorflow as tf
import linecache

IMG_DISPARITY = 4

ACC_PIXEL_START = 2
ACC_PIXEL_NUM = 4  # 2,3,4,5
ACC_PIXEL = 3
RELATE_ERR = 0.05
ACC_EPSILON = 1e-7
DEPTH_DIVIDING = 256.0

# get the acc from 2 piexl to 5 piexl


def MatchingAcc(result, labels):
    # get the err mat
    err = tf.abs(tf.subtract(result, labels))
    mask = tf.cast(labels > 0, dtype=tf.bool)  # get the goundtrue's vaild points
    # mask2 = tf.cast(labels <= IMG_DISPARITY, dtype=tf.bool)  # get the goundtrue's vaild points
    mask2 = tf.cast(mask, tf.float32)

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

    mas = tf.div(tf.reduce_sum(err * mask2), total_num)
    return totalAcc, mas


def Acc(result, labels):
    res = np.array(result, dtype=np.float32)/float(DEPTH_DIVIDING)
    res = res.reshape(res.shape[0], res.shape[1])

    groundTrue = np.array(labels, dtype=np.float32)/float(DEPTH_DIVIDING)
    groundTrue = groundTrue.reshape(groundTrue.shape[0], groundTrue.shape[1])

    err = abs(res - groundTrue)

    num = 0
    errTotal = 0
    total = 0
    for i in xrange(err.shape[0]):
        for j in xrange(err.shape[1]):
            if groundTrue[i, j] != 0:     # this point is effect
                point = err[i, j]         # get the error
                errTotal = errTotal + point
                if point > ACC_PIXEL and point / groundTrue[i, j] > RELATE_ERR:
                    num = num + 1
                total = total + 1

    acc = 0
    if total != 0:
        acc = num / float(str(total))
        mas = errTotal / float(str(total))
        # diff = result - labels
        # num = np.sum(diff == True)
        # total = diff.shape[0] * diff.shape[1]
        # acc = num / float(str(total))
    return acc, mas


if __name__ == "__main__":
    a = tf.placeholder(tf.float32, shape=(None, None))
    b = tf.placeholder(tf.float32, shape=(None, None))

    acc, mas = MatchingAcc(a, b)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    total = 0
    num = 0
    errTotal = 0
    for i in xrange(40):
        imgPath = './ResultImg/%06d_10.png' % (i)
        groundPath = linecache.getline('./Dataset/val_labellist_kitti_2015.txt', i+1)
        #groundPath = linecache.getline('./Dataset/labellist_kitti_2015.txt', i+1)

        groundPath = groundPath.rstrip("\n")
#        if i % 5 != 0:
        # continue
        img = np.array(Image.open(imgPath), dtype=np.float32)/float(DEPTH_DIVIDING)
        imgGround = np.array(Image.open(groundPath), dtype=np.float32)/float(DEPTH_DIVIDING)

        #img = np.array([[1.0, 4, 3], [4, 5, 6], [7, 8, 9]])
        #imgGround = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])

        # print img.shape

        acc_out, mas_out = sess.run([acc, mas], feed_dict={a: img, b: imgGround})

        total = total + acc_out[1]
        errTotal = errTotal + mas_out
        num = num + 1
        print str(i) + ':' + str(acc_out[1]) + ',' + str(mas_out)

    print 'total :%f,%f' % (total/num, errTotal/num)
