# -*- coding: utf-8 -*-
from JackBasicStructLib.Basic.Define import *


def ListAdd(listA, listB):
    assert len(listA) == len(listB)

    return [listA[i] + listB[i] for i in range(len(listA))]


def ListDiv(listA, num):
    return [listA[i] / num for i in range(len(listA))]


def ListMean(listA):
    res = []
    for i in range(len(listA)):
        tem_res = tf.expand_dims(listA[i], 0)
        res.append(tem_res)

    res = tf.concat(axis=0, values=res)
    res = tf.reduce_mean(res, 0)

    return res


def ListMean_1(listA):
    num = len(listA)

    res = listA[0]
    for i in range(num - 1):
        res = ListAdd(res, listA[i + 1])
    res = ListDiv(res, num)

    return res


def NumpyListMean(listA):
    num = len(listA)
    res = np.array(listA[0])

    for i in range(num - 1):
        res = res + np.array(listA[i + 1])

    res = res / num

    return res
