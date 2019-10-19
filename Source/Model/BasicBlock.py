# -*- coding: utf-8 -*-
from JackBasicStructLib.JackBasicLayer import *
from JackBasicStructLib.JackClassicalBlock import *


def ExtractUnaryFeatureBlock1(x, training=True):
    with tf.variable_scope("ExtractUnaryFeatureBlock1"):
        x = Conv2DLayer(x, 3, 2, 32, "Conv_1", training=training)
        x = Conv2DLayer(x, 3, 1, 32, "Conv_2", training=training)
        x = Conv2DLayer(x, 3, 1, 32, "Conv_3", training=training)

        res_block_num = 3
        for i in range(res_block_num):
            x = Res2DBlock(x, 3, "Res_" + str(i), training=training)

    return x


def ExtractUnaryFeatureBlock2(x, training=True):
    with tf.variable_scope("ExtractUnaryFeatureBlock2"):
        shortcut = Conv2DLayer(x, 3, 2, 64, "Conv_1", training=training)
        x = Conv2DLayer(x, 1, 2, 64, "Conv_w", training=training)
        x = x + shortcut

        res_block_num = 16
        for i in range(res_block_num):
            x = Bottleneck2DBlock(x, "BottleNeck_" + str(i), training=training)

    return x


def ExtractUnaryFeatureBlock3(x, training=True):
    with tf.variable_scope("ExtractUnaryFeatureBlock3"):
        x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
        x = ResAtrousBlock(x, 3, 2, "Atrous_1", training=training)

    return x


def ExtractUnaryFeatureBlock4(x, training=True):
    with tf.variable_scope("ExtractUnaryFeatureBlock4"):
        x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
        x = ResAtrousBlock(x, 3, 4, "Atrous_1", training=training)
    return x


def ExtractUnaryFeatureBlock5(x, training=True):
    with tf.variable_scope("ExtractUnaryFeatureBlock5"):
        x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
        x = Conv2DLayer(x, 3, 1, 64, "Conv_2", biased=True,
                        bn=False, relu=False, training=training)
    return x


def ExtractCostFeatureBlock(x, training=True):
    with tf.variable_scope("ExtractCostFeatureBlock"):
        x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
        x = Conv2DLayer(x, 3, 1, 64, "Conv_2", biased=True,
                        bn=False, relu=False, training=training)
    return x


def BuildCostVolumeBlock(imgL, imgR, disp_num):
    with tf.variable_scope("BuildCostVolumeBlock"):
        batchsize, height, width, feature_num = imgL.get_shape().as_list()
        cost_vol = []
        for d in xrange(1, disp_num/4 + 1):
            paddings = [[0, 0], [0, 0], [d, 0], [0, 0]]
            slice_featuresR = tf.slice(imgR, [0, 0, 0, 0],
                                       [-1, height, width - d, feature_num])
            slice_featuresR = tf.pad(slice_featuresR, paddings, "CONSTANT")
            ave_feature = (imgL + slice_featuresR) / 2
            ave_feature2 = (tf.square(imgL) + tf.square(slice_featuresR)) / 2
            cost = ave_feature2 - tf.square(ave_feature)
            cost_vol.append(cost)

        cost_vol = tf.stack(cost_vol, axis=1)
    return cost_vol


def MatchingBlock(x, training=True):
    with tf.variable_scope("MatchingBlock"):
        x = Conv3DLayer(x, 3, 1, 64, "Conv_1", training=training)
        x = Conv3DLayer(x, 3, 1, 32, "Conv_2", training=training)
        x = Res3DBlock(x, 3, "Res_1", training=training)
    return x


def EncoderBlock(x, training=True):
    with tf.variable_scope("EncoderBlock"):
        level_list = []
        # 1/8
        x = DownSamplingBlock(x, 3, 32, "Level_1", training=training)
        level_list.append(x)

        # 1/16
        x = DownSamplingBlock(x, 3, 64, "Level_2", training=training)
        level_list.append(x)

        # 1/32
        x = DownSamplingBlock(x, 3, 96, "Level_3", training=training)
        level_list.append(x)

        # 1/64
        x = DownSamplingBlock(x, 3, 128, "Level_4", training=training)
    return x, level_list


def NonLocalGroupBlock(x, training=True):
    with tf.variable_scope("NonLocalGroupBlock"):
        x = SpaceTimeNonlocalBlock(x, "NonLocal_0", training=training)
        x = SpaceTimeNonlocalBlock(x, "NonLocal_1", training=training)
        x = SpaceTimeNonlocalBlock(x, "NonLocal_2", training=training)
        #x = SpaceTimeNonlocalBlock(x, "NonLocal_3", training=training)
        #x = SpaceTimeNonlocalBlock(x, "NonLocal_4", training=training)

    return x


def FeatureMatchingBlock(x, training=True):
    with tf.variable_scope("FeatureMatchingBlock"):
        x = MatchingBlock(x, training=training)
        shortcut = x
        x, level_list = EncoderBlock(x, training=training)
        x = NonLocalGroupBlock(x, training=training)
        x = DecoderBlock(x, level_list, training=training)
        x = shortcut + x

    return x


def DecoderBlock(x, level_list, training=True):
    with tf.variable_scope("DecoderBlock"):
        # 1/32
        x = UpSamplingBlock(x, 3, 96, "Level_3", training=training)
        x = x + level_list[2]

        # 1/16
        x = UpSamplingBlock(x, 3, 64, "Level_2", training=training)
        x = x + level_list[1]

        # 1/8
        x = UpSamplingBlock(x, 3, 32, "Level_1", training=training)
        x = x + level_list[0]

        # 1/4
        x = UpSamplingBlock(x, 3, 32, "Level_0", training=training)

    return x


def RecoverSizeBlock(x, training=True):
    with tf.variable_scope("RecoverSizeBlock"):
        # 1/2
        x = DeConv3DLayer(x, 3, 2, 8, "DeConv_1", training=training)
        x = DeConv3DLayer(x, 3, 2, 1, "DeConv_2", biased=True,
                          relu=False, bn=False, training=training,)
        x = Conv3DLayer(x, 3, 1, 1, "Conv_1", biased=True,
                        relu=False, bn=False, training=training)
    return x


def GetWeightBlock(batchsize, disp_num, height, width):
    disp = tf.range(0, disp_num, 1.0)
    disp = tf.cast(disp, tf.float32)
    w = tf.reshape(disp, [1, disp_num, 1, 1])
    w = tf.tile(w, [batchsize, 1, height, width])
    return w


def SoftArgMinBlock(x):
    with tf.variable_scope("SoftArgMin"):
        x = tf.squeeze(x, axis=4)
        batchsize, disp_num, height, width = x.get_shape().as_list()
        w = GetWeightBlock(batchsize, disp_num, height, width)
        x = tf.nn.softmax(x, axis=1)
        x = tf.multiply(x, w)
        x = tf.reduce_sum(x, axis=1)
    return x


def FeatureConcat(x, imgL, seg, training=True):
    with tf.variable_scope("FeatureConcat"):
        _, height, width = x.get_shape().as_list()
        x = tf.expand_dims(x, axis=3)
        #seg = tf.argmax(seg, axis=3)
        seg = tf.image.resize_images(seg, [height, width])
        x = Conv2DLayer(x, 1, 1, 32, "Conv_1", training=training)
        x = tf.concat([x, imgL, seg], axis=3)
    return x


def ResidualLearning(x, training=True):
    with tf.variable_scope("ResidualLearning"):
        x = Conv2DLayer(x, 3, 1, 32, "Conv_1", training=training)

        res_block_num = 3
        for i in range(res_block_num):
            x = Res2DBlock(x, 3, "Res_" + str(i), training=training)

        x = Conv2DLayer(x, 3, 1, 1, "Conv_2", biased=True,
                        relu=False, bn=False, training=training)
        x = tf.squeeze(x, axis=3)
    return x


def ExtractSegFeatureBlock(x, training=True):
    with tf.variable_scope("ExtractSegFeatureBlock"):
        x = Conv2DLayer(x, 3, 1, 256, "Conv_1", training=training)
        x = Conv2DLayer(x, 3, 1, 128, "Conv_2", training=training)
        x = Res2DBlock(x, 3, "Res_1", training=training)

        res_block_num = 16
        for i in range(res_block_num):
            x = Bottleneck2DBlock(x, "BottleNeck_" + str(i), training=training)
    return x


def FeatureFusionBlock(x, training=True):
    with tf.variable_scope("FeatureFusionBlock"):
        x = Conv2DLayer(x, 3, 1, 128, "Conv_1", training=training)
        x = Conv2DLayer(x, 3, 1, 64, "Conv_2", training=training)
    return x


def RecoveryCLSSizeBlock(x, cls_num, training=True):
    with tf.variable_scope("RecoveryCLSSizeBlock"):
        x = DeConv2DLayer(x, 3, 2, 32, "DeConv_1", training=training)
        x = DeConv2DLayer(x, 3, 2, cls_num, "DeConv_2", biased=True,
                          relu=False, bn=False, training=training)
        x = Conv2DLayer(x, 3, 1, cls_num, "Conv_3", biased=True,
                        relu=False, bn=False, training=training)
    return x


def ClsResidualLearning(x, cls_num, training=True):
    with tf.variable_scope("ResidualLearning"):
        x = Conv2DLayer(x, 3, 1, 32, "Conv_1", training=training)

        res_block_num = 3
        for i in range(res_block_num):
            x = Res2DBlock(x, 3, "Res_" + str(i), training=training)

        x = Conv2DLayer(x, 3, 1, cls_num, "Conv_2", biased=True,
                        relu=False, bn=False, training=training)
    return x


def ClsFeatureConcat(x, imgL, disp, training=True):
    with tf.variable_scope("ClsFeatureConcat"):
        disp = tf.expand_dims(disp, axis=3)
        disp = Conv2DLayer(disp, 1, 1, 32, "Conv_1", training=training)
        x = tf.concat([x, imgL, disp], axis=3)
    return x


def ExtractAttentionFeatureBlock(x, training=True):
    with tf.variable_scope("ExtractAttentionFeatureBlock"):
        x = Conv2DLayer(x, 3, 1, 256, "Conv_1", training=training)
        x = Conv2DLayer(x, 3, 1, 128, "Conv_2", training=training)
        x = Res2DBlock(x, 3, "Res_1", training=training)
    return x


def AttentionBlock(x, training=True):
    with tf.variable_scope("AttentionBlock"):
        block_num = 4
        for i in range(block_num):
            x = GCBlock(x, "GlobalContext_" + str(i), training=training)
            x = Res2DBlock(x, 3, "Res_" + str(i), training=training)
    return x


def SegAttentionBlock(x, training=True):
    with tf.variable_scope("SegAttentionBlock"):
        x = Conv2DLayer(x, 1, 1, 64, "Conv_1", training=training)
        x = Conv2DLayer(x, 1, 1, 1, "Conv_2", biased=False,
                        relu=False, bn=False, training=training)
        x = tf.nn.sigmoid(x)
    return x


def DspAttentionBlock(x, dsp_num, training=True):
    with tf.variable_scope("DspAttentionBlock"):
        x = Conv2DLayer(x, 1, 1, 64, "Conv_1", training=training)
        x = Conv2DLayer(x, 1, 1, dsp_num // 4, "Conv_2", biased=False,
                        relu=False, bn=False, training=training)
        x = tf.nn.sigmoid(x)
    return x


def SegAttentionAddBlock(x, w, training=True):
    with tf.variable_scope("SegAttentionAddBlock"):
        batchsize, height, width, shortcutFeatureNum = x.get_shape().as_list()
        w = tf.reshape(w, shape=(batchsize, height, width, 1))
        w = tf.tile(w, [1, 1, 1, shortcutFeatureNum])
        x = x + w * x
        x = Conv2DLayer(x, 3, 1, shortcutFeatureNum, "Conv_1", training=training)
    return x


def DspAttentionAddBlock(x, w, training=True):
    with tf.variable_scope("DspAttentionAddBlock"):
        batchsize, dsp_num, height, width, shortcutFeatureNum = x.get_shape().as_list()
        w = tf.transpose(w, [0, 3, 1, 2])
        w = tf.reshape(w, shape=(batchsize, dsp_num, height, width, 1))
        w = tf.tile(w, [1, 1, 1, 1, shortcutFeatureNum])
        x = x + w * x
        x = Conv3DLayer(x, 3, 1, shortcutFeatureNum, "Conv_1", training=training)
    return x
