# -*- coding: utf-8 -*-
from Basic.Define import *
from BasicModule import *
from Basic.LogHandler import *
from JackBasicStructLib.Model.Template.ModelTemplate import ModelTemplate
from Evaluation.Accuracy import *
from Evaluation.Loss import *
import math


class NLCANet(ModelTemplate):
    def __init__(self, args, training=True):
        self.__args = args
        self.input_imgL_id = 0
        self.input_imgR_id = 1
        self.label_disp_id = 0
        self.output_coarse_img_id = 0
        self.output_refine_img_id = 1

        if training == True:
            self.height = args.corpedImgHeight
            self.width = args.corpedImgWidth
        else:
            self.height = args.padedImgHeight
            self.width = args.padedImgWidth

    def GenInputInterface(self):
        input = []

        args = self.__args
        imgL = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 3))
        input.append(imgL)

        imgR = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width, 3))
        input.append(imgR)

        return input

    def GenLabelInterface(self):
        label = []
        args = self.__args

        imgGround = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, self.height, self.width))
        label.append(imgGround)

        return label

    def Optimizer(self, lr):
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        return opt

    def Accuary(self, output, label):
        acc = []

        coarse_acc = MatchingAcc(output[self.output_coarse_img_id], label[self.label_disp_id])
        refine_acc = MatchingAcc(output[self.output_refine_img_id], label[self.label_disp_id])
        acc.append(coarse_acc[1])
        acc.append(refine_acc[1])

        return acc

    def Loss(self, output, label):
        loss = []
        loss_0 = MAE_Loss(output[self.output_coarse_img_id], label[self.label_disp_id])
        loss_1 = MAE_Loss(output[self.output_refine_img_id], label[self.label_disp_id])
        total_loss = loss_0 + loss_1
        loss.append(total_loss)
        loss.append(loss_0)
        loss.append(loss_1)
        return loss

    # This is the Inference, and you must have it!
    def Inference(self, input, training=True):
        imgL, imgR = self.__GetVar(input)
        coarse_map, refine_map = self.__NetWork(imgL, imgR, self.height, self.width, training)
        output = self.__GenRes(coarse_map, refine_map)
        return output

    def __NetWork(self, imgL, imgR, height, width, training=True):
        with tf.variable_scope("NLCANet"):
            Info('├── Begin Build ExtractUnaryFeature')
            with tf.variable_scope("ExtractUnaryFeature") as scope:
                imgL_feature = ExtractUnaryFeatureModule(imgL, training=training)
                scope.reuse_variables()
                imgR_feature = ExtractUnaryFeatureModule(imgR, training=training)
            Info('│   └── After ExtractUnaryFeature:' + str(imgL_feature.get_shape()))

            Info('├── Begin Build Cost Volume')
            cost_vol = BuildCostVolumeModule(imgL_feature, imgR_feature,
                                             IMG_DISPARITY, training=training)
            Info('│   └── After Cost Volume:' + str(cost_vol.get_shape()))

            Info('├── Begin Build 3DMatching')
            coarse_map = MatchingModule(cost_vol, training=training)
            Info('│   └── After 3DMatching:' + str(coarse_map.get_shape()))

            Info('└── Begin Build DispMapRefine')
            refine_map = DispRefinementModule(coarse_map, imgL,
                                              imgL_feature, training=training)
            Info('    └── After DispMapRefine:' + str(refine_map.get_shape()))

        return coarse_map, refine_map

    def __GetVar(self, input):
        return input[self.input_imgL_id], input[self.input_imgR_id]

    def __GenRes(self, coarse_map, refine_map):
        res = []
        res.append(coarse_map)
        res.append(refine_map)
        return res
