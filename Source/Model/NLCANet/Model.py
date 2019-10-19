# -*- coding: utf-8 -*-
from Basic.Define import *
from BasicModule import *
from Basic.LogHandler import *
import math


class NLCANet(object):
    def __init__(self):
        pass

    def GetVar(self, args):
        return args[0], args[1], args[2], args[3]

    def GenRes(self, coarse_map, refine_map):
        res = []
        res.append(coarse_map)
        res.append(refine_map)
        return res

    # This is the Inference, and you must have it!
    def Inference(self, args, training=True):
        imgL, imgR, height, width = self.GetVar(args)
        coarse_map, refine_map = self.NetWork(imgL, imgR, height, width, training)
        res = self.GenRes(coarse_map, refine_map)
        return res

    def NetWork(self, imgL, imgR, height, width, training=True):
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
