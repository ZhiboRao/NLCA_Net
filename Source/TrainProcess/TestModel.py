# -*- coding: utf-8 -*-
from Basic.Define import *
from Basic.LogHandler import *
from Basic.Input import *
from Basic.Output import *
from Evaluation.Accuracy import *
from Evaluation.Loss import *
from Evaluation.GradientsAnalysis import *
from Model.Interface import StereoMatchingNetWorks as smn
import random


class TestModel(object):
    # init Model
    def __init__(self, args):
        # save the args and file path
        self.args = args
        self.checkpoint_path = os.path.join(self.args.modelDir, MODEL_NAME)
        self.fd_test_acc = CreateResultFile(args)

        # Build Graph
        #
        # The input data
        self.imgL = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, args.padedImgHeight, args.padedImgWidth, 3))
        self.imgR = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, args.padedImgHeight, args.padedImgWidth, 3))

        self.tower_coarse_map, self.tower_refine_map = self.BuildNet(
            self.args, self.imgL, self.imgR)

        # set the sess
        self.sess = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        writer = tf.summary.FileWriter(self.args.log, self.sess.graph)
        writer.close()
        Info("Finish testing init work")

    def BuildNet(self, args, imgL, imgR):
        with tf.device('/cpu:0'):
            tower_coarse_map = []
            tower_refine_map = []

            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(self.args.gpu):
                    with tf.device('/gpu:%d' % i):
                        Info("Begin init the gpus %d" % i)
                        with tf.name_scope('%s_%d' % ('TOWER', i)):
                            # divid the data
                            start = self.args.batchSize * i
                            end = start + self.args.batchSize
                            # get the result

                            # coarse_map, refine_map = MSGFNet().NetWork(
                            #    imgL[start:end], imgR[start:end],
                            #    args.padedImgHeight,
                            #    args.padedImgWidth, True)  # get the result
                            x = []
                            x.append(imgL[start:end])
                            x.append(imgR[start:end])
                            x.append(args.padedImgHeight)
                            x.append(args.padedImgWidth)
                            res = smn().Inference('NLCANet', x, True)

                            coarse_map = res[0]
                            refine_map = res[1]

                            #coarse_map = coarse_map[:, :, :, 0:LABLE_NUM-1]
                            #coarse_map = tf.nn.softmax(coarse_map, axis=3)
                            #coarse_map = tf.argmax(coarse_map, 3)

                            tower_coarse_map.append(coarse_map)
                            tower_refine_map.append(refine_map)

                            # next model use the same para
                            tf.get_variable_scope().reuse_variables()
                            Info("Finished init the gpus %d" % i)

        return tower_coarse_map, tower_refine_map

    def RestoreModel(self):
        ckpt = tf.train.get_checkpoint_state(self.args.modelDir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver(var_list=tf.global_variables()).restore(
                self.sess, ckpt.model_checkpoint_path)
            Info("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            Info('No checkpoint file found.')

    @classmethod
    def Count(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim.value

            total_parameters += variable_parameters

        info = 'The total parameter: %d' % total_parameters
        Info(info)

    def TestProcess(self, args, randomTestList, num_test_batch):
        if num_test_batch == 0:
            return

        start_time = time.time()
        for step in range(num_test_batch):
            imgLs, imgRs, top_pads, left_pads = GetBatchTestImage(args, randomTestList, step, False)
            res = self.sess.run(
                self.tower_refine_map,
                feed_dict={self.imgL: imgLs,
                           self.imgR: imgRs})

            res = np.array(res)
            for i in range(args.gpu):
                for j in range(args.batchSize):
                    temRes = res[i, j, :, :]
                    temRes = temRes[top_pads[i*args.batchSize+j]:, :-left_pads[i*args.batchSize+j]]
                    SaveTestData(args, temRes, args.gpu*args.batchSize*step + i*args.batchSize + j)
                    Info('[TestProcess] Finish ' +
                         str(args.gpu * args.batchSize*step + i*args.batchSize + j) + ' image.')

        duration = time.time() - start_time
        format_str = ('[TestProcess] Finish Test (%.3f sec/batch)')
        Info(format_str % (duration))

    def Test(self):
        Info("Start Test work")
        # init the para
        args = self.args

        # The total parameter:
        self.Count()
        if not args.pretrain:                           # restore model
            self.RestoreModel()

        randomTestList = range(args.imgNum)
        num_test_batch = args.imgNum / args.batchSize / args.gpu

        # every VAL_TIMES to do val test
        self.TestProcess(args, randomTestList, num_test_batch)
