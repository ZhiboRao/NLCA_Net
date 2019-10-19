# -*- coding: utf-8 -*-
from Basic.Define import *
from Basic.LogHandler import *
from Basic.Input import *
from Basic.Output import *
from Evaluation.Accuracy import *
from Evaluation.Loss import *
from Evaluation.GradientsAnalysis import *
from Model.Model import MSGFNet
import random


class TrainModel(object):
    # init Model
    def __init__(self, args):
        # save the args and file path
        self.args = args
        self.checkpoint_path = os.path.join(self.args.modelDir, MODEL_NAME)
        self.fd_train_acc, self.fd_train_loss, self.fd_val_acc, self.fd_val_loss = CreateResultFile(
            args)

        # Build Graph
        #
        # The input data
        self.imgL = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, args.corpedImgHeight, args.corpedImgWidth, 3))
        self.imgR = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, args.corpedImgHeight, args.corpedImgWidth, 3))
        self.imgGround = tf.placeholder(tf.float32, shape=(
            args.batchSize * args.gpu, args.corpedImgHeight, args.corpedImgWidth))

        self.train_step, self.res, self.loss, self.acc = self.BuildNet(
            self.args, self.imgL, self.imgR, self.imgGround)

        # set the sess
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     gpu_options=tf.GPUOptions(allow_growth=True)))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
        writer = tf.summary.FileWriter(self.args.log, self.sess.graph)
        writer.close()

        Info("Finish training init work")

    def BuildNet(self, args, imgL, imgR, imgGround):
        with tf.device('/cpu:0'):
            global_step = tf.get_variable(
                'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            # Optimizer
            # opt = tf.train.RMSPropOptimizer(learning_rate=self.args.learningRate)
            # opt = tf.train.MomentumOptimizer(learning_rate=self.args.learningRate, momentum=0.9)
            opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate)
            # opt = tf.train.GradientDescentOptimizer(learning_rate=self.args.learningRate)

            Info("The learning rate: %f" % self.args.learningRate)
            tower_grads = []
            tower_coarse_map = []
            tower_refine_map = []
            ave_loss = 0
            ave_loss_0 = 0
            ave_loss_1 = 0
            ave_coarse_acc = 0
            ave_refine_acc = 0

            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(self.args.gpu):
                    Info("Begin init the gpus %d" % i)
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ('TOWER', i)):
                            # divid the data
                            start = self.args.batchSize * i
                            end = start + self.args.batchSize

                            # get the result
                            coarse_map, refine_map = MSGFNet().NetWork(
                                imgL[start:end], imgR[start:end],
                                args.corpedImgHeight, args.corpedImgWidth, True)

                            tower_coarse_map.append(coarse_map)
                            tower_refine_map.append(refine_map)

                            # get the acc and loss, 3 piexl error, acc[1] means 3 piexl
                            acc = MatchingAcc(coarse_map, imgGround[start:end])
                            ave_coarse_acc = ave_coarse_acc + acc[1]
                            acc = MatchingAcc(refine_map, imgGround[start:end])
                            ave_refine_acc = ave_refine_acc + acc[1]

                            # loss
                            loss_0 = MAE_Loss(coarse_map, imgGround[start:end])
                            loss_1 = MAE_Loss(refine_map, imgGround[start:end])

                            loss = loss_0 + loss_1

                            ave_loss = ave_loss + loss
                            ave_loss_0 = ave_loss_0 + loss_0
                            ave_loss_1 = ave_loss_1 + loss_1

                            # next model use the same para
                            tf.get_variable_scope().reuse_variables()

                            # get the grad
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)

                            Info("Finished init the gpus %d" % i)
                            # gradients, m = GradientsMAS(loss, self.args.learningRate)
            # get the loss and acc
            ave_loss = ave_loss / self.args.gpu
            ave_loss_0 = ave_loss_0 / self.args.gpu
            ave_loss_1 = ave_loss_1 / self.args.gpu
            ave_coarse_acc = ave_coarse_acc / self.args.gpu
            ave_refine_acc = ave_refine_acc / self.args.gpu

            # get the ave grads
            grads = AverageGradients(tower_grads)
            train_step = opt.apply_gradients(grads, global_step=global_step)

            tower_loss = []
            tower_loss.append(ave_loss)
            tower_loss.append(ave_loss_0)
            tower_loss.append(ave_loss_1)

            # get the res
            res = []
            res.append(tower_coarse_map)
            res.append(tower_refine_map)

            # get the acc
            acc = []
            acc.append(ave_coarse_acc)
            acc.append(ave_refine_acc)

        return train_step, res, tower_loss, acc

    def RestoreModel(self):
        ckpt = tf.train.get_checkpoint_state(self.args.modelDir)
        if ckpt and ckpt.model_checkpoint_path:

            variables = tf.global_variables()
            variables_to_resotre = variables
            # variables_to_resotre = [v for v in variables
            #                        if ('ExtractSegFeatureBlock/BottleNeck_' not in v.name
            #                            and 'AttentionModule' not in v.name
            #                            and 'SegAttentionAddBlock' not in v.name
            #                            and 'DspAttentionAddBlock' not in v.name
            #                            and 'ExtractSegFeatureBlock/BottleNeck_' not in v.name
            #                            and 'SegModule/FeatureFusionBlock' not in v.name
            #                            and 'SegModule/RecoveryCLSSizeBlock/DeConv_2' not in v.name
            #                            and 'SegModule/RecoveryCLSSizeBlock/Conv_3' not in v.name)]

            # and
            # 'ASPP' not in v.name and
            # 'pool1_3x3_s2' not in v.name and
            # 'Adam' not in v.name and
            # 'conv2_1' not in v.name and
            # 'beta1_power' not in v.name and
            # 'beta2_power' not in v.name)]
            tf.train.Saver(var_list=variables_to_resotre).restore(
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

    def TrainProcess(self, args, epoch, dataloader, num_tr_batch):
        if num_tr_batch == 0:
            return

        # init var
        tr_ave_coarse_acc = 0
        tr_ave_refine_acc = 0
        tr_ave_loss = 0
        tr_ave_coarse_disp_loss = 0
        tr_ave_refine_disp_loss = 0

        # start the training process
        start_time = time.time()
        for step in xrange(num_tr_batch):
            # imgLs, imgRs, imgGrounds = GetBatchImage(args, randomTrainingList, step)
            imgLs, imgRs, dispImgGrounds = dataloader.GetTrainData()

            _, loss, acc = self.sess.run([self.train_step, self.loss, self.acc],
                                         feed_dict={self.imgL: imgLs,
                                                    self.imgR: imgRs,
                                                    self.imgGround: dispImgGrounds})

            tr_ave_loss = tr_ave_loss + loss[0]
            tr_ave_coarse_disp_loss = tr_ave_coarse_disp_loss + loss[1]
            tr_ave_refine_disp_loss = tr_ave_refine_disp_loss + loss[2]

            tr_ave_coarse_acc = tr_ave_coarse_acc + acc[0]
            tr_ave_refine_acc = tr_ave_refine_acc + acc[1]

        # stop the training process, and compute the ave loss and acc
        duration = time.time() - start_time
        tr_ave_loss = tr_ave_loss / num_tr_batch
        tr_ave_coarse_disp_loss = tr_ave_coarse_disp_loss / num_tr_batch
        tr_ave_refine_disp_loss = tr_ave_refine_disp_loss / num_tr_batch

        tr_ave_coarse_acc = tr_ave_coarse_acc / num_tr_batch
        tr_ave_refine_acc = tr_ave_refine_acc / num_tr_batch

        format_str = ('[TrainProcess] epochs = %d ,loss = %.6f, ' +
                      'coarse_disp_loss = %.6f, refine_disp_loss = %.6f, ' +
                      'coarse_acc = %.6f, refine_acc = %.6f (%.3f sec/batch)')
        Info(format_str % (epoch, tr_ave_loss, tr_ave_coarse_disp_loss, tr_ave_refine_disp_loss,
                           tr_ave_coarse_acc, tr_ave_refine_acc, duration))
        OutputData(self.fd_train_acc, loss)
        OutputData(self.fd_train_loss, tr_ave_refine_acc)

    def ValProcess(self, args, epoch, dataloader, num_val_batch):
        if num_val_batch == 0:
            return

        val_ave_coarse_acc = 0
        val_ave_refine_acc = 0
        val_ave_loss = 0
        val_ave_coarse_disp_loss = 0
        val_ave_refine_disp_loss = 0

        start_time = time.time()

        for step in range(num_val_batch):
            # imgLs, imgRs, imgGrounds = GetBatchImage(args, randomValList, step, True)
            imgLs, imgRs, dispImgGrounds = dataloader.GetValData()

            loss, acc = self.sess.run(
                [self.loss, self.acc],
                feed_dict={self.imgL: imgLs,
                           self.imgR: imgRs,
                           self.imgGround: dispImgGrounds})
            val_ave_loss = val_ave_loss + loss[0]
            val_ave_coarse_disp_loss = val_ave_coarse_disp_loss + loss[1]
            val_ave_refine_disp_loss = val_ave_refine_disp_loss + loss[2]

            val_ave_coarse_acc = val_ave_coarse_acc + acc[0]
            val_ave_refine_acc = val_ave_refine_acc + acc[1]
            '''
            res = np.array(res)
            for i in range(args.gpu):
                for j in range(args.batchSize):
                    temRes = res[1, i, j, :, :]
                    # temRes = temRes[top_pads[i*args.batchSize+j]:, :-left_pads[i*args.batchSize+j]]
                    # SaveTestData(args, temRes, args.gpu*args.batchSize*step + i*args.batchSize + j)
                    SaveDFCTestImg(args, temRes, str(
                        args.gpu * args.batchSize*step + i*args.batchSize + j))
            '''
        duration = time.time() - start_time

        val_ave_loss = val_ave_loss / num_val_batch
        val_ave_coarse_cls_loss = val_ave_coarse_cls_loss / num_val_batch
        val_ave_refine_cls_loss = val_ave_refine_cls_loss / num_val_batch
        val_ave_coarse_disp_loss = val_ave_coarse_disp_loss / num_val_batch
        val_ave_refine_disp_loss = val_ave_refine_disp_loss / num_val_batch

        val_ave_coarse_acc = val_ave_coarse_acc / num_val_batch
        val_ave_refine_acc = val_ave_refine_acc / num_val_batch
        val_ave_coarse_pixel_acc = self.acc[2].eval(session=self.sess)
        val_ave_coarse_mIoU_acc = self.acc[3].eval(session=self.sess)
        val_ave_refine_pixel_acc = self.acc[4].eval(session=self.sess)
        val_ave_refine_mIoU_acc = self.acc[5].eval(session=self.sess)

        format_str = ('[ValProcess] epochs = %d ,loss = %.6f, ' +
                      'coarse_disp_loss = %.6f, refine_disp_loss = %.6f, ' +
                      'coarse_acc = %.6f, refine_acc = %.6f (%.3f sec/batch)')
        Info(format_str % (epoch, val_ave_loss, val_ave_coarse_disp_loss, val_ave_refine_disp_loss,
                           val_ave_coarse_acc, val_ave_refine_acc, duration))
        OutputData(self.fd_val_acc, val_ave_refine_acc)
        OutputData(self.fd_val_loss, val_ave_loss)

    def Train(self):
        Info("Start train work")
        # init the para
        args = self.args

        # The total parameter:
        self.Count()
        if not args.pretrain:                           # restore model
            self.RestoreModel()

        randomTrainingList = range(args.imgNum)
        num_tr_batch = args.imgNum / args.batchSize / args.gpu

        randomValList = range(args.valImgNum)
        num_val_batch = args.valImgNum / args.batchSize / args.gpu

        dataloader = Dataloader(args)

        for epoch in xrange(args.maxEpochs):

            # training process of each epoch
            self.TrainProcess(args, epoch, dataloader, num_tr_batch)

            # every VAL_TIMES to do val test
            if (epoch+1) % VAL_TIMES == 0:
                self.ValProcess(args, epoch, dataloader, num_val_batch)

            # save the model data
            if (epoch + 1) % args.auto_save_num == 0:
                self.saver.save(self.sess, self.checkpoint_path, global_step=epoch)
                Info('The model has been created')
