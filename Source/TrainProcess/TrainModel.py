# -*- coding: utf-8 -*-
import random

from Basic.Define import *
from Basic.LogHandler import *
from Basic.Input import *
from Basic.Output import *
from Evaluation.Accuracy import *
from Evaluation.Loss import *
from Evaluation.GradientsAnalysis import *
from JackBasicStructLib.Proc.BuildGraph import *
from JackBasicStructLib.Evaluation.Algorithm import *
from Model.Interface import StereoMatchingNetWorks as smn


class TrainModel(object):
    # init Model
    def __init__(self, args):
        # save the args and file path
        self.args = args
        self.fd_train_acc, self.fd_train_loss, self.fd_val_acc, self.fd_val_loss = CreateResultFile(
            args)

        self.checkpoint_path = os.path.join(self.args.modelDir, MODEL_NAME)

        paras = Paras(args.learningRate, args.batchSize, args.gpu,
                      args.log, args.modelDir, self.checkpoint_path, max_save_num=10)
        model = smn().Inference("NLCANet", args, True)
        self.graph = BuildGraph(paras, model, True)

        Info("Finish training init work")

    def TrainProcess(self, args, epoch, dataloader, num_tr_batch):
        if num_tr_batch == 0:
            return

        # init var
        tr_loss = []
        tr_acc = []

        # start the training process
        start_time = time.time()
        for step in xrange(num_tr_batch):
            # imgLs, imgRs, imgGrounds = GetBatchImage(args, randomTrainingList, step)
            imgLs, imgRs, dispImgGrounds = dataloader.GetTrainData()
            input = []
            label = []
            input.append(imgLs)
            input.append(imgRs)
            label.append(dispImgGrounds)

            _, loss, acc = self.graph.TrainRun(input, label, True)

            tr_loss.append(loss)
            tr_acc.append(acc)

        # stop the training process, and compute the ave loss and acc
        duration = time.time() - start_time
        tr_loss = NumpyListMean(tr_loss)
        tr_acc = NumpyListMean(tr_acc)

        format_str = ('[TrainProcess] epochs = %d ,loss = %.6f, ' +
                      'coarse_disp_loss = %.6f, refine_disp_loss = %.6f, ' +
                      'coarse_acc = %.6f, refine_acc = %.6f (%.3f sec/batch)')
        Info(format_str % (epoch, tr_loss[0], tr_loss[1], tr_loss[2],
                           tr_acc[0], tr_acc[1], duration))
        OutputData(self.fd_train_acc, tr_loss[0])
        OutputData(self.fd_train_loss, tr_acc[1])

    def ValProcess(self, args, epoch, dataloader, num_val_batch):
        if num_val_batch == 0:
            return

        val_loss = []
        val_acc = []
        start_time = time.time()

        for step in range(num_val_batch):
            # imgLs, imgRs, imgGrounds = GetBatchImage(args, randomValList, step, True)
            imgLs, imgRs, dispImgGrounds = dataloader.GetValData()

            input = []
            label = []
            input.append(imgLs)
            input.append(imgRs)
            label.append(dispImgGrounds)

            _, loss, acc = self.graph.ValRun(input, label, True)

            val_loss.append(loss)
            val_acc.append(acc)

        # stop the training process, and compute the ave loss and acc
        duration = time.time() - start_time
        val_loss = NumpyListMean(val_loss)
        val_acc = NumpyListMean(val_acc)

        format_str = ('[ValProcess] epochs = %d ,loss = %.6f, ' +
                      'coarse_disp_loss = %.6f, refine_disp_loss = %.6f, ' +
                      'coarse_acc = %.6f, refine_acc = %.6f (%.3f sec/batch)')
        Info(format_str % (epoch, val_loss[0], val_loss[1], val_loss[2],
                           val_acc[0], val_acc[1], duration))
        OutputData(self.fd_val_acc, val_loss[0])
        OutputData(self.fd_val_loss, val_acc[1])

    def Train(self):
        Info("Start train work")
        # init the para
        args = self.args

        # The total parameter:
        self.graph.Count()
        if not args.pretrain:                           # restore model
            self.graph.RestoreModel()

        num_tr_batch = args.imgNum / args.batchSize / args.gpu
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
                self.graph.SaveModel(epoch)
                Info('The model has been created')
