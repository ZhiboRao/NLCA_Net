# -*- coding: utf-8 -*-
from JackBasicStructLib.Basic.Define import *
from JackBasicStructLib.Evaluation.Algorithm import *
from JackBasicStructLib.Basic.Processbar import *
from BuildGraph import *
from LoadWoker import *

from ModelImplementation.NetWorkInference import NetWorkInference as ni


class Executor(object):
    """docstring for Executor"""

    def __init__(self, args, is_training=True):
        super(Executor, self).__init__()

        self.__paras, self.__model, self.__dataloader = ni().Inference(args, is_training)
        self.__loadWoker = LoadWoker(self.__paras, self.__dataloader)
        self.__graph = BuildGraph(self.__paras, self.__model, is_training)

    def Train(self):
        Info("Start training work")
        self.__graph.Count()

        # restore model
        if not self.__paras.pretrain:
            self.__graph.RestoreModel()

        num_tr_batch, num_val_batch = self.__loadWoker.GetMaxBatch()

        for epoch in xrange(self.__paras.maxEpoch):
            self.__TrainProc(self.__graph.TrainRun, self.__loadWoker.GetTrainData,
                             num_tr_batch, self.__dataloader.ShowTrainingResult,
                             epoch, "Train")
            self.__TrainProc(self.__graph.ValRun, self.__loadWoker.GetValData,
                             num_val_batch, self.__dataloader.ShowValResult,
                             epoch, "Val")

            if (epoch + 1) % self.__paras.save_epoch == 0:
                self.__graph.SaveModel(epoch)

        Info("Finish training work")

    def Test(self):
        Info("Start testing work")
        self.__graph.Count()

        # restore model
        if not self.__paras.pretrain:
            self.__graph.RestoreModel()

        self.__TestProc()
        Info("Finish testing work")

    def __TestProc(self):
        num_tr_batch, _ = self.__loadWoker.GetMaxBatch()

        if num_tr_batch == 0:
            return

        start_time = time.time()
        process_bar = ShowProcess(num_tr_batch, 'Test')

        for step in range(num_tr_batch):
            for testID in range(self.__paras.test_times):
                input, supplement = self.__loadWoker.GetTrainData()
                output = self.__graph.TestRun(input, supplement, False)
                self.__dataloader.SaveResult(output, supplement, step, testID)
            duration = (time.time() - start_time) / (step + 1)
            duration = (num_tr_batch - step - 1) * duration
            process_bar.show_process(restTime=duration)

        duration = time.time() - start_time
        format_str = ('[TestProcess] Finish Test (%.3f sec/batch)')
        Info(format_str % (duration))

    def __TrainProc(self, execFunc, dataloader, num_batch, showFunc, epoch, info="Train"):
        if num_batch == 0:
            return

        tr_loss = []
        tr_acc = []
        process_bar = ShowProcess(num_batch, info)
        start_time = time.time()

        for step in xrange(num_batch):
            #tem_start_time = time.time()
            input, label = dataloader()
            _, loss, acc = execFunc(input, label, True)
            tr_loss.append(loss)
            tr_acc.append(acc)
            tem_loss = NumpyListMean(tr_loss)
            tem_acc = NumpyListMean(tr_acc)
            info_str = self.__dataloader.ShowIntermediateResult(epoch, tem_loss, tem_acc)
            duration = (time.time() - start_time) / (step + 1)
            duration = (num_batch - step - 1) * duration
            process_bar.show_process(showInfo=info_str, restTime=duration)

        duration = time.time() - start_time
        tr_loss = NumpyListMean(tr_loss)
        tr_acc = NumpyListMean(tr_acc)
        showFunc(epoch, tr_loss, tr_acc, duration)

        return tr_loss, tr_acc, duration
