# -*- coding: utf-8 -*-
import random
from Basic.Define import *
from Basic.LogHandler import *
from Basic.Input import *
from Basic.Output import *
from Evaluation.Accuracy import *
from Evaluation.Loss import *
from Evaluation.GradientsAnalysis import *
from Model.Interface import StereoMatchingNetWorks as smn
from JackBasicStructLib.Proc.BuildGraph import *
from JackBasicStructLib.Evaluation.Algorithm import *


class TestModel(object):
    # init Model
    def __init__(self, args):
        # save the args and file path
        self.args = args
        self.checkpoint_path = os.path.join(self.args.modelDir, MODEL_NAME)
        self.fd_test_acc = CreateResultFile(args)

        # set para and model, then build the graph
        #
        paras = Paras(args.learningRate, args.batchSize, args.gpu,
                      args.log, args.modelDir, self.checkpoint_path, max_save_num=10)
        model = smn().Inference("NLCANet", args, False)
        self.graph = BuildGraph(paras, model, False)

        Info("Finish testing init work")

    def TestProcess(self, args, randomTestList, num_test_batch):
        if num_test_batch == 0:
            return

        start_time = time.time()
        for step in range(num_test_batch):
            imgLs, imgRs, top_pads, left_pads = GetBatchTestImage(args, randomTestList, step, False)
            input = []
            label = []
            input.append(imgLs)
            input.append(imgRs)
            label.append(None)

            res = self.graph.TestRun(input, label, False)

            res = np.array(res)
            # print res.shape
            for i in range(args.gpu):
                for j in range(args.batchSize):
                    temRes = res[i, 1, j, :, :]
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
        self.graph.Count()
        if not args.pretrain:                           # restore model
            self.graph.RestoreModel()

        randomTestList = range(args.imgNum)
        num_test_batch = args.imgNum / args.batchSize / args.gpu

        # every VAL_TIMES to do val test
        self.TestProcess(args, randomTestList, num_test_batch)
