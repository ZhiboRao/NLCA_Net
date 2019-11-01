# -*- coding: utf-8 -*-
from JackBasicStructLib.Basic.Define import *
from JackBasicStructLib.Evaluation.Gradients import *
from JackBasicStructLib.Evaluation.Algorithm import *
from JackBasicStructLib.Basic.Paras import *


class BuildGraph(object):
    """docstring for BuildGraph"""

    def __init__(self, paras, model, training=True):
        super(BuildGraph, self).__init__()
        Info("Beigin build network graph")
        self._paras = paras
        self._model = model
        self._input, self._label = self.__GenProcInterface(self._model)
        self._train_step, self._output, self._loss, self._acc = self.__BuildComputeringGraph(
            self._input, self._label, self._model, self._paras, training)
        self._sess, self._saver = self.__Postprocess(self._paras)
        Info("Finished build network graph")

    def SetParas(self, paras):
        self._paras = paras

    def SetModel(self, model):
        self._model = model

    def SaveModel(self, epoch):
        self._saver.save(self._sess, self._paras.save_path, global_step=epoch)
        Info('The model has been created')

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

    def GlobalVariables(self):
        return tf.global_variables()

    def RestoreModel(self):
        variables = self.GlobalVariables()
        # variables_to_resotre = [v for v in variables
        #                        if ('BN' not in v.name)]
        self.RestorePartModel(variables)

    def RestorePartModel(self, variables_to_resotre):
        ckpt = tf.train.get_checkpoint_state(self._paras.save_dir)
        if ckpt and ckpt.model_checkpoint_path:

            tf.train.Saver(var_list=variables_to_resotre).restore(
                self._sess, ckpt.model_checkpoint_path)
            Info("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
        else:
            Info('No checkpoint file found.')

        self._sess.graph.finalize()

    def TrainRun(self, input, label, training=True):
        feed_dict = self.__GenInterfaceDict(input, label, training)

        _, output, loss, acc = self._sess.run([self._train_step,
                                               self._output,
                                               self._loss,
                                               self._acc],
                                              feed_dict=feed_dict)
        return output, loss, acc

    def ValRun(self, input, label, training=True):
        feed_dict = self.__GenInterfaceDict(input, label, training)
        output, loss, acc = self._sess.run([self._output,
                                            self._loss,
                                            self._acc],
                                           feed_dict=feed_dict)
        return output, loss, acc

    def TestRun(self, input, label, training=False):
        feed_dict = self.__GenInterfaceDict(input, label, training)
        output = self._sess.run(self._output,
                                feed_dict=feed_dict)
        return output

    def __GenInterfaceDict(self, input, label, training=True):

        res = {}
        if training == True:
            input_dict = self.__List2Dict(self._input, input)
            label_dict = self.__List2Dict(self._label, label)
            res = dict(input_dict.items() + label_dict.items())
        else:
            res = self.__List2Dict(self._input, input)

        return res

    def __List2Dict(self, key, value):
        assert len(key) == len(value)
        res = {}
        for i in range(len(key)):
            res[key[i]] = value[i]

        return res

    def __GenProcInterface(self, model, training=True):
        input = self.__GenInputInterface(self._model)

        if training == True:
            label = self.__GenLabelInterface(self._model)
        else:
            label = None

        return input, label

    def __GenInputInterface(self, model):
        return model.GenInputInterface()

    def __GenLabelInterface(self, model):
        return model.GenLabelInterface()

    def __SeparateBatch(self, data, start, end):
        res = []
        for i in range(len(data)):
            tem_data = data[i]
            res.append(tem_data[start:end])
        return res

    def __SeparateData(self, input, label, paras, num, training=True):
        start = paras.batchsize * num
        end = start + paras.batchsize

        tem_input = self.__SeparateBatch(input, start, end)

        if training == True:
            tem_label = self.__SeparateBatch(label, start, end)
        else:
            tem_label = None

        return tem_input, tem_label

    def __GenTrainingData(self, model, output, label, opt, training=True):
        if training == True:
            acc = model.Accuary(output, label)
            loss = model.Loss(output, label)
            grads = opt.compute_gradients(loss[0])
        else:
            acc = None
            loss = None
            grads = None
        return grads, loss, acc

    def __InitGPUs(self, input, label, model, paras, opt, num, training=True):
        with tf.device('/gpu:%d' % num):
            with tf.name_scope('%s_%d' % ('TOWER', num)):
                Info("Begin init the gpus %d" % num)

                tem_input, tem_label = self.__SeparateData(input, label, paras, num, training)
                output = model.Inference(tem_input, True)
                grads, loss, acc = self.__GenTrainingData(model, output, tem_label, opt, training)

                tf.get_variable_scope().reuse_variables()

                Info("Finished init the gpus %d" % num)
        return output, grads, loss, acc

    def __Optimizer(self, opt, tower_grads, global_step):
        grads = AverageGradients(tower_grads)
        train_step = opt.apply_gradients(grads, global_step=global_step)
        return train_step

    def __CalculateEvaluationStandard(self, tower_loss, tower_acc):
        tower_loss = ListMean(tower_loss)
        tower_acc = ListMean(tower_acc)
        return tower_loss, tower_acc

    def __InitOptimizer(self, model, paras, training=True):
        if training == True:
            Info("The learning rate: %f" % paras.lr)
            opt = model.Optimizer(paras.lr)
        else:
            opt = None

        return opt

    def __ApplyOptimizer(self, opt, tower_grads, tower_loss, tower_acc, global_step, training=True):
        if training == True:
            train_step = self.__Optimizer(opt, tower_grads, global_step)
            tower_loss, tower_acc = self.__CalculateEvaluationStandard(tower_loss, tower_acc)
            #tower_loss = tower_loss[0]
            #tower_acc = tower_acc[0]
        else:
            train_step = None
            tower_loss = None
            tower_acc = None

        return train_step, tower_loss, tower_acc

    def __GenResObj(self):
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        tower_output = []
        tower_grads = []
        tower_loss = []
        tower_acc = []
        return global_step, tower_output, tower_grads, tower_loss, tower_acc

    def __BuildComputeringGraph(self, input, label, model, paras, training=True):
        with tf.device('/cpu:0'):
            global_step, tower_output, tower_grads, tower_loss, tower_acc = self.__GenResObj()
            opt = self.__InitOptimizer(model, paras, training)

            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(paras.gpu):
                    output, grads, loss, acc = self.__InitGPUs(
                        input, label, model, paras, opt, i, training)
                    tower_output.append(output)
                    tower_grads.append(grads)
                    tower_loss.append(loss)
                    tower_acc.append(acc)

            train_step, tower_loss, tower_acc = self.__ApplyOptimizer(
                opt, tower_grads, tower_loss, tower_acc, global_step, training)

        return train_step, tower_output, tower_loss, tower_acc

    def __Postprocess(self, paras):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=paras.max_save_num)

        writer = tf.summary.FileWriter(paras.log_path, sess.graph)
        writer.close()

        return sess, saver
