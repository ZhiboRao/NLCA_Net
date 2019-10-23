from JackBasicStructLib.Basic.Define import *


class NetWorkInferenceTemplate(object):
	__metaclass__ = ABCMeta
    def __init__(self):
        pass

    @abstractmethod
    def Inference(self, args, is_training=True):
        return paras, model, dataloader
