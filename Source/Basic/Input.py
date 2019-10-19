# -*- coding: utf-8 -*-
from Define import *
from PIL import Image
from ArgParser import *
from LogHandler import *
from DataAugmentation import *
import random
import linecache
import ReadPFM
import Queue
import thread
import time
import tifffile
import glob
import cv2


# path handler
def GetPath(filename, num):
    path = linecache.getline(filename, num)
    path = path.rstrip("\n")
    return path


def GenerateImgPath(dirPath, filenameFormat, imgType, num):
    path = dirPath + filenameFormat % num + imgType
    return path


# random, get org points
def RandomOrg(w, h, crop_w, crop_h):
    x = random.randint(0, w - crop_w)
    y = random.randint(0, h - crop_h)
    return x, y


# read img, only png
def ReadImg(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)

    return img


def ReadPfmGroundTrue(path):
    imgGround, _ = ReadPFM.readPFM(path)
    imgGround = np.expand_dims(imgGround, axis=0)
    return imgGround


def ReadGroundTrue(path):
    img = Image.open(path)
    imgGround = np.ascontiguousarray(img, dtype=np.float32)/float(DEPTH_DIVIDING)
    imgGround = np.expand_dims(imgGround, axis=0)
    return imgGround


# flying thing groundtrue
def ReadRandomPfmGroundTrue(path, x, y, w, h):
    imgGround, _ = ReadPFM.readPFM(path)
    imgGround = ImgGroundSlice(imgGround, x, y, w, h)
    imgGround = np.expand_dims(imgGround, axis=0)
    return imgGround


# kitti groundtrue
def ReadRandomGroundTrue(path, x, y, w, h):
    img = Image.open(path)
    imgGround = np.ascontiguousarray(img, dtype=np.float32)/float(DEPTH_DIVIDING)
    imgGround = ImgGroundSlice(imgGround, x, y, w, h)
    imgGround = np.expand_dims(imgGround, axis=0)
    return imgGround


# Flying Things and Kitti
def RandomCropRawImage(args, num):
    # Get path
    # Get path
    pathL = GetPath(args.trainListPath, 2*num+1)
    pathR = GetPath(args.trainListPath, 2*(num + 1))
    pathGround = GetPath(args.trainLabelListPath, num + 1)

    w = args.corpedImgWidth
    h = args.corpedImgHeight

    # get the img, the random crop
    imgL = ReadImg(pathL)

    # random crop
    x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w, h)
    imgL = ImgSlice(imgL, x, y, w, h)

    imgL = ImgProcessing(imgL)
    imgL = np.expand_dims(imgL, axis=0)

    # the right img
    imgR = ReadImg(pathR)
    imgR = ImgSlice(imgR, x, y, w, h)
    imgR = ImgProcessing(imgR)
    imgR = np.expand_dims(imgR, axis=0)

    # get groundtrue
    clsImgGround = None
    if args.dataset == "KITTI":
        imgGround = ReadRandomGroundTrue(pathGround, x, y, w, h)
    else:
        imgGround = ReadRandomPfmGroundTrue(pathGround, x, y, w, h)

    return imgL, imgR, imgGround


# Val Flying Things and Kitti
def ValRandomCropRawImage(args, num):
    # Get path
    pathL = GetPath(args.valListPath, 2*num+1)
    pathR = GetPath(args.valListPath, 2*(num + 1))
    pathGround = GetPath(args.valLabelListPath, num + 1)

    w = args.corpedImgWidth
    h = args.corpedImgHeight
    # get the img, the random crop
    imgL = ReadImg(pathL)
    x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w, h)
    imgL = ImgSlice(imgL, x, y, w, h)

    imgL = ImgProcessing(imgL)
    imgL = np.expand_dims(imgL, axis=0)

    imgR = ReadImg(pathR)
    imgR = ImgSlice(imgR, x, y, w, h)
    imgR = ImgProcessing(imgR)
    imgR = np.expand_dims(imgR, axis=0)

    # get groundtrue
    if args.dataset == "KITTI":
        imgGround = ReadRandomGroundTrue(pathGround, x, y, w, h)
    else:
        imgGround = ReadRandomPfmGroundTrue(pathGround, x, y, w, h)

    return imgL, imgR, imgGround


# Padding Img, used in testing
def GetPadingTestData(args, num):
    pathL = GetPath(args.testListPath, 2*num+1)
    pathR = GetPath(args.testListPath, 2*(num + 1))

    imgL = ReadImg(pathL)
    imgL = np.expand_dims(imgL, axis=0)
    imgR = ReadImg(pathR)
    imgR = np.expand_dims(imgR, axis=0)

    # pading size
    top_pad = args.padedImgHeight - imgL.shape[1]
    left_pad = args.padedImgWidth - imgL.shape[2]

    # pading
    imgL = np.lib.pad(imgL, ((0, 0), (top_pad, 0), (0, left_pad),
                             (0, 0)), mode='constant', constant_values=0)
    imgR = np.lib.pad(imgR, ((0, 0), (top_pad, 0), (0, left_pad),
                             (0, 0)), mode='constant', constant_values=0)

    return imgL, imgR, top_pad, left_pad


def GetPadingValData(args, num):
    pathL = GetPath(args.valListPath, 2*num+1)
    pathR = GetPath(args.valListPath, 2*(num + 1))
    pathGround = GetPath(args.valLabelListPath, num + 1)

    imgL = ReadImg(pathL)
    imgR = ReadImg(pathR)

    if args.dataset == "KITTI":
        imgGround = ReadGroundTrue(pathGround)
    else:
        imgGround = ReadPfmGroundTrue(pathGround)

    # pading size
    top_pad = args.padedImgHeight - imgL.shape[1]
    left_pad = args.padedImgWidth - imgL.shape[2]

    # pading
    imgL = np.lib.pad(imgL, ((0, 0), (top_pad, 0), (0, left_pad),
                             (0, 0)), mode='constant', constant_values=0)
    imgR = np.lib.pad(imgR, ((0, 0), (top_pad, 0), (0, left_pad),
                             (0, 0)), mode='constant', constant_values=0)

    imgGround = np.lib.pad(imgGround, ((0, 0), (top_pad, 0), (0, left_pad)),
                           mode='constant', constant_values=0)

    return imgL, imgR, imgGround


def GetBatchImage(args, randomlist, num, isVal=False):
    for i in xrange(args.batchSize * args.gpu):
        idNum = randomlist[args.batchSize * args.gpu * num + i]

        if isVal == False:
            imgL, imgR, imgGround = RandomCropRawImage(args, idNum)       # get img
        else:
            imgL, imgR, imgGround = ValRandomCropRawImage(args, idNum)       # get img

        if i == 0:
            imgLs = imgL
            imgRs = imgR
            imgGrounds = imgGround
        else:
            imgLs = np.concatenate((imgLs, imgL), axis=0)
            imgRs = np.concatenate((imgRs, imgR), axis=0)
            imgGrounds = np.concatenate((imgGrounds, imgGround), axis=0)

    return imgLs, imgRs, imgGrounds


def GetBatchTestImage(args, randomlist, num, isVal=False):

    top_pads = []
    left_pads = []
    for i in xrange(args.batchSize * args.gpu):
        idNum = randomlist[args.batchSize * args.gpu * num + i]
        imgL, imgR, top_pad, left_pad = GetPadingTestData(args, idNum)       # get img

        top_pads.append(top_pad)
        left_pads.append(left_pad)
        if i == 0:
            imgLs = imgL
            imgRs = imgR
        else:
            imgLs = np.concatenate((imgLs, imgL), axis=0)
            imgRs = np.concatenate((imgRs, imgR), axis=0)

    return imgLs, imgRs, top_pads, left_pads


class Dataloader(object):
    """docstring for Dataloader"""

    def __init__(self, args):
        super(Dataloader, self).__init__()
        self.args = args
        self.randomTrainingList = range(args.imgNum)
        self.randomValList = range(args.valImgNum)
        self.trainQueue = Queue.Queue(maxsize=30)
        self.valQueue = Queue.Queue(maxsize=30)
        self.exitFlag = 0
        self.num_tr_batch = args.imgNum / args.batchSize / args.gpu
        self.num_val_batch = args.valImgNum / args.batchSize / args.gpu
        if self.num_tr_batch > 0:
            thread.start_new_thread(self._trainDataGenQueue_Thread, ())
        if self.num_val_batch > 0:
            thread.start_new_thread(self._valDataGenQueue_Thread, ())

        Info("The total img num : %d , The max batch : %d" % (args.imgNum, self.num_tr_batch))
        Info("The total val img num : %d , The max batch : %d" %
             (args.valImgNum, self.num_val_batch))

    def __del__(self):
        if self.num_tr_batch > 0:
            self.trainQueue.get()
        if self.num_val_batch > 0:
            self.valQueue.get()

        self.exitFlag = 1
        time.sleep(5)
        print "Dataloader has been deleted"

    def _randomList(self, randomTrainingList):
        random.shuffle(randomTrainingList)

    def _trainDataGenQueue_Thread(self):
        i = 0
        while True:
            if self.exitFlag:
                print "_trainDataGenQueue_Thread safe exited!"
                thread.exit()

            if self.trainQueue.full():
                time.sleep(5)
                continue

            if i >= self.num_tr_batch:
                i = 0
                self._randomList(self.randomTrainingList)

            imgLs, imgRs, imgGrounds = GetBatchImage(self.args, self.randomTrainingList, i)
            i = i + 1

            if i % 1000 == 0:
                print "Finish %d" % i

            dataList = []
            dataList.append(imgLs)
            dataList.append(imgRs)
            dataList.append(imgGrounds)
            self.trainQueue.put(dataList)

    def _valDataGenQueue_Thread(self):
        i = 0
        while True:
            if self.exitFlag:
                print "_valDataGenQueue_Thread safe exited!"
                thread.exit()

            if self.valQueue.full():
                time.sleep(5)
                continue

            if i >= self.num_val_batch:
                i = 0

            imgLs, imgRs, imgGrounds = GetBatchImage(self.args, self.randomValList, i, True)
            i = i + 1
            dataList = []
            dataList.append(imgLs)
            dataList.append(imgRs)
            dataList.append(imgGrounds)
            self.valQueue.put(dataList)

    def GetTrainData(self):
        dataList = self.trainQueue.get()
        return dataList[0], dataList[1], dataList[2]

    def GetValData(self):
        dataList = self.valQueue.get()
        return dataList[0], dataList[1], dataList[2]


class Dataloader_dfc(object):
    def __init__(self, args):
        self.args = args
        self.datasetNum = args.datasetNum
        self.randomDatasetInduces = range(self.datasetNum)
        self.trainQueue = Queue.Queue(maxsize=30)
        self.valQueue = Queue.Queue(maxsize=30)
        self.exitFlag = 0
        self.num_tr_batch = args.imgNum / args.batchSize / args.gpu
        self.num_val_batch = args.valImgNum / args.batchSize / args.gpu
        if self.num_tr_batch > 0:
            thread.start_new_thread(self._trainDataGenQueue_Thread, ())
        if self.num_val_batch > 0:
            thread.start_new_thread(self._valDataGenQueue_Thread, ())

    def __del__(self):
        self.exitFlag = 1
        time.sleep(5)
        print "Dataloader has been deleted"

    def _valDataGenQueue_Thread(self):
        data_imgLs, data_imgRs, data_imgGrounds, data_clsImgGrounds = load_val_data(self.args)

        valList = range(self.args.valImgNum)
        i = 0

        while True:
            if self.exitFlag:
                print "_valDataGenQueue_Thread safe exited!"
                thread.exit()

            if i >= self.num_val_batch:
                i = 0

            imgLs, imgRs, imgGrounds, clsImgGrounds = GetDFCBatchImg(
                self.args, valList, data_imgLs, data_imgRs,
                data_imgGrounds, data_clsImgGrounds, i, False)

            i = i + 1
            dataList = []
            dataList.append(imgLs)
            dataList.append(imgRs)
            dataList.append(imgGrounds)
            dataList.append(clsImgGrounds)
            self.valQueue.put(dataList)

    def _trainDataGenQueue_Thread(self):
        i = 0
        while True:
            if self.exitFlag:
                print "_trainDataGenQueue_Thread safe exited!"
                thread.exit()

            if i >= self.datasetNum:
                i = 0
                random.shuffle(self.randomDatasetInduces)

            data_imgLs, data_imgRs, data_disp_imgGrounds, data_cls_imgGrounds = load_train_data(
                self.args, self.randomDatasetInduces[i])
            imgNum = data_imgLs.shape[0]
            randomList = range(imgNum)
            random.shuffle(randomList)
            num_tr_batch = imgNum / self.args.batchSize / self.args.gpu
            print "Dataset Num:" + str(i) + ", Max Batch Size:" + str(num_tr_batch)
            for j in range(num_tr_batch):
                imgLs, imgRs, dispImgGrounds, clsImgGrounds = GetDFCBatchImg(
                    self.args, randomList, data_imgLs, data_imgRs,
                    data_disp_imgGrounds, data_cls_imgGrounds, j)
                dataList = []
                dataList.append(imgLs)
                dataList.append(imgRs)
                dataList.append(dispImgGrounds)
                dataList.append(clsImgGrounds)
                self.trainQueue.put(dataList)

            i = i + 1  # net dataset

    def GetTrainData(self):
        dataList = self.trainQueue.get()
        return dataList[0], dataList[1], dataList[2], dataList[3]

    def GetValData(self):
        dataList = self.valQueue.get()
        return dataList[0], dataList[1], dataList[2], dataList[3]


def load_train_data(args, index):
    filename = args.trainListPath + ".train.left.%d.npz" % (index+1)
    print("Loading... ", filename)
    imgLs = np.load(filename)['arr_0']

    filename = args.trainListPath + ".train.right.%d.npz" % (index+1)
    print("Loading... ", filename)
    imgRs = np.load(filename)['arr_0']

    filename = args.trainListPath + ".train.disparity.%d.npz" % (index+1)
    print("Loading... ", filename)
    dispImgGrounds = np.load(filename)['arr_0']

    filename = args.trainListPath + ".train.left_label.%d.npz" % (index+1)
    print("Loading... ", filename)
    clsImgGrounds = np.load(filename)['arr_0']

    return imgLs, imgRs, dispImgGrounds, clsImgGrounds


def GetDFCImg(args, data_imgLs, data_imgRs,
              data_disp_imgGrounds, data_cls_imgGrounds, index):
    # Read Img
    imgL = data_imgLs[index]
    imgR = data_imgRs[index]
    dispImgGround = data_disp_imgGrounds[index]
    clsImgGround = data_cls_imgGrounds[index]

    # Crop
    w = args.corpedImgWidth
    h = args.corpedImgHeight
    x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w, h)
    imgL = ImgSlice(imgL, x, y, w, h)
    imgR = ImgSlice(imgR, x, y, w, h)
    dispImgGround = ImgGroundSlice(dispImgGround, x, y, w, h)
    clsImgGround = ImgGroundSlice(clsImgGround, x, y, w, h)

    imgL, imgR, dispImgGround, clsImgGround = Vertical_Flip(imgL, imgR,
                                                            dispImgGround, clsImgGround)

    # Random color
    # imgL = RandomHueSaturationValue(imgL,
    #                                hue_shift_limit=(-30, 30),
    #                                sat_shift_limit=(-5, 5),
    #                                val_shift_limit=(-15, 15))
    # imgR = RandomHueSaturationValue(imgR,
    #                                hue_shift_limit=(-30, 30),
    #                                sat_shift_limit=(-5, 5),
    #                                val_shift_limit=(-15, 15))

    imgL = ImgProcessing(imgL)
    imgR = ImgProcessing(imgR)

    imgL = np.expand_dims(imgL, axis=0)
    imgR = np.expand_dims(imgR, axis=0)
    dispImgGround = np.expand_dims(dispImgGround, axis=0)
    clsImgGround = np.expand_dims(clsImgGround, axis=0)

    return imgL, imgR, dispImgGround, clsImgGround


def GetDFCBatchImg(args, randomlist, data_imgLs, data_imgRs,
                   data_disp_imgGrounds, data_cls_imgGrounds,
                   index, isTraining=True):
    for i in xrange(args.batchSize * args.gpu):
        idNum = randomlist[args.batchSize * args.gpu * index + i]

        if isTraining is True:
            imgL, imgR, dispImgGround, clsImgGround = GetDFCImg(
                args, data_imgLs, data_imgRs, data_disp_imgGrounds, data_cls_imgGrounds, idNum)
        else:
            imgL, imgR, dispImgGround, clsImgGround = GetDFCValImg(
                args, data_imgLs, data_imgRs, data_disp_imgGrounds, data_cls_imgGrounds, idNum)

        if i == 0:
            imgLs = imgL
            imgRs = imgR
            dispImgGrounds = dispImgGround
            clsImgGrounds = clsImgGround
        else:
            imgLs = np.concatenate((imgLs, imgL), axis=0)
            imgRs = np.concatenate((imgRs, imgR), axis=0)
            dispImgGrounds = np.concatenate((dispImgGrounds, dispImgGround), axis=0)
            clsImgGrounds = np.concatenate((clsImgGrounds, clsImgGround), axis=0)

    return imgLs, imgRs, dispImgGrounds, clsImgGrounds


def load_val_data(args):
    filename = args.valListPath + ".test.left.npz"
    print("Loading... ", filename)
    imgLs = np.load(filename)['arr_0']

    filename = args.valListPath + ".test.right.npz"
    print("Loading... ", filename)
    imgRs = np.load(filename)['arr_0']

    filename = args.valListPath + ".test.disparity.npz"
    print("Loading... ", filename)
    dispImgGrounds = np.load(filename)['arr_0']

    filename = args.valListPath + ".test.left_label.npz"
    print("Loading... ", filename)
    clsImgGrounds = np.load(filename)['arr_0']

    return imgLs, imgRs, dispImgGrounds, clsImgGrounds


def GetDFCValImg(args, data_imgLs, data_imgRs, data_imgGrounds, data_cls_imgGrounds, index):
    imgL = data_imgLs[index]
    imgR = data_imgRs[index]
    dispImgGround = data_imgGrounds[index]
    clsImgGround = data_cls_imgGrounds[index]

    imgL = ImgProcessing(imgL)
    imgR = ImgProcessing(imgR)

    imgL = np.expand_dims(imgL, axis=0)
    imgR = np.expand_dims(imgR, axis=0)
    dispImgGround = np.expand_dims(dispImgGround, axis=0)
    clsImgGround = np.expand_dims(clsImgGround, axis=0)

    return imgL, imgR, dispImgGround, clsImgGround


def load_test_data(args):
    files = glob.glob(args.testListPath + '*LEFT_RGB.tif')
    return files


def GetDFCTestImg(args, data_imgLs, index):
    # Get Img
    name = data_imgLs[index]
    left_name = name
    pos = name.find('LEFT_RGB')
    right_name = name[0:pos] + 'RIGHT_RGB.tif'

    name = os.path.basename(name)
    pos = name.find('LEFT_RGB')
    name = name[0:pos]

    # Read Img
    imgL = tifffile.imread(left_name)
    imgR = tifffile.imread(right_name)

    # Processing
    imgL = ImgProcessing(imgL)
    imgR = ImgProcessing(imgR)

    imgL = np.expand_dims(imgL, axis=0)
    imgR = np.expand_dims(imgR, axis=0)

    return imgL, imgR, name


def GetDFCTestImg_v2(args, data_imgLs, index):
    name = data_imgLs[index]
    left_name = name
    pos = name.find('LEFT_RGB')
    right_name = name[0:pos] + 'RIGHT_RGB.tif'

    name = os.path.basename(name)
    pos = name.find('LEFT_RGB')
    name = name[0:pos]

    # Read Img
    imgL_s = tifffile.imread(left_name)
    imgR_s = tifffile.imread(right_name)

    imgL = ImgProcessing(imgL_s)
    imgL = TestDataAugmentation(imgL)
    imgR = ImgProcessing(imgR_s)
    imgR = TestDataAugmentation(imgR)

    #imgL = np.expand_dims(imgL, axis=0)
    #imgR = np.expand_dims(imgR, axis=0)

    # print imgL.shape

    #imgL = np.expand_dims(imgL, axis=0)

    return imgL, imgR, name, imgL_s, imgR_s


def GetBatchDFCTestImage(args, randomlist, data_imgLs, num, isVal=False):
    names = []
    for i in xrange(args.batchSize * args.gpu):
        idNum = randomlist[args.batchSize * args.gpu * num + i]
        imgL, imgR, name, imgL_s, imgR_s = GetDFCTestImg_v2(args, data_imgLs, idNum)       # get img

        names.append(name)
        imgL_s = np.expand_dims(imgL_s, axis=0)
        imgR_s = np.expand_dims(imgR_s, axis=0)

        if i == 0:
            imgLs = imgL
            imgRs = imgR
            imgL_gs = imgL_s
            imgR_gs = imgR_s
        else:
            imgLs = np.concatenate((imgLs, imgL), axis=0)
            imgRs = np.concatenate((imgRs, imgR), axis=0)
            imgL_gs = np.concatenate((imgL_gs, imgL_s), axis=0)
            imgR_gs = np.concatenate((imgRs, imgR_s), axis=0)

    return imgLs, imgRs, names, imgL_gs, imgR_gs


if __name__ == "__main__":
    args = ParseArgs()
    args.trainListPath = '/home2/Documents/DFC2019_track2_trainval/Track_Train_npz/dfc2019.track2'
    args.valListPath = '/home2/Documents/DFC2019_track2_trainval/Track_Train_npz/dfc2019.track2'
    args.testListPath = '/home2/Documents/DFC2019_track2_trainval/Test-Track2/'
    args.corpedImgWidth = 512
    args.corpedImgHeight = 512
    args.batchSize = 5
    args.gpu = 2
    args.imgNum = 0
    args.valImgNum = 215

    data_imgLs = load_test_data(args)
    print len(data_imgLs)
    randlist = range(len(data_imgLs))

    for i in range(5):
        imgLs, imgRs, names = GetBatchDFCTestImage(args, randlist, data_imgLs, i)

        for j in range(args.batchSize * args.gpu):
            print names[j]

    '''
    dataloader = Dataloader_dfc(args)

    for i in range(30):
        imgLs, imgRs, imgGrounds = dataloader.GetValData()
        print imgLs.shape
        print imgRs.shape
        print imgGrounds.shape
        print '*****'

    '''

    '''
    data_imgLs, data_imgRs, data_imgGrounds = load_val_data(args)
    print data_imgLs.shape
    print data_imgGrounds.shape
    randomlist = range(data_imgLs.shape[0])

    for j in range(25):
        imgLs, imgRs, imgGrounds = GetDFCBatchImg(
            args, randomlist, data_imgLs, data_imgRs, data_imgGrounds, j, False)
        print imgLs.shape
        print imgRs.shape
        print imgGrounds.shape

    '''

    '''
    KITTI Test
    args = ParseArgs()      # parse args
    args.testListPath = '../../Dataset/testlist_kitti_2015.txt'
    args.trainListPath = '../../Dataset/trainlist_kitti_2015.txt'
    args.trainLabelListPath = '../../Dataset/labellist_kitti_2015.txt'
    args.valListPath = '../../Dataset/trainlist_kitti_2015.txt'
    args.valLabelListPath = '../../Dataset/labellist_kitti_2015.txt'
    args.corpedImgWidth = 640
    args.corpedImgHeight = 320
    args.batchSize = 1
    args.gpu = 2
    args.imgNum = 400
    randomTrainingList = range(args.imgNum)
    print args.imgNum / args.batchSize / args.gpu
    dataloader = Dataloader(args)
    print '**************'
    start_time = time.time()
    for i in range(args.imgNum / args.gpu / args.batchSize):
        imgLs, imgRs, imgGrounds = GetBatchImage(args, randomTrainingList, i, True)
        #imgLs, imgRs, imgGrounds = dataloader.GetTrainData()
        # print imgLs.shape
        # print imgRs.shape
        # print imgGrounds.shape
    duration = time.time() - start_time
    print '%.3f' % duration
    '''
