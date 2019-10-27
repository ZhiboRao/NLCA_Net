# -*- coding: utf-8 -*-
from JackBasicStructLib.ImgProc.ImgHandler import *
from JackBasicStructLib.FileProc.FileHandler import *
from JackBasicStructLib.ImgProc.DataAugmentation import *


# output file setting
DEPTH_DIVIDING = 256.0


class KittiFlyingDataloader(object):
    def __init__(self):
        super(KittiFlyingDataloader, self).__init__()
        pass

    def SaveTestData(self, args, img, num):
        path = self.__GenerateOutImgPath(args.resultImgDir, args.saveFormat, args.imgType, num)
        img = self.__DepthToImgArray(img)
        self.__SavePngImg(path, img)

    def GetBatchImage(self, args, randomlist, num, isVal=False):
        for i in xrange(args.batchSize * args.gpu):
            idNum = randomlist[args.batchSize * args.gpu * num + i]

            if isVal == False:
                imgL, imgR, imgGround = self.__RandomCropRawImage(args, idNum)       # get img
            else:
                imgL, imgR, imgGround = self.__ValRandomCropRawImage(args, idNum)       # get img

            if i == 0:
                imgLs = imgL
                imgRs = imgR
                imgGrounds = imgGround
            else:
                imgLs = np.concatenate((imgLs, imgL), axis=0)
                imgRs = np.concatenate((imgRs, imgR), axis=0)
                imgGrounds = np.concatenate((imgGrounds, imgGround), axis=0)

        return imgLs, imgRs, imgGrounds

    def GetBatchTestImage(self, args, randomlist, num, isVal=False):
        top_pads = []
        left_pads = []
        for i in xrange(args.batchSize * args.gpu):
            idNum = randomlist[args.batchSize * args.gpu * num + i]
            imgL, imgR, top_pad, left_pad = self.__GetPadingTestData(args, idNum)       # get img

            top_pads.append(top_pad)
            left_pads.append(left_pad)
            if i == 0:
                imgLs = imgL
                imgRs = imgR
            else:
                imgLs = np.concatenate((imgLs, imgL), axis=0)
                imgRs = np.concatenate((imgRs, imgR), axis=0)

        return imgLs, imgRs, top_pads, left_pads

    def __GenerateOutImgPath(self, dirPath, filenameFormat, imgType, num):
        path = dirPath + filenameFormat % num + imgType
        return path

    def __DepthToImgArray(self, img):
        img = np.array(img)
        img = (img * float(DEPTH_DIVIDING)).astype(np.uint16)
        return img

        # save the png file
    def __SavePngImg(self, path, img):
        cv2.imwrite(path, img)

    def __ReadRandomPfmGroundTrue(self, path, x, y, w, h):
        # flying thing groundtrue
        imgGround, _ = ReadPFM(path)
        imgGround = ImgGroundSlice(imgGround, x, y, w, h)
        imgGround = np.expand_dims(imgGround, axis=0)
        return imgGround

    def __ReadRandomGroundTrue(self, path, x, y, w, h):
        # kitti groundtrue
        img = Image.open(path)
        imgGround = np.ascontiguousarray(img, dtype=np.float32)/float(DEPTH_DIVIDING)
        imgGround = ImgGroundSlice(imgGround, x, y, w, h)
        imgGround = np.expand_dims(imgGround, axis=0)
        return imgGround

    def __ReadData(self, args, pathL, pathR, pathGround):
        # Flying Things and Kitti
        w = args.corpedImgWidth
        h = args.corpedImgHeight

        # get the img, the random crop
        imgL = ReadImg(pathL)
        imgR = ReadImg(pathR)

        # random crop
        x, y = RandomOrg(imgL.shape[1], imgL.shape[0], w, h)
        imgL = ImgSlice(imgL, x, y, w, h)
        imgL = Standardization(imgL)
        imgL = np.expand_dims(imgL, axis=0)

        # the right img
        imgR = ImgSlice(imgR, x, y, w, h)
        imgR = Standardization(imgR)
        imgR = np.expand_dims(imgR, axis=0)

        # get groundtrue
        if args.dataset == "KITTI":
            imgGround = self.__ReadRandomGroundTrue(pathGround, x, y, w, h)
        else:
            imgGround = self.__ReadRandomPfmGroundTrue(pathGround, x, y, w, h)

        return imgL, imgR, imgGround

    def __RandomCropRawImage(self, args, num):
        # Get path
        pathL = GetPath(args.trainListPath, 2*num+1)
        pathR = GetPath(args.trainListPath, 2*(num + 1))
        pathGround = GetPath(args.trainLabelListPath, num + 1)

        imgL, imgR, imgGround = self.__ReadData(args, pathL, pathR, pathGround)

        return imgL, imgR, imgGround

    # Val Flying Things and Kitti
    def __ValRandomCropRawImage(self, args, num):
        # Get path
        pathL = GetPath(args.valListPath, 2*num+1)
        pathR = GetPath(args.valListPath, 2*(num + 1))
        pathGround = GetPath(args.valLabelListPath, num + 1)
        imgL, imgR, imgGround = self.__ReadData(args, pathL, pathR, pathGround)

        return imgL, imgR, imgGround

    # Padding Img, used in testing
    def __GetPadingTestData(self, args, num):
        pathL = GetPath(args.testListPath, 2*num+1)
        pathR = GetPath(args.testListPath, 2*(num + 1))

        imgL = ReadImg(pathL)
        imgL = Standardization(imgL)
        imgL = np.expand_dims(imgL, axis=0)
        imgR = ReadImg(pathR)
        imgR = Standardization(imgR)
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
