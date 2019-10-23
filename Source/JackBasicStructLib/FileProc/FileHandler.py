# -*- coding: utf-8 -*-
import linecache
import os


def GetPath(filename, num):
    path = linecache.getline(filename, num)
    path = path.rstrip("\n")
    return path


def Mkdir(path):
    # new folder
    path = path.strip()
    path = path.rstrip("\\")

    # check the file path
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

    return


def OutputData(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def OpenLogFile(path, is_continue=True):
    if is_continue == False:
        if os.path.exists(path):
            os.remove(path)

    file = open(path, 'a')

    return file
