# -*- coding: utf-8 -*-
#
# The loghandler's define
# Date: 2018-05-04

# import the lib
import logging
import os

# output file setting
LOG_FILE = 'output.log'                                 # log file's path
# define some struct
LOG_FORMAT = '[%(levelname)s] %(asctime)s %(filename)s[line:%(lineno)d]: %(message)s'
LOG_DATE_FORMAT = '[%a] %Y-%m-%d %H:%M:%S'              # the date's format


# init
def InitLog(path, renew):
    if renew and os.path.exists(path):
        os.remove(path)

    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt=LOG_DATE_FORMAT,
                        filename=path,
                        filemode='a')
    return


# the info
def Info(str):
    print "[INFO] " + str
    logging.info(str)


# the debug
def Debug(str):
    print "[DEBUG] " + str
    logging.debug(str)
    return


# the warning
def Warning(str):
    print "[WARNING] " + str
    logging.warning(str)
    return


# the error
def Error(str):
    print "[ERROR] " + str
    logging.error(str)
    return
