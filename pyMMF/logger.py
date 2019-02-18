#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Sebastien M. Popoff
"""

import logging

def get_logger(name):
        loglevel = logging.DEBUG
        logger = logging.getLogger(name)
        if not getattr(logger, 'handler_set', None):
            logger.setLevel(logging.INFO)
            logFormatter = logging.Formatter("%(asctime)s - %(name)-10.10s [%(levelname)-7.7s]  %(message)s") #[%(threadName)-12.12s] 
            fileHandler = logging.FileHandler("{0}/{1}.log".format('./', 'pyMMF'))
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)
            logger.addHandler(consoleHandler)
            logger.setLevel(loglevel)
            logger.handler_set = True
        return logger
    

def handleException(excType, excValue, traceback, logger):
    logger.error("Uncaught exception", exc_info=(excType, excValue, traceback))