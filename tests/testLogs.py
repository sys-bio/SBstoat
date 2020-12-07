# -*- coding: utf-8 -*-
"""
Created on Nov 20, 2020

@author: joseph-hellerstein
"""

from SBstoat import logs
from SBstoat.logs import BlockSpecification, Logger, Statistic

import io
import numpy as np
import os
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(DIR, "testLogger.png")
FILES = [LOG_PATH]
MSG = "Sample text"
BLOCK1 = "block1"
BLOCK2 = "block2"
COUNT = 50
       
 
class TestLogger(unittest.TestCase):

    def setUp(self):
        self.remove()
        self.logger = Logger(toFile=LOG_PATH, logPerformance=True,
               logLevel=logs.LEVEL_MAX)
    
    def tearDown(self):
        self.remove()

    def remove(self):
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)

    def isFile(self):
        return os.path.isfile(LOG_PATH)

    def read(self):
        if not self.isFile():
            raise RuntimeError("Missing log file.")
        with open(LOG_PATH, "r") as fd:
            lines = fd.readlines()
        return lines

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertFalse(self.isFile())
        self.assertEqual(self.logger.logLevel, logs.LEVEL_MAX)

    def testFileDescriptor(self):
        if IGNORE_TEST:
            return
        fd = self.logger.getFileDescriptor()
        self.assertIsInstance(fd, io.TextIOWrapper)
        fd.close()

    def _checkMsg(self, msg):
        lines = self.read()
        true = any([MSG in t for t in lines])
        self.assertTrue(true)
        return lines

    def testWrite(self):
        if IGNORE_TEST:
            return
        self.logger._write(MSG, 0)
        _ = self._checkMsg(MSG)

    def _testApi(self, method, logLevel):
        if IGNORE_TEST:
            return
        logger = Logger(toFile=LOG_PATH, logLevel=logLevel)
        stmt = "logger.%s(MSG)" % method
        exec(stmt)
        line1s = self._checkMsg(MSG)
        #
        logger = Logger(toFile=LOG_PATH, logLevel=0)
        stmt = "logger.%s(MSG)" % method
        exec(stmt)
        line2s = self.read()
        self.assertEqual(len(line1s), len(line2s))

    def testActivity(self):
        if IGNORE_TEST:
            return
        self._testApi("activity", logs.LEVEL_ACTIVITY)

    def testResult(self):
        if IGNORE_TEST:
            return
        self._testApi("result", logs.LEVEL_RESULT)

    def testStatus(self):
        if IGNORE_TEST:
            return
        self._testApi("status", logs.LEVEL_STATUS)

    def testException(self):
        if IGNORE_TEST:
            return
        self._testApi("status", logs.LEVEL_EXCEPTION)

    def testStartBlock(self):
        if IGNORE_TEST:
            return
        guid = self.logger.startBlock(BLOCK1)
        self.assertLess(guid, BlockSpecification.guid)
        self.assertEqual(len(self.logger.blockDct), 1)

    def testEndBlock(self):
        if IGNORE_TEST:
            return
        guid1 = self.logger.startBlock(BLOCK1)
        guid2 = self.logger.startBlock(BLOCK2)
        self.logger.endBlock(guid2)
        self.logger.endBlock(guid1)
        self.assertGreater(self.logger.statisticDct[BLOCK1].total,
              self.logger.statisticDct[BLOCK2].total)

    def testNoLogPerformance(self):
        if IGNORE_TEST:
            return
        logger = Logger(toFile=LOG_PATH, logPerformance=False,
               logLevel=logs.LEVEL_MAX)
        guid = logger.startBlock(BLOCK1)
        self.assertEqual(len(self.logger.blockDct), 0)
        logger.endBlock(guid)
        self.assertEqual(len(self.logger.blockDct), 0)

    def testPerformanceReport(self):
        if IGNORE_TEST:
            return
        def test(numBlock, sleepTime):
            logger = Logger(logPerformance=True)
            for idx in range(numBlock):
                block = "blk_%d" % idx
                guid = logger.startBlock(block)
                time.sleep(sleepTime)
                logger.endBlock(guid)
            df = logger.performanceDF
            self.assertLess(np.abs(sleepTime - df["mean"].mean()), sleepTime)
            self.assertEqual(df["count"].mean(), 1.0)
        #
        test(3, 0.1)
        test(30, 0.1)

    def testJoin(self):
        if IGNORE_TEST:
            return
        NAMES = ["aa", "bbb", "z"]
        result = Logger.join(*NAMES)
        for name in NAMES:
            self.assertGreaterEqual(result.index(name), 0)

    def testCopy(self):
        if IGNORE_TEST:
            return
        newLogger = self.logger.copy()
        self.assertTrue(self.logger.equals(newLogger))

    
        
       
 
class TestBlockSpecification(unittest.TestCase):

    def setUp(self):
        self.spec = BlockSpecification(BLOCK1)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(time.time(), self.spec.startTime)
        self.assertEqual(self.spec.block, BLOCK1)
        self.assertLess(self.spec.guid, BlockSpecification.guid)
        self.assertIsNone(self.spec.duration)

    def testSetDuration(self):
        if IGNORE_TEST:
            return
        self.spec.setDuration()
        self.assertLess(self.spec.duration, 10e-4)

 
class TestStatistic(unittest.TestCase):

    def setUp(self):
        self.statistic = Statistic(BLOCK1)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsNone(self.statistic.mean)

    def makeUpdates(self, statistic=None):
        if statistic is None:
            statistic = self.statistic
        for _ in range(COUNT):
             statistic.update(np.random.uniform())

    def testUpdate(self):
        if IGNORE_TEST:
            return
        self.makeUpdates()
        self.assertEqual(COUNT, self.statistic.count)
        self.assertLess(np.abs(self.statistic.total - COUNT*0.5), COUNT*0.2)

    def testCopyEqual(self):
        if IGNORE_TEST:
            return
        self.makeUpdates()
        statistic = self.statistic.copy()
        self.assertTrue(statistic.equals(self.statistic))

    def testMerge(self):
        if IGNORE_TEST:
            return
        statistic = Statistic(BLOCK1)
        self.makeUpdates()
        mergedStatistic = statistic.merge(self.statistic)
        self.assertTrue(mergedStatistic.equals(self.statistic))
        #
        mergedStatistic = self.statistic.merge(self.statistic)
        self.assertEqual(mergedStatistic.count, 2*COUNT)

    def testSummarize(self):
        if IGNORE_TEST:
            return
        self.statistic.summarize()
        self.assertEqual(self.statistic.mean, 0)
        #
        self.makeUpdates()
        self.statistic.summarize()
        self.assertLess(np.abs(self.statistic.mean - 0.5), 0.2)


if __name__ == '__main__':
    unittest.main()
