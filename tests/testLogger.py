# -*- coding: utf-8 -*-
"""
Created on Nov 20, 2020

@author: joseph-hellerstein
"""

from SBstoat import _logger

import io
import os
import unittest


IGNORE_TEST = False
IS_PLOT = False
DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(DIR, "testLogger.png")
FILES = [LOG_PATH]
MSG = "Sample text"
       
 
class TestLogger(unittest.TestCase):

    def setUp(self):
        self.remove()
        self.logger = _logger.Logger(toFile=LOG_PATH,
               logLevel=_logger.LEVEL_MAX)
    
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
        self.assertEqual(self.logger.level, _logger.LEVEL_MAX)

    def testFileDescriptor(self):
        if IGNORE_TEST:
            return
        fd = self.logger.getFileDescriptor()
        self.assertIsInstance(fd, io.TextIOWrapper)

    def testWrite(self):
        if IGNORE_TEST:
            return
        self.logger._write(MSG, 0)
        lines = self.read()
        self.assertTrue(MSG in lines[0])

    def _testApi(self, method, logLevel):
        if IGNORE_TEST:
            return
        logger = _logger.Logger(toFile=LOG_PATH, logLevel=logLevel)
        stmt = "logger.%s(MSG)" % method
        exec(stmt)
        line1s = self.read()
        true = any([MSG in t for t in line1s])
        self.assertTrue(true)
        #
        logger = _logger.Logger(toFile=LOG_PATH, logLevel=0)
        stmt = "logger.%s(MSG)" % method
        exec(stmt)
        line2s = self.read()
        self.assertEqual(len(line1s), len(line2s))

    def testActivity(self):
        if IGNORE_TEST:
            return
        self._testApi("activity", _logger.LEVEL_ACTIVITY)

    def testResult(self):
        if IGNORE_TEST:
            return
        self._testApi("result", _logger.LEVEL_RESULT)

    def testStatus(self):
        if IGNORE_TEST:
            return
        self._testApi("status", _logger.LEVEL_STATUS)

    def testException(self):
        if IGNORE_TEST:
            return
        self._testApi("status", _logger.LEVEL_EXCEPTION)


if __name__ == '__main__':
    unittest.main()
