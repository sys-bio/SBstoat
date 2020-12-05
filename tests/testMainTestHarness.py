# -*- coding: utf-8 -*-
"""
Created on Nov 16, 2020

@author: joseph-hellerstein
"""

from SBstoat.mainTestHarness import Runner
from SBstoat.logs import Logger

import os
import unittest
import matplotlib
import matplotlib.pyplot as plt


IGNORE_TEST = False
IS_PLOT = False
DIR = os.path.dirname(os.path.abspath(__file__))
PCL_PATH = os.path.join(DIR, "testMainTestHarness.pcl")
FIG_PATH = os.path.join(DIR, "testMainTestHarness.png")
FILES = [PCL_PATH, FIG_PATH]
FIRST_MODEL = 200
NUM_MODEL = 1
DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(DIR, "testMainTestHarness.log")
if IGNORE_TEST:
    LOGGER = Logger()
else:
    LOGGER = Logger(toFile=LOG_FILE)

if os.path.isfile(LOG_FILE):
    os.remove(LOG_FILE)
       
 
class TestRunner(unittest.TestCase):

    def setUp(self):
        self._remove()
        self.runner = Runner(firstModel=FIRST_MODEL, numModel=NUM_MODEL,
              useExistingData=False, figPath=FIG_PATH, pclPath=PCL_PATH,
              isPlot=IS_PLOT, numIteration=20, logger=LOGGER)
    
    def tearDown(self):
        self._remove()

    def _remove(self):
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(os.path.isfile(PCL_PATH))
        self.assertEqual(len(self.runner.fitModelRelerrors), 0)

    def testRunPlot(self):
        if IGNORE_TEST:
            return
        runner = Runner(firstModel=300, numModel=2,
              useExistingData=False, figPath=FIG_PATH, pclPath=PCL_PATH,
              isPlot=IS_PLOT, logger=LOGGER)
        runner.run()
        self.assertGreater(len(runner.fitModelRelerrors), 0)
        self.assertGreater(len(runner.bootstrapRelerrors), 0)
        self.assertEqual(runner.numModel, runner.numModel)

    def testSaveRestore(self):
        if IGNORE_TEST:
            return
        self.runner.run()
        self.assertTrue(os.path.isfile(PCL_PATH))
        #
        runner = Runner(firstModel=FIRST_MODEL, numModel=NUM_MODEL,
              useExistingData=True, figPath=FIG_PATH, pclPath=PCL_PATH,
              isPlot=IS_PLOT, logger=LOGGER)
        runner.run()
        runner.useExistingData = False  # Change so that test works
        isEqual = self.runner.equals(runner)
        self.assertTrue(self.runner.equals(runner))

    def testBug(self):
        if IGNORE_TEST:
            return
        runner = Runner(firstModel=607, numModel=1,
              useExistingData=False, figPath=FIG_PATH, pclPath=PCL_PATH,
              isPlot=IS_PLOT, logger=Logger())
        runner.run()


if __name__ == '__main__':
    if IS_PLOT:
        matplotlib.use('TkAgg')
    unittest.main()
