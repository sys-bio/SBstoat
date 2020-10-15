# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19, 2020

@author: joseph-hellerstein
"""

from SBstoat.modelStudy import ModelStudy
import tests._testHelpers as th

import matplotlib
import numpy as np
import os
import shutil
import unittest


IGNORE_TEST = True
TIMESERIES = th.getTimeseries()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SERIALIZE_DIR = os.path.join(THIS_DIR, "modelStudy")
DATA_FILE = os.path.join(THIS_DIR, "tst_data.txt")
DATA_FILE2 = os.path.join(THIS_DIR, "tst_data2.txt")
FILES = []
DIRS = [SERIALIZE_DIR]
        

class TestModelFitterCore(unittest.TestCase):

    def setUp(self):
        self._remove()
        parametersToFit = list(th.PARAMETER_DCT.keys())
        self.study = ModelStudy(th.ANTIMONY_MODEL,
              [DATA_FILE, DATA_FILE2], parametersToFit,
              dirPath=SERIALIZE_DIR)
    
    def tearDown(self):
        self._remove()

    def _remove(self):
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)
        for ddir in DIRS:
            if os.path.isdir(ddir):
                shutil.rmtree(ddir)

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.study.fitterDct.values()), 0)
        self.assertTrue(os.path.isfile(DATA_FILE))
        self.assertTrue(os.path.isfile(DATA_FILE2))

    def testFitModel(self):
        if IGNORE_TEST:
            return
        # Smoke test
        self.study.fitModel()

    def testFitBootstrap(self):
        if IGNORE_TEST:
            return
        self.study.bootstrap(numIteration=10)
        for fitter in self.study.fitterDct.values():
            self.assertIsNotNone(fitter.bootstrapResult)

    def testPlotFitAll(self):
        # TESTING
        self.study.fitModel()
        self.study.plotFitAll()
        #
        self.study.bootstrap()
        self.study.plotFitAll()
        

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    unittest.main()
