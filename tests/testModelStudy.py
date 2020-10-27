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


IGNORE_TEST = False
IS_PLOT = False
TIMESERIES = th.getTimeseries()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SERIALIZE_DIR = os.path.join(THIS_DIR, "modelStudy")
DATA_FILE = os.path.join(THIS_DIR, "tst_data.txt")
DATA_FILES = [DATA_FILE, DATA_FILE, DATA_FILE]
FILES = []
DIRS = [SERIALIZE_DIR]
        

class TestModelFitterCore(unittest.TestCase):

    def setUp(self):
        self._remove()
        self.parametersToFit = list(th.PARAMETER_DCT.keys())
        self.study = ModelStudy(th.ANTIMONY_MODEL,
              DATA_FILES, self.parametersToFit,
              dirPath=SERIALIZE_DIR, isPlot=IS_PLOT)
    
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
        # Ensure that ModelFitters are serialized correctly
        study = ModelStudy(th.ANTIMONY_MODEL, DATA_FILES,
              self.parametersToFit,
              dirPath=SERIALIZE_DIR, isPlot=IS_PLOT)
        for name in self.study.instanceNames:
            self.assertEqual(study.fitterDct[name].modelSpecification,
                  self.study.fitterDct[name].modelSpecification)

    def testFitModel(self):
        if IGNORE_TEST:
            return
        self.study.fitModel()
        names = [v for v in self.study.fitterDct.keys()]
        params0 = self.study.fitterDct[names[0]].params
        params1 = self.study.fitterDct[names[1]].params
        dct0 = params0.valuesdict()
        dct1 = params1.valuesdict()
        for key, value in dct0.items():
            self.assertEqual(value, dct1[key]) 

    def testFitBootstrap(self):
        if IGNORE_TEST:
            return
        self.study.bootstrap(numIteration=10)
        for fitter in self.study.fitterDct.values():
            self.assertIsNotNone(fitter.bootstrapResult)

    def testPlotFitAll(self):
        if IGNORE_TEST:
            return
        self.study.fitModel()
        self.study.plotFitAll()
        #
        self.study.bootstrap()
        self.study.plotFitAll()

    def testPlotParameterEstimates(self):
        if IGNORE_TEST:
            return
        self.study.bootstrap(numIteration=10)
        self.study.plotParameterEstimates()
        

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    unittest.main()
