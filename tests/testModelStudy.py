# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19, 2020

@author: joseph-hellerstein
"""

from SBstoat.modelStudy import ModelStudy, mkDataSourceDct
import tests._testHelpers as th

import matplotlib
import numpy as np
import os
import shutil
import unittest


COLNAME = "V"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(THIS_DIR, "tst_data.txt")
DATA_PATH2 = os.path.join(THIS_DIR, "data_file2.csv")
DATA_PATHS = [DATA_PATH, DATA_PATH, DATA_PATH]
SERIALIZE_DIR = os.path.join(THIS_DIR, "modelStudy")
DIRS = [SERIALIZE_DIR]
FILES = []
IGNORE_TEST = False
IS_PLOT = False
PARAMETERS_TO_FIT = [v for v in th.PARAMETER_DCT.keys()]
TIMESERIES = th.getTimeseries()
        

class TestModelFitterCore(unittest.TestCase):

    def setUp(self):
        self._remove()
        self.parametersToFit = list(th.PARAMETER_DCT.keys())
        self.study = ModelStudy(th.ANTIMONY_MODEL, DATA_PATHS,
              parametersToFit=PARAMETERS_TO_FIT,
              dirStudyPath=SERIALIZE_DIR, isPlot=IS_PLOT, useSerialized=True)
    
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
        study = ModelStudy(th.ANTIMONY_MODEL, DATA_PATHS,
              parametersToFit=self.parametersToFit,
              dirStudyPath=SERIALIZE_DIR, isPlot=IS_PLOT)
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
        self.study.bootstrap(numIteration=20)
        self.study.plotParameterEstimates()
        

class TestFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def testMkDataSourceDct(self):
        if IGNORE_TEST:
            return
        def test(dataSourceNames=None):
            dataSourceDct = mkDataSourceDct(DATA_PATH2, "V",
                  dataSourceNames=dataSourceNames)
            trues = [d.colnames[0] == COLNAME for d in dataSourceDct.values()]
            self.assertTrue(all(trues))
            keys = [k for k in dataSourceDct.keys()]
            firstTS = dataSourceDct[keys[0]]
            trues = [len(d) == len(firstTS) for d in dataSourceDct.values()]
        test()
        test(dataSourceNames=["P%d" % d for d in range(6)])

    def testMkDataSourceDctTimeRows(self):
        if IGNORE_TEST:
            return
        dataSourceDct1 = mkDataSourceDct(DATA_PATH2, "V", isTimeColumns=True)
        dataSourceDct2 = mkDataSourceDct(DATA_PATH2, "V", isTimeColumns=False)
        keys1 = list(dataSourceDct1)
        keys2 = list(dataSourceDct2)
        self.assertEqual(len(dataSourceDct1[keys1[0]]), len(keys2))
        

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    unittest.main()
