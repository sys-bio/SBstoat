# -*- coding: utf-8 -*-
"""
Created on Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

from SBstoat import _modelFitterBootstrap as mfb
from SBstoat.namedTimeseries import NamedTimeseries, TIME
from tests import _testHelpers as th
from SBstoat.observationSynthesizer import  \
      ObservationSynthesizerRandomizedResiduals,  \
      ObservationSynthesizerRandomErrors

import copy
import lmfit
import numpy as np
import os
import pandas as pd
import pickle
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
TIMESERIES = th.getTimeseries()
FITTER = th.getFitter(cls=mfb.ModelFitterBootstrap)
FITTER.fitModel()
NUM_ITERATION = 50
DIR = os.path.dirname(os.path.abspath(__file__))
FILE_SERIALIZE = os.path.join(DIR, "modelFitterBootstrap.pcl")
FILES = [FILE_SERIALIZE]
MEAN_UNIFORM = 0.5  # Mean of uniform distribution
STD_UNIFORM = np.sqrt(1.0/12)  # Standard deviation of uniform
        

class TestModelFitterBootstrap(unittest.TestCase):

    def setUp(self):
        self._remove()
        self.timeseries = TIMESERIES
        self.fitter = FITTER
        self.fitter.bootstrapResult = None
    
    def tearDown(self):
        self._remove()

    def _remove(self):
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)

    def testRunBootstrap(self):
        if IGNORE_TEST:
            return
        NUM_ITERATION = 10
        MAX_DIFF = 4
        arguments = mfb._Arguments(self.fitter, 1, 0, 
              synthesizerClass=ObservationSynthesizerRandomErrors,
              std=0.01)
        arguments.numIteration = NUM_ITERATION
        bootstrapResult = mfb._runBootstrap(arguments)
        self.assertEqual(bootstrapResult.numIteration, NUM_ITERATION)
        trues = [len(v)==NUM_ITERATION for _, v in 
              bootstrapResult.parameterDct.items()]
        self.assertTrue(all(trues))
        # Test not too far from true values
        trues = [np.abs(np.mean(v) - th.PARAMETER_DCT[p]) <= MAX_DIFF
              for p, v in bootstrapResult.parameterDct.items()]
        self.assertTrue(all(trues))

    def checkParameterValues(self):
        dct = self.fitter.params.valuesdict()
        self.assertEqual(len(dct), len(self.fitter.parametersToFit))
        #
        for value in dct.values():
            self.assertTrue(isinstance(value, float))
        return dct
        
    def testGetFittedParameters(self):
        if IGNORE_TEST:
            return
        values = self.fitter.getFittedParameters()
        _ = self.checkParameterValues()
        #
        self.fitter.bootstrap(numIteration=5)
        values = self.fitter.getFittedParameters()
        _ = self.checkParameterValues()

    def testBoostrapTimeMultiprocessing(self):
        return
        if IGNORE_TEST:
            return
        print("\n")
        def timeIt(maxProcess):
            startTime = time.time()
            self.fitter.bootstrap(numIteration=10000,
                  reportInterval=1000, maxProcess=maxProcess)
            elapsed_time = time.time() - startTime
            print("%s processes: %3.2f" % (str(maxProcess), elapsed_time))
        #
        timeIt(None)
        timeIt(1)
        timeIt(2)
        timeIt(4)

    def testBoostrap(self):
        if IGNORE_TEST:
            return
        self.fitter.bootstrap(numIteration=500,
              reportInterval=100, maxProcess=2,
              serializePath=FILE_SERIALIZE)
        NUM_STD = 10
        result = self.fitter.bootstrapResult
        for p in self.fitter.parametersToFit:
            isLowerOk = result.meanDct[p]  \
                  - NUM_STD*result.stdDct[p]  \
                  < th.PARAMETER_DCT[p]
            isUpperOk = result.meanDct[p]  \
                  + NUM_STD*result.stdDct[p]  \
                  > th.PARAMETER_DCT[p]
            self.assertTrue(isLowerOk)
            self.assertTrue(isUpperOk)
        #
        fitter = mfb.ModelFitterBootstrap.deserialize(FILE_SERIALIZE)
        self.assertIsNotNone(fitter.bootstrapResult)

    def testGetFittedParameterStds(self):
        if IGNORE_TEST:
            return
        with self.assertRaises(ValueError):
            _ = self.fitter.getFittedParameterStds()
        #
        self.fitter.bootstrap(numIteration=3)
        stds = self.fitter.getFittedParameterStds()
        for std in stds:
            self.assertTrue(isinstance(std, float))

    def testBootstrap(self):
        if IGNORE_TEST:
            return
        self.fitter.bootstrap(numIteration=500,
              synthesizerClass=ObservationSynthesizerRandomErrors,
              reportInterval=100, maxProcess=2, std=0.01)
        result = self.fitter.bootstrapResult
        self.assertTrue(result is not None)
        

class TestBootstrapResult(unittest.TestCase):

    def setUp(self):
        self.names = ["A", "B"]
        self.parameterDct = {n: np.random.randint(10, 20, NUM_ITERATION)
              for n in self.names}
        timeseriesDF = pd.DataFrame(self.parameterDct)
        timeseriesDF.index = range(NUM_ITERATION)
        timeseriesDF.index.name = TIME
        self.ts = NamedTimeseries(dataframe=timeseriesDF)
        self.sumTS = self.ts.copy(isInitialize=True)
        self.ssqTS = self.ts.copy(isInitialize=True)
        for _ in range(NUM_ITERATION):
            for name in self.names:
                vec = np.random.random(NUM_ITERATION)
                self.sumTS[name] = vec + self.sumTS[name]
                self.ssqTS[name] = vec**2 + self.ssqTS[name]
        self.statisticDct = {
              mfb.COL_SUM: self.sumTS,
              mfb.COL_SSQ: self.ssqTS,
              }
        self.bootstrapResult = mfb.BootstrapResult(NUM_ITERATION,
              self.parameterDct, self.statisticDct)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        diff = set(self.parameterDct.keys()).symmetric_difference(
              self.bootstrapResult.parameters)
        self.assertEqual(len(diff), 0)
        #
        diff = set(self.bootstrapResult.statisticDct.keys()).symmetric_difference(
              [mfb.COL_SSQ, mfb.COL_SUM])
        self.assertEqual(len(diff), 0)

    def testParams(self):
        if IGNORE_TEST:
            return
        params = self.bootstrapResult.params
        name = self.names[0]
        self.assertEqual(params.valuesdict()[name],
              np.mean(self.parameterDct[name]))

    def testMerge(self):
        if IGNORE_TEST:
            return
        bootstrapResult = mfb.BootstrapResult(NUM_ITERATION,
              self.parameterDct, self.statisticDct)
        mergedResult = mfb.BootstrapResult.merge(
              [self.bootstrapResult, bootstrapResult])
        self.assertEqual(mergedResult.numIteration, 2*NUM_ITERATION)
        self.assertEqual(len(mergedResult.parameterDct[self.names[0]]),
              mergedResult.numIteration)

    def testMeanFittedTS(self):
        if IGNORE_TEST:
            return
        meanFittedTS = self.bootstrapResult.meanFittedTS
        overallMean = np.mean(meanFittedTS[self.names].flatten())
        self.assertLess(np.abs(overallMean - MEAN_UNIFORM), 0.1)

    def testStdFittedTS(self):
        if IGNORE_TEST:
            return
        stdFittedTS = self.bootstrapResult.stdFittedTS
        overallStd = np.mean(stdFittedTS[self.names].flatten())
        self.assertLess(np.abs(overallStd - STD_UNIFORM), 0.01)


if __name__ == '__main__':
    unittest.main()
