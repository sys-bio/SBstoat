# -*- coding: utf-8 -*-
"""
Created on Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

from SBStoat import _modelFitterBootstrap as mfb
from SBStoat.namedTimeseries import NamedTimeseries, TIME
from tests import _testHelpers as th

import numpy as np
import os
import time
import unittest


IGNORE_TEST = False
IS_PLOT = False
TIMESERIES = th.getTimeseries()
FITTER = th.getFitter(cls=mfb.ModelFitterBootstrap)
FITTER.fitModel()
        

class TestModelFitterBootstrap(unittest.TestCase):

    def setUp(self):
        self.timeseries = TIMESERIES
        self.fitter = FITTER
        self.fitter.bootstrapResult = None

    def testRunBootstrap(self):
        if IGNORE_TEST:
            return
        NUM_ITERATION = 10
        MAX_DIFF = 4
        arguments = mfb._Arguments(self.fitter, 1, 0)
        arguments.numIteration = NUM_ITERATION
        parameterDct, numSuccessIteration = mfb._runBootstrap(arguments)
        self.assertEqual(numSuccessIteration, NUM_ITERATION)
        trues = [len(v)==NUM_ITERATION for _, v in parameterDct.items()]
        self.assertTrue(all(trues))
        # Test not too far from true values
        trues = [np.abs(np.mean(v) - th.PARAMETER_DCT[p]) <= MAX_DIFF
              for p, v in parameterDct.items()]
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

    def testCalcNewObserved(self):
        if IGNORE_TEST:
            return
        ts = mfb.calcObservedTS(self.fitter)
        self.assertEqual(len(ts),
              len(self.fitter.observedTS))
        self.assertGreater(ts["S1"][0], ts["S6"][0])
        self.assertGreater(ts["S6"][-1], ts["S1"][-1])

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
              reportInterval=100, maxProcess=2)
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


if __name__ == '__main__':
    unittest.main()
