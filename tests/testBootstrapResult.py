# -*- coding: utf-8 -*-
"""
Created on Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

from SBstoat import _bootstrapResult as br
from SBstoat.timeseriesStatistic import TimeseriesStatistic
from SBstoat import _modelFitterBootstrap as mfb
from SBstoat.namedTimeseries import NamedTimeseries, TIME
from tests import _testHelpers as th

import copy
import numpy as np
import pandas as pd
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_ITERATION = 50
TIMESERIES = th.getTimeseries()
FITTER = th.getFitter(cls=mfb.ModelFitterBootstrap)
FITTER.fitModel()
FITTER.bootstrap()
        

class TestBootstrapResult(unittest.TestCase):

    def setUp(self):
        self.fitter = FITTER
        self.speciesNames = list(th.VARIABLE_NAMES)
        self.parameterNames = list(th.PARAMETER_DCT.keys())
        self.parameterDct = {n: np.random.randint(10, 20, NUM_ITERATION)
              for n in self.parameterNames}
        self.fittedStatistic = TimeseriesStatistic(self.fitter.fittedTS)
        for _ in range(NUM_ITERATION):
            ts = self.fitter.fittedTS.copy()
            for name in self.speciesNames:
                arr = 3*np.random.random(len(self.fitter.fittedTS))
                ts[name] += arr
            self.fittedStatistic.accumulate(ts)
        self.bootstrapResult = br.BootstrapResult(self.fitter, NUM_ITERATION,
              self.parameterDct, self.fittedStatistic)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        keys = self.bootstrapResult.parameterStdDct.keys()
        diff = set(keys).symmetric_difference(self.bootstrapResult.parameters)
        self.assertEqual(len(diff), 0)

    def testParams(self):
        if IGNORE_TEST:
            return
        params = self.bootstrapResult.params
        name = self.parameterNames[0]
        self.assertEqual(params.valuesdict()[name],
              np.mean(self.parameterDct[name]))

    def testMerge(self):
        if IGNORE_TEST:
            return
        bootstrapResult = br.BootstrapResult(self.fitter, NUM_ITERATION,
              self.parameterDct, self.fittedStatistic)
        mergedResult = br.BootstrapResult.merge(
              [self.bootstrapResult, bootstrapResult])
        self.assertEqual(mergedResult.numIteration, 2*NUM_ITERATION)
        self.assertEqual(len(mergedResult.parameterDct[self.parameterNames[0]]),
              mergedResult.numIteration)

    def testMeanFittedTS(self):
        if IGNORE_TEST:
            return
        meanFittedTS = self.bootstrapResult.meanBootstrapFittedTS
        overallMean = np.mean(meanFittedTS[self.speciesNames].flatten())
        self.assertGreater(overallMean, 0)

    def testStdFittedTS(self):
        if IGNORE_TEST:
            return
        stdFittedTS = self.bootstrapResult.stdBootstrapFittedTS
        overallStd = np.mean(stdFittedTS[self.speciesNames].flatten())
        self.assertGreater(overallStd, 0)

    def testSimulate(self):
        if IGNORE_TEST:
            return
        statistic = self.bootstrapResult.simulate()
        lowers = statistic.percentileDct[5].flatten()
        uppers = statistic.percentileDct[95].flatten()
        trues = [l <= u for l, u in zip(lowers, uppers)]
        self.assertTrue(all(trues))


if __name__ == '__main__':
    unittest.main()
