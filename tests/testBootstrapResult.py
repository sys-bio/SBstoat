# -*- coding: utf-8 -*-
"""
Created on Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

from SBstoat import _bootstrapResult as br
from SBstoat import _modelFitterBootstrap as mfb
from SBstoat.namedTimeseries import NamedTimeseries, TIME
from tests import _testHelpers as th

import copy
import numpy as np
import pandas as pd
import unittest


IGNORE_TEST = False
NUM_ITERATION = 50
MEAN_UNIFORM = 0.5  # Mean of uniform distribution
STD_UNIFORM = np.sqrt(1.0/12)  # Standard deviation of uniform
TIMESERIES = th.getTimeseries()
FITTER = th.getFitter(cls=mfb.ModelFitterBootstrap)
FITTER.fitModel()
        

class TestBootstrapResult(unittest.TestCase):

    def setUp(self):
        self.fitter = FITTER
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
              br.COL_SUM: self.sumTS,
              br.COL_SSQ: self.ssqTS,
              }
        self.bootstrapResult = br.BootstrapResult(self.fitter, NUM_ITERATION,
              self.parameterDct, self.statisticDct)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        diff = set(self.parameterDct.keys()).symmetric_difference(
              self.bootstrapResult.parameters)
        self.assertEqual(len(diff), 0)
        #
        diff = set(self.bootstrapResult.statisticDct.keys()).symmetric_difference(
              [br.COL_SSQ, br.COL_SUM])
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
        bootstrapResult = br.BootstrapResult(self.fitter, NUM_ITERATION,
              self.parameterDct, self.statisticDct)
        mergedResult = br.BootstrapResult.merge(
              [self.bootstrapResult, bootstrapResult])
        self.assertEqual(mergedResult.numIteration, 2*NUM_ITERATION)
        self.assertEqual(len(mergedResult.parameterDct[self.names[0]]),
              mergedResult.numIteration)

    def testMeanFittedTS(self):
        if IGNORE_TEST:
            return
        meanFittedTS = self.bootstrapResult.meanBootstrapFittedTS
        overallMean = np.mean(meanFittedTS[self.names].flatten())
        self.assertLess(np.abs(overallMean - MEAN_UNIFORM), 0.1)

    def testStdFittedTS(self):
        if IGNORE_TEST:
            return
        stdFittedTS = self.bootstrapResult.stdBootstrapFittedTS
        overallStd = np.mean(stdFittedTS[self.names].flatten())
        self.assertLess(np.abs(overallStd - STD_UNIFORM), 0.01)


if __name__ == '__main__':
    unittest.main()
