# -*- coding: utf-8 -*-
"""
Created on Aug 20, 2020

@author: joseph-hellerstein
"""

from SBstoat.modelFitter import ModelFitter
import SBstoat._bootstrapResult as bsr
from SBstoat.residualsAnalyzer import ResidualsAnalyzer
import tests._testHelpers as th

import matplotlib
import os
import unittest


IGNORE_TEST = False
IS_PLOT = False
OBSERVED_TS, FITTED_TS = th.getObservedFitted()
FITTER = th.getFitter(cls=ModelFitter)
FITTER.fitModel()
FITTER.bootstrap(numIteration=10)
LOW_PERCENTILE = bsr.PERCENTILES[0]
HIGH_PERCENTILE = bsr.PERCENTILES[-1]
        

class TestReidualAnalyzer(unittest.TestCase):

    def setUp(self):
        self.observedTS = OBSERVED_TS
        self.fittedTS = FITTED_TS
        self.fitter = FITTER
        self.statistic = self.fitter.bootstrapResult.simulate(numPoint=30,
              numSample=300)
        self.analyzer = ResidualsAnalyzer(self.observedTS, self.fittedTS,
              isPlot=IS_PLOT)

    def testPlotResidualsOverTime(self):
        if IGNORE_TEST:
            return
        self.analyzer.plotResidualsOverTime(numCol=3, numRow=2,
              ylim=[-1.5, 1.5])

    def testPlotFittedObservedOverTime1(self):
        if IGNORE_TEST:
            return
        analyzer = ResidualsAnalyzer(self.observedTS, self.statistic.meanTS,
              bandLowTS=self.statistic.percentileDct[LOW_PERCENTILE],
              bandHighTS=self.statistic.percentileDct[HIGH_PERCENTILE],
              isPlot=IS_PLOT)
        analyzer.plotFittedObservedOverTime(numCol=3, numRow=2)
        #
        self.analyzer.plotFittedObservedOverTime(numCol=3, numRow=2)

    def testPlotFittedObservedOverTime2(self):
        if IGNORE_TEST:
            return
        meanTS = self.fittedTS.copy(isInitialize=True)
        stdTS = self.fittedTS.copy(isInitialize=True)
        for col in meanTS.colnames:
            meanTS[col] = 2
            stdTS[col] = 1
        analyzer = ResidualsAnalyzer(self.observedTS, self.fittedTS,
              meanFittedTS=meanTS, stdFittedTS=stdTS,
              isPlot=IS_PLOT)
        analyzer.plotFittedObservedOverTime(numCol=3, numRow=2)

    def testPlotResidualsHistograms(self):
        if IGNORE_TEST:
            return
        self.analyzer.plotResidualsHistograms(ylim=[0, 5],
              xlim=[0, 6], titlePosition=[.3, .9],
              bins=10)

    def testPlotAll(self):
        if IGNORE_TEST:
            return
        self.analyzer.plotAll()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    unittest.main()
