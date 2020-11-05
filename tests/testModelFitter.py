# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:24:09 2020

@author: hsauro
@author: joseph-hellerstein
"""

from SBstoat.modelFitter import ModelFitter
from tests import _testHelpers as th

import matplotlib
import os
import unittest


IGNORE_TEST = False
IS_PLOT = False
if not IGNORE_TEST:
    TIMESERIES = th.getTimeseries()
    FITTER = th.getFitter(cls=ModelFitter, isPlot=IS_PLOT)
    FITTER.fitModel()
    FITTER.bootstrap(numIteration=10)
ANTIMONY_MODEL_BENCHMARK = """
# Reactions   
    J1: S1 -> S2; k1*S1
    J2: S2 -> S3; k2*S2
# Species initializations     
    S1 = 10; S2 = 0;
    k1 = 1; k2 = 2
"""
DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(DIR, "groundtruth_2_step_0_1.txt")
BENCHMARK1_TIME = 30  # Actual is 20 sec
        

class TestModelFitter(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.timeseries = TIMESERIES
        self.fitter = FITTER

    def testPlotResiduals(self):
        if IGNORE_TEST:
            return
        self.fitter.plotResiduals(numCol=3, numRow=2, ylim=[-1.5, 1.5])

    def testPlotFitAll(self):
        if IGNORE_TEST:
            return
        self.fitter.plotFitAll(isMultiple=True, numPoint=3,
              params=self.fitter.params)
        self.fitter.plotFitAll()
        self.fitter.plotFitAll(isMultiple=True)

    def testPlotParameterEstimates(self):
        if IGNORE_TEST:
            return
        self.fitter.plotParameterEstimatePairs(ylim=[0, 5], xlim=[0, 5],
                                               markersize=5)

    def testPlotParameterHistograms(self):
        if IGNORE_TEST:
            return
        self.fitter.plotParameterHistograms(ylim=[0, 5],
                                            xlim=[0, 6],
                                            titlePosition=[.3, .9],
                                            bins=10,
                                            parameters=["k1", "k2"])

    def testPlotPlotResidualsAll(self):
        if IGNORE_TEST:
            return
        self.fitter.plotResidualsAll()

    def testBootstrapAccuracy(self):
        if IGNORE_TEST:
            return
        model = """
            J1: S1 -> S2; k1*S1
            J2: S2 -> S3; k2*S2
           
            S1 = 1; S2 = 0; S3 = 0;
            k1 = 0; k2 = 0; 
        """
        columns = ["S1", "S3"]
        fitter = ModelFitter(model, BENCHMARK_PATH, ["k1", "k2"],
                             selectedColumns=columns, isPlot=IS_PLOT)
        fitter.fitModel()
        print(fitter.reportFit())
        print (fitter.getParameterMeans())  
        
        fitter.bootstrap(numIteration=2000,
              reportInterval=500)
              #calcObservedFunc=ModelFitter.calcObservedTSNormal, std=0.01)
        fitter.plotParameterEstimatePairs(['k1', 'k2'],
              markersize=2)
        print("Mean: %s" % str(fitter.getParameterMeans()))
        print("Std: %s" % str(fitter.getParameterStds()))
        fitter.reportBootstrap()

    def testPlot(self):
        if IGNORE_TEST:
            return
        with self.assertRaises(ValueError):
            self.fitter.plot("dummy")
        pass
        #
        self.fitter.plot("FitAll")
        self.fitter.plot('ResidualsHistograms')


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    unittest.main()
