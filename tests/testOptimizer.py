# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

import SBstoat._constants as cn
from SBstoat._optimizer import Optimizer
from SBstoat import _helpers
from SBstoat.logs import Logger, LEVEL_MAX

import collections
import matplotlib
import numpy as np
import lmfit
import unittest

try:
    matplotlib.use('TkAgg')
except ImportError:
    pass


IGNORE_TEST = False
IS_PLOT = False
XKEY = "x"
YKEY = "y"
INITIAL_VALUE = 1
MIN_VALUE = -4
MAX_VALUE = 10
BEST_DCT = {XKEY: 4, YKEY:8}
BEST_VALUES = list(BEST_DCT.values())

########## FUNCTIONS #################
def parabola(params:lmfit.Parameters, minArgs:float=BEST_VALUES,
      isRawData=False):
    """
    Implements a function used for optimization with Optimizer.

    Parameters
    ----------
    params: lmfit.Parameters
    minArgs: tupe-float
    isRawData: bool
        Return raw data as baseline
    
    Returns
    -------
    np.array
        residuals

    Usage
    -----
    residuals = parabola(params)
    """
    if isRawData:
        return np.array([MAX_VALUE, MAX_VALUE])
    xValue = params.valuesdict()[XKEY]
    yValue = params.valuesdict()[YKEY]
    residuals = np.array([(xValue-BEST_VALUES[0])**4, (yValue-BEST_VALUES[1])**4])
    return np.array(residuals)


def parabolaWithoutRaw(params:lmfit.Parameters, minArgs:float=BEST_VALUES):
    return parabola(params, minArgs=minArgs)
        

################ TEST CLASSES #############
class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.function = parabola
        self.params = lmfit.Parameters()
        self.params.add(XKEY, value=INITIAL_VALUE, min=MIN_VALUE, max=MAX_VALUE)
        self.params.add(YKEY, value=INITIAL_VALUE, min=MIN_VALUE, max=MAX_VALUE)
        self.methods = Optimizer.mkOptimizerMethod()
        self.optimizer = Optimizer(self.function, self.params, self.methods,
              isCollect=False)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual( len(self.optimizer.performanceStats), 0)

    def testMkOptimizerMethod(self):
        if IGNORE_TEST:
            return
        def test(results):
            for result in results:
                self.assertTrue(isinstance(result.method, str))
                self.assertTrue(isinstance(result.kwargs, dict))
                self.assertTrue(cn.MAX_NFEV in result.kwargs.keys())
        #
        test(Optimizer.mkOptimizerMethod())
        test(Optimizer.mkOptimizerMethod(methodNames="aa"))
        test(Optimizer.mkOptimizerMethod(methodNames=["aa", "bb"]))
        test(Optimizer.mkOptimizerMethod(methodNames=["aa", "bb"],
              methodKwargs={cn.MAX_NFEV: 10}))

    def checkResult(self, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        optimizer.execute()
        values = optimizer.params.valuesdict().values()
        for expected, actual in zip(BEST_VALUES, values):
            self.assertLess(np.abs(expected-actual), 0.01)

    def testOptimize1(self):
        if IGNORE_TEST:
            return
        self.checkResult()

    def testOptimize2(self):
        if IGNORE_TEST:
            return
        methods = Optimizer.mkOptimizerMethod(
              methodNames=[cn.METHOD_LEASTSQ, cn.METHOD_DIFFERENTIAL_EVOLUTION])
        for function in [parabola, parabolaWithoutRaw]:
            optimizer = Optimizer(self.function, self.params, methods,
                  isCollect=True)
            optimizer.execute()
            self.checkResult()
            for idx in range(len(optimizer.performanceStats)):
                self.assertGreater(len(optimizer.performanceStats[idx]), 100)

    def testPlotPerformance(self):
        if IGNORE_TEST:
            return
        methods = Optimizer.mkOptimizerMethod(
              methodNames=[cn.METHOD_LEASTSQ, cn.METHOD_DIFFERENTIAL_EVOLUTION])
        optimizer = Optimizer(self.function, self.params, methods,
              isCollect=True)
        optimizer.execute()
        optimizer.plotPerformance(isPlot=IS_PLOT)

    def testPlotQuality(self):
        if IGNORE_TEST:
            return
        methods = Optimizer.mkOptimizerMethod(
              methodNames=[cn.METHOD_DIFFERENTIAL_EVOLUTION, cn.METHOD_LEASTSQ])
              #methodNames=[cn.METHOD_LEASTSQ, cn.METHOD_DIFFERENTIAL_EVOLUTION])
        optimizer = Optimizer(self.function, self.params, methods,
              isCollect=True)
        optimizer.execute()
        optimizer.plotQuality(isPlot=IS_PLOT)

    def testSetRandomValue(self):
        if IGNORE_TEST:
            return
        def test1(params):
            for _, parameter in params.items():
                self.assertLessEqual(parameter.min, parameter.value)
                self.assertLessEqual(parameter.value, parameter.max)
        #
        def test2(param1s, param2s):
            valueDct1 = param1s.valuesdict()
            valueDct2 = param2s.valuesdict()
            for name, value in valueDct1.items():
                self.assertFalse(np.isclose(value, valueDct2[name]))
        #
        newParams = Optimizer._setRandomValue(self.params)
        test1(newParams)
        #
        newerParams = Optimizer._setRandomValue(self.params)
        test1(newerParams)
        test2(newerParams, newParams)

    def testOptimize(self):
        if IGNORE_TEST:
            return
        methods = Optimizer.mkOptimizerMethod(maxFev=10)
        optimizer0 = Optimizer.optimize(self.function, self.params, methods,
              isCollect=False, numRestart=0)
        optimizer100 = Optimizer.optimize(self.function, self.params, methods,
              isCollect=False, numRestart=100)
        #
        valuesDct0 = optimizer0.params.valuesdict()
        valuesDct100 = optimizer100.params.valuesdict()
        for name, value in valuesDct100.items():
            diff0 = (BEST_DCT[name] - valuesDct0[name])**2
            diff100 = (BEST_DCT[name] - value)**2
            self.assertLess(diff100, diff0)
        

if __name__ == '__main__':
    unittest.main()
