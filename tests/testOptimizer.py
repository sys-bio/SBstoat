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
import numpy as np
import lmfit
import unittest


IGNORE_TEST = False
IS_PLOT = False
XKEY = "x"
YKEY = "y"
INITIAL_VALUE = 1
MIN_VALUE = -4
MAX_VALUE = 10
BEST_VALUES = (4, 8)

########## FUNCTIONS #################
def parabola(params:lmfit.Parameters, minArgs:float=BEST_VALUES):
    """
    Implements a function used for optimization with Optimizer.

    Parameters
    ----------
    params: lmfit.Parameters
    
    Returns
    -------
    np.array
        residuals

    Usage
    -----
    residuals = parabola(params)
    """
    xValue = params.valuesdict()[XKEY]
    yValue = params.valuesdict()[YKEY]
    residuals = np.array([(xValue-BEST_VALUES[0])**4, (yValue-BEST_VALUES[1])**4])
    return np.array(residuals)
        

################ TEST CLASSES #############
class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.function = parabola
        self.params = lmfit.Parameters()
        self.params.add(XKEY, value=INITIAL_VALUE, min=MIN_VALUE, max=MAX_VALUE)
        self.params.add(YKEY, value=INITIAL_VALUE, min=MIN_VALUE, max=MAX_VALUE)
        self.methods = Optimizer.mkOptimizerMethod()
        self.optimizer = Optimizer(self.function, self.params, self.methods)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(len(self.optimizer._methods),
              len(self.optimizer._statistics))

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
        optimizer.optimize()
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
              methodNames=[cn.METHOD_DIFFERENTIAL_EVOLUTION])
              #methodNames=[cn.METHOD_LEASTSQ, cn.METHOD_DIFFERENTIAL_EVOLUTION])
        optimizer = Optimizer(self.function, self.params, methods)
        optimizer.optimize()
        self.checkResult()
        

if __name__ == '__main__':
    unittest.main()
