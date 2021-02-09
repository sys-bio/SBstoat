# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

import SBstoat._constants as cn
from SBstoat._optimizer import Optimizer
from SBstoat import _helpers
from SBstoat.logs import Logger, LEVEL_MAX

import numpy as np
import lmfit
import unittest


IGNORE_TEST = False
IS_PLOT = False
KEY = "x"
INITIAL_VALUE = 1
MIN_VALUE = -4
MAX_VALUE = 10
BEST_VALUE = 4

########## FUNCTIONS #################
def parabola(params:lmfit.Parameters, minArg:float=BEST_VALUE):
    xValue = params.valuesdict()[KEY]
    return (xValue-4)**2
        

################ TEST CLASSES #############
class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.function = parabola
        self.params = lmfit.Parameters()
        self.params.add(KEY, value=INITIAL_VALUE, min=MIN_VALUE, max=MAX_VALUE)
        self.optimizerMethods = Optimizer.mkOptimizerMethod()

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

    def testOptimize(self):
:

        

if __name__ == '__main__':
    unittest.main()
