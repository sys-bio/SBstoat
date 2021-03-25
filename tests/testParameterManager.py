# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

from SBstoat import Parameter
from SBstoat.modelFitter import ModelFitter
import SBstoat._constants as cn
from SBstoat._parameterManager import ParameterManager
from tests import _testHelpers as th

import matplotlib
import numpy as np
import lmfit
import unittest


IGNORE_TEST = True
IS_PLOT = True
NAME = "parameter"
LOWER = 1
UPPER = 11
VALUE = 5
MODEL_NAMES = ["W", "X", "Y", "Z"]
PARAMETER_NAMES = ["A", "B", "C"]
LOWERS = [10, 20, 30]
UPPERS = [100, 200, 300]
VALUES = [15, 25, 35]
PARAMETERS = [Parameter(n, lower=l, upper=u, value=v)
      for n, l, u, v in zip(PARAMETER_NAMES, LOWERS, UPPERS, VALUES)]
PARAMETERS_COLLECTION = [[PARAMETERS[0]], [PARAMETERS[0], PARAMETERS[2]],
      [PARAMETERS[1]]]
PARAMETERS_COLLECTION = [ModelFitter.mkParameters(c)
      for c in PARAMETERS_COLLECTION]


def mkRepeatedList(list, repeat):
    return [list for _ in range(repeat)]


################ TEST CLASSES #############
class TestParameter(unittest.TestCase):

    def setUp(self):
        self.parameter = Parameter(NAME, lower=LOWER,
              upper=UPPER, value=VALUE)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.parameter.name, NAME)
        self.assertEqual(self.parameter.value, VALUE)
        self.assertEqual(self.parameter.lower, LOWER)
        self.assertEqual(self.parameter.upper, UPPER)

    def testUpdateLower(self):
        if IGNORE_TEST:
            return
        def test(newValue, expected):
            parameter = Parameter(NAME, LOWER, UPPER, VALUE)
            parameter.updateLower(newValue)
            self.assertEqual(parameter.lower, expected)
        #
        test(LOWER-1, LOWER-1)
        test(LOWER+1, LOWER)

    def testUpdateUpper(self):
        # TESTING
        def test(newValue, expected):
            parameter = Parameter(NAME, lower=LOWER, upper=UPPER, value=VALUE)
            parameter.updateUpper(newValue)
            self.assertEqual(parameter.upper, expected)
        #
        test(UPPER-1, UPPER)
        test(UPPER+1, UPPER+1)


class TestParameterManager(unittest.TestCase):

    def setUp(self):
        self.numModel = 3
        self.manager = ParameterManager(MODEL_NAMES[:self.numModel],
        PARAMETERS_COLLECTION[:self.numModel])

    def testConstructor(self):
        if IGNORE_TEST:
            return
        def test(dct, dtype, keys):
            trues = [isinstance(o, dtype) for o in dct.values()]
            self.assertTrue(all(trues))
            diff = set(keys).symmetric_difference(dct.keys())
            self.assertEqual(len(diff), 0)
        #
        modelNames = list(MODEL_NAMES[:self.numModel])
        modelNames.append(ParameterManager.ALL)
        test(self.manager.modelDct, list, modelNames)
        test(self.manager.parameterDct, Parameter, PARAMETER_NAMES)

    def testUpdateValues(self):
        if IGNORE_TEST:
            return
        MULT = 10
        parameters = [Parameter(n, lower=l, upper=u, value=v*MULT)
              for n, l, u, v in zip(PARAMETER_NAMES, LOWERS, UPPERS, VALUES)]
        parametersCollection = [[parameters[0]], [parameters[0], parameters[2]],
              [parameters[1]]]
        parametersCollection = [ModelFitter.mkParameters(c)
              for c in parametersCollection]
        for oldParameters, newParameters in zip(PARAMETERS_COLLECTION,
              parametersCollection):
            self.manager.updateValues(newParameters)
            oldValuesDct = oldParameters.valuesdict()
            newValuesDct = newParameters.valuesdict()
            trues = [oldValuesDct[k]*MULT == newValuesDct[k]
                  for k in newValuesDct.keys()]
            self.assertTrue(all(trues))

    def testMkParameters(self):
        if IGNORE_TEST:
            return
        def test(modelName):
            lmfitParameters = self.manager.mkParameters(modelName)
            self.assertEqual(len(lmfitParameters.valuesdict()),
                  len(self.manager.modelDct[modelName]))
        #
        test(ParameterManager.ALL)
        _ = [test(n) for n in self.manager.modelDct.keys()]
        

if __name__ == '__main__':
    unittest.main()
