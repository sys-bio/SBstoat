# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

import SBstoat
import SBstoat._constants as cn
from SBstoat.modelFitter import ModelFitter
from SBstoat import suiteFitter as sf
from tests import _testHelpers as th

import matplotlib
import numpy as np
import lmfit
import unittest

try:
    matplotlib.use('TkAgg')
except ImportError:
    pass


IGNORE_TEST = True
IS_PLOT = True
NAME = "parameter"
LOWER = 1
UPPER = 11
VALUE = 5
MODEL_NAMES = ["X", "Y", "Z"]
PARAMETER_NAMES = ["A", "B", "C"]
LOWERS = [10, 20, 30]
UPPERS = [100, 200, 300]
VALUES = [15, 25, 35]
PARAMETERS = [SBstoat.Parameter(n, lower=l, upper=u, value=v)
      for n, l, u, v in zip(PARAMETER_NAMES, LOWERS, UPPERS, VALUES)]
PARAMETERS_COLLECTION = [[PARAMETERS[0]], [PARAMETERS[0], PARAMETERS[2]],
      [PARAMETERS[1]]]
PARAMETERS_COLLECTION = [ModelFitter.mkParameters(c)
      for c in PARAMETERS_COLLECTION]
METHODS = [SBstoat.OptimizerMethod("differential_evolution", {cn.MAX_NFEV: 100})]


################ TEST CLASSES #############
class TestParameter(unittest.TestCase):

    def setUp(self):
        self.parameter = sf._Parameter(NAME, lower=LOWER,
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
            parameter = sf._Parameter(NAME, LOWER, UPPER, VALUE)
            parameter.updateLower(newValue)
            self.assertEqual(parameter.lower, expected)
        #
        test(LOWER-1, LOWER-1)
        test(LOWER+1, LOWER)

    def testUpdateUpper(self):
        if IGNORE_TEST:
            return
        def test(newValue, expected):
            parameter = sf._Parameter(NAME, LOWER, UPPER, VALUE)
            parameter.updateUpper(newValue)
            self.assertEqual(parameter.upper, expected)
        #
        test(UPPER+1, UPPER+1)
        test(UPPER-1, UPPER)


class TestParameterManager(unittest.TestCase):

    def setUp(self):
        self.manager = sf._ParameterManager(MODEL_NAMES,
        PARAMETERS_COLLECTION)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        def test(dct, dtype, keys):
            trues = [isinstance(o, dtype) for o in dct.values()]
            self.assertTrue(all(trues))
            diff = set(keys).symmetric_difference(dct.keys())
            self.assertEqual(len(diff), 0)
        #
        modelNames = list(MODEL_NAMES)
        modelNames.append(sf._ParameterManager.ALL)
        test(self.manager.modelDct, list, modelNames)
        test(self.manager.parameterDct, sf._Parameter, PARAMETER_NAMES)

    def testUpdateValues(self):
        if IGNORE_TEST:
            return
        MULT = 10
        parameters = [SBstoat.Parameter(n, lower=l, upper=u, value=v*MULT)
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
        test(sf._ParameterManager.ALL)
        _ = [test(n) for n in self.manager.modelDct.keys()]


class TestSuiteFitter(unittest.TestCase):

    def setUp(self):
        self.modelSpecifications = [th.ANTIMONY_MODEL, th.ANTIMONY_MODEL,
              th.ANTIMONY_MODEL]
        self.datasets = [th.TEST_DATA_PATH, th.TEST_DATA_PATH,
              th.TEST_DATA_PATH]
        self.parameterNames = list(th.PARAMETER_DCT.keys())
        self.parametersCollection = [self.parameterNames[:-1],
              self.parameterNames[1:], [self.parameterNames[-2]]]
        self.fitter = sf.SuiteFitter(self.modelSpecifications, self.datasets,
              self.parametersCollection, modelNames=MODEL_NAMES,
              fitterMethods=METHODS)

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        parameterNames = []
        for modelName in MODEL_NAMES:
            parameterNames.extend(self.fitter.parameterManager.modelDct[modelName])
        diff = set(parameterNames).symmetric_difference(
            self.fitter.parameterManager.modelDct[sf._ParameterManager.ALL])
        self.assertEqual(len(diff), 0)

    def testConstructor2(self):
        if IGNORE_TEST:
            return
        with self.assertRaises(ValueError):
            self.fitter = sf.SuiteFitter(self.modelSpecifications,
                  self.datasets[0],
                  self.parametersCollection, modelNames=MODEL_NAMES)

    def testCalcResiduals(self):
        if IGNORE_TEST:
            return
        parameters = self.fitter.parameterManager.mkParameters()
        residuals = self.fitter.calcResiduals(parameters)
        expectedSize = th.NUM_POINT*len(MODEL_NAMES)*len(th.VARIABLE_NAMES)
        self.assertTrue(np.isclose(expectedSize, np.size(residuals)))

    def testFitSuite(self):
        # TESTING
        self.fitter.fitSuite()


if __name__ == '__main__':
    unittest.main()
