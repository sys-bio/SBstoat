# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

import SBstoat
import SBstoat._constants as cn
from SBstoat.modelFitter import ModelFitter
from SBstoat._suiteFitterCore import _Parameter, _ParameterManager, \
      ResidualsServer, SuiteFitterCore

from tests import _testHelpers as th

import matplotlib
import numpy as np
import lmfit
import unittest


IGNORE_TEST = False
IS_PLOT = False
NAME = "parameter"
LOWER = 1
UPPER = 11
VALUE = 5
MODEL_NAMES = ["W", "X", "Y", "Z"]
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
METHODS = [SBstoat.OptimizerMethod("leastsq", {cn.MAX_NFEV: 100})]



def mkRepeatedList(list, repeat):
    return [list for _ in range(repeat)]


################ TEST CLASSES #############
class TestParameter(unittest.TestCase):

    def setUp(self):
        self.parameter = _Parameter(NAME, lower=LOWER,
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
            parameter = _Parameter(NAME, LOWER, UPPER, VALUE)
            parameter.updateLower(newValue)
            self.assertEqual(parameter.lower, expected)
        #
        test(LOWER-1, LOWER-1)
        test(LOWER+1, LOWER)

    def testUpdateUpper(self):
        if IGNORE_TEST:
            return
        def test(newValue, expected):
            parameter = _Parameter(NAME, LOWER, UPPER, VALUE)
            parameter.updateUpper(newValue)
            self.assertEqual(parameter.upper, expected)
        #
        test(UPPER+1, UPPER+1)
        test(UPPER-1, UPPER)


class TestParameterManager(unittest.TestCase):

    def setUp(self):
        self.numModel = 3
        self.manager = _ParameterManager(MODEL_NAMES[:self.numModel],
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
        modelNames.append(_ParameterManager.ALL)
        test(self.manager.modelDct, list, modelNames)
        test(self.manager.parameterDct, _Parameter, PARAMETER_NAMES)

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
        test(_ParameterManager.ALL)
        _ = [test(n) for n in self.manager.modelDct.keys()]


class TestResidualsServer(unittest.TestCase):

    def setUp(self):
        self.fitter = th.getFitter(cls=ModelFitter)
        self.server = ResidualsServer(self.fitter, None, None)

    def testRunFunction(self):
        if IGNORE_TEST:
            return
        residuals = self.server.runFunction(self.fitter.params)
        self.assertTrue(isinstance(residuals, np.ndarray))


class TestSuiteFitterCore(unittest.TestCase):

    def _init(self, numModel=3):
        self.numModel = numModel
        self.modelNames = MODEL_NAMES[0:numModel]
        self.modelSpecifications = mkRepeatedList(th.ANTIMONY_MODEL, numModel)
        self.datasets = mkRepeatedList(th.TEST_DATA_PATH, numModel)
        self.parameterNames = list(th.PARAMETER_DCT.keys())
        self.parameterNamesCollection = mkRepeatedList(self.parameterNames,
              numModel)
        self.fitter = SBstoat.mkSuiteFitter(self.modelSpecifications,
              self.datasets,
              self.parameterNamesCollection,
              modelNames=self.modelNames,
              fitterMethods=METHODS)

    def tearDown(self):
        if "fitter" in self.__dict__.keys():
            self.fitter.clean()

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        self._init()
        parameterNames = []
        for modelName in self.modelNames:
            parameterNames.extend(self.fitter.parameterManager.modelDct[modelName])
        diff = set(parameterNames).symmetric_difference(
            self.fitter.parameterManager.modelDct[_ParameterManager.ALL])
        self.assertEqual(len(diff), 0)

    def testConstructor2(self):
        if IGNORE_TEST:
            return
        self._init()
        with self.assertRaises(ValueError):
            self.fitter = SBstoat.mkSuiteFitter(self.modelSpecifications,
                  self.datasets[0],
                  self.parameterNamesCollection, modelNames=MODEL_NAMES)
        self.fitter.clean()

    def testCalcResiduals(self):
        if IGNORE_TEST:
            return
        self._init()
        parameters = self.fitter.parameterManager.mkParameters()
        residuals = self.fitter.calcResiduals(parameters)
        expectedSize = th.NUM_POINT*self.numModel*len(th.VARIABLE_NAMES)
        self.assertTrue(np.isclose(expectedSize, np.size(residuals)))

    def testFitSuite(self):
        if IGNORE_TEST:
            return
        self._init(numModel=1)
        fitter = ModelFitter(self.modelSpecifications[0],
              self.datasets[0],
              parametersToFit=self.parameterNamesCollection[0])
        fitter.fitModel()
        self.fitter.fitSuite()
        valuesDct1 = fitter.params.valuesdict()
        valuesDct2 = self.fitter.params.valuesdict()
        for name, value in valuesDct1.items():
            self.assertLess(np.abs(valuesDct2[name] - value), 0.5)

    def testReportFit(self):
        if IGNORE_TEST:
            return
        self._init(numModel=3)
        self.fitter.fitSuite()
        result = self.fitter.reportFit()
        for name in self.fitter.params.valuesdict().keys():
            self.assertTrue(name in result)

    def testPlotResidualsSSQ(self, **kwargs):
        if IGNORE_TEST:
            return
        # Smoke test
        self._init(numModel=3)
        self.fitter.fitSuite()
        self.fitter.plotResidualsSSQ(isPlot=IS_PLOT)
        

if __name__ == '__main__':
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        pass
    unittest.main()
