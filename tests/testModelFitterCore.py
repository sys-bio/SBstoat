# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

import SBstoat._modelFitterCore as mf
from SBstoat.modelFitter import ModelFitter
from SBstoat._modelFitterCore import ModelFitterCore
from SBstoat.namedTimeseries import NamedTimeseries, TIME
from tests import _testHelpers as th

import copy
import numpy as np
import os
import tellurium
import unittest


IGNORE_TEST = False
TIMESERIES = th.getTimeseries()
DIR = os.path.dirname(os.path.abspath(__file__))
FILE_SERIALIZE = os.path.join(DIR, "modelFitterCore.pcl")
FILES = [FILE_SERIALIZE]
        

class TestModelFitterCore(unittest.TestCase):

    def setUp(self):
        self._remove()
        self.timeseries = copy.deepcopy(TIMESERIES)
        self.fitter = th.getFitter(cls=ModelFitterCore)
    
    def tearDown(self):
        self._remove()

    def _remove(self):
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsNone(self.fitter.roadrunnerModel)
        self.assertGreater(len(self.fitter.observedTS), 0)
        #
        for variable in self.fitter.selectedColumns:
            self.assertTrue(variable in th.VARIABLE_NAMES)

    def testCopy(self):
        if IGNORE_TEST:
            return
        newFitter = self.fitter.copy()
        self.assertTrue(isinstance(newFitter.modelSpecification, str))
        self.assertTrue(isinstance(newFitter, ModelFitterCore))

    def testSimulate(self):
        if IGNORE_TEST:
            return
        self.fitter._initializeRoadrunnerModel()
        self.fitter._simulate()
        self.assertTrue(self.fitter.observedTS.isEqualShape(
              self.fitter.fittedTS))

    def testResiduals(self):
        if IGNORE_TEST:
            return
        self.fitter._initializeRoadrunnerModel()
        arr = self.fitter._residuals(None)
        self.assertTrue(self.fitter.observedTS.isEqualShape(
              self.fitter.residualsTS))
        self.assertEqual(len(arr),
              len(self.fitter.observedTS)*len(self.fitter.observedTS.colnames))

    def checkParameterValues(self):
        dct = self.fitter.params.valuesdict()
        self.assertEqual(len(dct), len(self.fitter.parametersToFit))
        #
        for value in dct.values():
            self.assertTrue(isinstance(value, float))
        return dct

    def testInitializeParams(self):
        if IGNORE_TEST:
            return
        LOWER = -10
        UPPER = -1
        VALUE = -5
        NEW_SPECIFICATION = mf.ParameterSpecification(
              lower=LOWER,
              upper=UPPER,
              value=VALUE)
        DEFAULT_SPECIFICATION = mf.ParameterSpecification(
              lower=mf.PARAMETER_LOWER_BOUND,
              upper=mf.PARAMETER_UPPER_BOUND,
              value=(mf.PARAMETER_LOWER_BOUND+mf.PARAMETER_UPPER_BOUND)/2,
              )
        def test(params, exceptions=[]):
            def check(parameter, specification):
                self.assertEqual(parameter.min, specification.lower)
                self.assertEqual(parameter.max, specification.upper)
                self.assertEqual(parameter.value, specification.value)
            #
            names = params.valuesdict().keys()
            for name in names:
                parameter = params.get(name)
                if name in exceptions:
                    check(parameter, NEW_SPECIFICATION)
                else:
                    check(parameter, DEFAULT_SPECIFICATION)
        #
        fitter = ModelFitterCore(
              self.fitter.modelSpecification,
              self.fitter.observedTS,
              self.fitter.parametersToFit,
              parameterDct={"k1": NEW_SPECIFICATION},
              )
        params = fitter._initializeParams()
        test(params, exceptions=["k1"])
        #
        params = self.fitter._initializeParams()
        test(params, [])

    def testFit1(self):
        if IGNORE_TEST:
            return
        def test(method):
            fitter = ModelFitterCore(th.ANTIMONY_MODEL, self.timeseries,
                  list(th.PARAMETER_DCT.keys()), method=method)
            fitter.fitModel()
            PARAMETER = "k2"
            diff = np.abs(th.PARAMETER_DCT[PARAMETER]
                  - dct[PARAMETER])
            self.assertLess(diff, 1)
        #
        self.fitter.fitModel()
        dct = self.checkParameterValues()
        #
        for method in [mf.METHOD_LEASTSQ, mf.METHOD_BOTH,
              mf.METHOD_DIFFERENTIAL_EVOLUTION]:
            test(method)

    def testFit2(self):
        if IGNORE_TEST:
            return
        def calcResidualStd(selectedColumns):
            columns = self.timeseries.colnames[:3]
            fitter = ModelFitterCore(th.ANTIMONY_MODEL, self.timeseries,
                  list(th.PARAMETER_DCT.keys()), selectedColumns=selectedColumns)
            fitter.fitModel()
            return np.std(fitter.residualsTS.flatten())
        #
        CASES = [th.COLUMNS[0], th.COLUMNS[:3], th.COLUMNS]
        stds = [calcResidualStd(c) for c in CASES]
        # Variance should decrease with more columns
        self.assertGreater(stds[0], stds[1])
        self.assertGreater(stds[1], stds[2])

    def testFitDataTransformDct(self):
        if IGNORE_TEST:
            return
        def test(col, func, maxDifference=0.0):
            timeseries = self.timeseries.copy()
            timeseries[col] = func(timeseries)
            fittedDataTransformDct = {col: func}
            fitter = ModelFitterCore(th.ANTIMONY_MODEL, timeseries,
                  list(th.PARAMETER_DCT.keys()),
                  fittedDataTransformDct=fittedDataTransformDct)
            fitter.fitModel()
            for name in self.fitter.params.valuesdict().keys():
                value1 = self.fitter.params.valuesdict()[name]
                value2 = fitter.params.valuesdict()[name]
                diff = np.abs(value1-value2)
                self.assertLessEqual(diff, maxDifference)
        #
        self.fitter.fitModel()
        col = "S1"
        #
        func2 = lambda t: 2*t[col]
        test(col, func2, maxDifference=0.3)
        #
        func1 = lambda t: t[col]
        test(col, func1)

    def testGetFittedModel(self):
        if IGNORE_TEST:
            return
        fitter1 = ModelFitterCore(th.ANTIMONY_MODEL, self.timeseries,
              list(th.PARAMETER_DCT.keys()))
        fitter1.fitModel()
        fittedModel = fitter1.getFittedModel()
        fitter2 = ModelFitterCore(fittedModel, self.timeseries,
              list(th.PARAMETER_DCT.keys()))
        fitter2.fitModel()
        # Should get same fit without changing the parameters
        self.assertTrue(np.isclose(np.var(fitter1.residualsTS.flatten()),
              np.var(fitter2.residualsTS.flatten())))

    def testSerialize(self):
        if IGNORE_TEST:
            return
        self.fitter.fitModel()
        self.assertFalse(os.path.isfile(FILE_SERIALIZE))
        self.fitter.serialize(FILE_SERIALIZE)
        self.assertTrue(os.path.isfile(FILE_SERIALIZE))

    def testSerialize(self):
        if IGNORE_TEST:
            return
        self.fitter.fitModel()
        self.fitter.serialize(FILE_SERIALIZE)
        fitter = ModelFitter.deserialize(FILE_SERIALIZE)
        self.assertEqual(fitter.modelSpecification,
              self.fitter.modelSpecification)
        

if __name__ == '__main__':
    unittest.main()
