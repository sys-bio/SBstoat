# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

import SBstoat._modelFitterCore as mf
from SBstoat.modelFitter import ModelFitter
from SBstoat.logs import Logger, LEVEL_MAX
from SBstoat._modelFitterCore import ModelFitterCore
from SBstoat.namedTimeseries import NamedTimeseries, TIME
import tellurium as te
from tests import _testHelpers as th

import copy
import numpy as np
import os
import tellurium
import unittest


IGNORE_TEST = False
IS_PLOT = False
TIMESERIES = th.getTimeseries()
DIR = os.path.dirname(os.path.abspath(__file__))
FILE_SERIALIZE = os.path.join(DIR, "modelFitterCore.pcl")
FILES = [FILE_SERIALIZE]
WOLF_MODEL = os.path.join(DIR, "Jana_WolfGlycolysis.antimony")
WOLF_DATA = os.path.join(DIR, "wolf_data.csv")
        

class TestModelFitterCore(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self._init()

    def _init(self):
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

    def testrpConstruct(self):
        if IGNORE_TEST:
            return
        fitter = ModelFitterCore.rpConstruct()
        def updateAttr(attr):
            if not attr in fitter.__dict__.keys():
                fitter.__setattr__(attr, None)
        #
        updateAttr("roadrunnerModel")
        updateAttr("observedTS")
        self.assertIsNone(self.fitter.roadrunnerModel)
        self.assertIsNone(fitter.observedTS)

    def testCopy(self):
        if IGNORE_TEST:
            return
        newFitter = self.fitter.copy()
        self.assertTrue(isinstance(newFitter.modelSpecification, str))
        self.assertTrue(isinstance(newFitter, ModelFitterCore))

    def testResiduals(self):
        if IGNORE_TEST:
            return
        self.fitter._initializeRoadrunnerModel()
        params = self.fitter.mkParams()
        arr = self.fitter._residuals(params)
        length = len(self.fitter.observedTS.flatten())
        self.assertEqual(len(arr), length)

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
              #self.fitter.parametersToFit,
              parameterDct={"k1": NEW_SPECIFICATION},
              )
        params = fitter.mkParams()
        test(params, exceptions=["k1"])
        #
        params = self.fitter.mkParams()
        test(params, [])
        #
        fitter = ModelFitterCore(
              self.fitter.modelSpecification,
              self.fitter.observedTS,
              parameterDct={"k1": (LOWER, UPPER, VALUE)},
              )
        params = fitter.mkParams()
        test(params, exceptions=["k1"])

    def testFit1(self):
        if IGNORE_TEST:
            return
        self._init()
        def test(method):
            fitter = ModelFitterCore(th.ANTIMONY_MODEL, self.timeseries,
                  list(th.PARAMETER_DCT.keys()), fitterMethod=method)
            fitter.fitModel()
            for parameter in ["k1", "k2", "k3", "k4", "k5"]:
                diff = np.abs(th.PARAMETER_DCT[parameter]
                      - dct[parameter])
                frac = diff/dct[parameter]
                self.assertLess(diff/dct[parameter], 1.0)
        #
        self.fitter.fitModel()
        dct = self.checkParameterValues()
        #
        for method in [mf.METHOD_LEASTSQ, mf.METHOD_BOTH,
              mf.METHOD_DIFFERENTIAL_EVOLUTION]:
            test(method)

    def testFitNanValues(self):
        if IGNORE_TEST:
            return
        PARAMETER = "k2"
        def calc(method, probNan=0.2):
            nanTimeseries = self.timeseries.copy()
            for col in self.timeseries.colnames:
                for idx in range(len(nanTimeseries)):
                    if np.random.random() <= probNan:
                        nanTimeseries[col][idx] = np.nan
            fitter = ModelFitterCore(th.ANTIMONY_MODEL, nanTimeseries,
                  list(th.PARAMETER_DCT.keys()), fitterMethod=method)
            fitter.fitModel()
            diff = np.abs(th.PARAMETER_DCT[PARAMETER]
                  - fitter.params.valuesdict()[PARAMETER])
            return diff
        #
        diff1 = calc(mf.METHOD_BOTH, probNan=0.05)
        diff2 = calc(mf.METHOD_BOTH, probNan=0.99)
        condition = (diff1 < diff2) or (np.abs(diff2 - diff1) < 1)
        self.assertTrue(condition)

    # FIXME
    def testFitDataTransformDct(self):
        return
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
        self._init()
        fitter1 = ModelFitterCore(th.ANTIMONY_MODEL, self.timeseries,
              list(th.PARAMETER_DCT.keys()))
        fitter1.fitModel()
        fittedModel = fitter1.getFittedModel()
        fitter2 = ModelFitterCore(fittedModel, self.timeseries,
              list(th.PARAMETER_DCT.keys()))
        fitter2.fitModel()
        # Should get same fit without changing the parameters
        self.assertTrue(np.isclose(np.var(fitter1.residualsTS.flatten()),
              np.var(fitter2.residualsTS.flatten()), rtol=0.01))

    def getFitter(self):
        fitter = th.getFitter(cls=ModelFitter)
        fitter.fitModel()
        fitter.bootstrap(numIteration=10)
        return fitter

    def testSerialize(self):
        if IGNORE_TEST:
            return
        fitter = self.getFitter()
        self.assertFalse(os.path.isfile(FILE_SERIALIZE))
        fitter.serialize(FILE_SERIALIZE)
        self.assertTrue(os.path.isfile(FILE_SERIALIZE))
        os.remove(FILE_SERIALIZE)

    def testDeserialize(self):
        if IGNORE_TEST:
            return
        fitter = self.getFitter()
        fitter.serialize(FILE_SERIALIZE)
        deserializedFitter = ModelFitter.deserialize(FILE_SERIALIZE)
        self.assertEqual(fitter.modelSpecification,
              deserializedFitter.modelSpecification)
        self.assertEqual(len(fitter.bootstrapResult.fittedStatistic.meanTS),
              len(deserializedFitter.bootstrapResult.fittedStatistic.meanTS))

    def testGetDefaultParameterValues(self):
        if IGNORE_TEST:
            return
        fitter = self.getFitter()
        parameterDct = self.fitter.getDefaultParameterValues()
        for name in parameterDct.keys():
            self.assertEqual(parameterDct[name], th.PARAMETER_DCT[name])

    def testWolfBug(self):
        if IGNORE_TEST:
            return
        fullDct = {
           "J1_n": (1, 1, 8),  # 4
           "J4_kp": (3600, 36000, 150000),  #76411
           "J5_k": (10, 10, 160),  # 80
           "J6_k": (1, 1, 10),  # 9.7
           "J9_k": (1, 50, 50),   # 28
           }
        for parameter in fullDct.keys():
            logger = Logger(logLevel=LEVEL_MAX)
            logger = Logger()
            ts = NamedTimeseries(csvPath=WOLF_DATA)
            parameterDct = {parameter: fullDct[parameter]}
            fitter = ModelFitter(WOLF_MODEL, ts[0:100],
                  parameterDct=parameterDct,
                  logger=logger, fitterMethods=["differential_evolution",
                         "leastsq"]) 
            fitter.fitModel(max_nfev=None)
        

if __name__ == '__main__':
    unittest.main()
