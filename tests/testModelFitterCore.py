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
from tests import _testConstants as tcn

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
        self._init()
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
        self._init()
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
                  list(th.PARAMETER_DCT.keys()), fitterMethods=method)
            fitter.fitModel()
            for parameter in ["k1", "k2", "k3", "k4", "k5"]:
                diff = np.abs(th.PARAMETER_DCT[parameter]
                      - dct[parameter])
                frac = diff/dct[parameter]
                self.assertLess(diff/dct[parameter], 5.0)
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
        self._init()
        NUM_FIT_REPEATS = [1, 5]
        def test(method):
            compareDct = {}
            for numFitRepeat in NUM_FIT_REPEATS:
                fitter = ModelFitterCore(th.ANTIMONY_MODEL, self.timeseries,
                      list(th.PARAMETER_DCT.keys()), fitterMethods=method,
                      numFitRepeat=numFitRepeat)
                fitter.fitModel()
                compareDct[numFitRepeat] = self.checkParameterValues()
            for parameter in ["k1", "k2", "k3", "k4", "k5"]:
                first = NUM_FIT_REPEATS[0]
                last = NUM_FIT_REPEATS[1]
                self.assertLessEqual(compareDct[first][parameter],
                      compareDct[last][parameter])
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
                  list(th.PARAMETER_DCT.keys()), fitterMethods=method)
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
        std1 = np.var(fitter1.residualsTS.flatten())
        std2 = np.var(fitter2.residualsTS.flatten())
        if tcn.IGNORE_ACCURACY:
            return
        self.assertTrue(np.isclose(std1, std2, rtol=0.1))

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
           #"J1_n": (1, 1, 8),  # 4
           #"J4_kp": (3600, 36000, 150000),  #76411
           #"J5_k": (10, 10, 160),  # 80
           #"J6_k": (1, 1, 10),  # 9.7
           "J9_k": (1, 50, 50),   # 28
           }
        for parameter in fullDct.keys():
            logger = Logger(logLevel=LEVEL_MAX)
            logger = Logger()
            ts = NamedTimeseries(csvPath=WOLF_DATA)
            parameterDct = {parameter: fullDct[parameter]}
            fitter = ModelFitter(WOLF_MODEL, ts[0:100],
                  parameterDct=parameterDct,
                  logger=logger, fitterMethods=[
                         "differential_evolution", "leastsq"]) 
            fitter.fitModel()
            self.assertTrue("J9_k" in fitter.reportFit())

    def testMikeBug(self):
        if IGNORE_TEST:
            return
        model = te.loada('''
        function Fi(v, ri, kf, kr, i, s, p, Kmi, Kms, Kmp, wi, ms, mp)
            ((ri+(1-ri)*(1/(1+i/Kmi)))^wi)*(kf*(s/Kms)^ms-kr*(p/Kmp)^mp)/((1+(s/Kms))^ms+(1+(p/Kmp))^mp-1)
        end
        function F0(v, kf, kr, s, p, Kms, Kmp, ms, mp)
            (kf*(s/Kms)^ms-kr*(p/Kmp)^mp)/((1+(s/Kms))^ms+(1+(p/Kmp))^mp-1)
        end
        function Fa(v, ra, kf, kr, a, s, p, Kma, Kms, Kmp, wa, ms, mp)
            ((ra+(1-ra)*((a/Kma)/(1+a/Kma)))^wa)*(kf*(s/Kms)^ms-kr*(p/Kmp)^mp)/((1+(s/Kms))^ms+(1+(p/Kmp))^mp-1)
        end
        function Fiii(v, ri1, ri2, ri3, kf, kr, i1, i2, i3, s, p, Kmi1, Kmi2, Kmi3, Kms, Kmp, wi1, wi2, wi3, ms, mp)
            ((ri1+(1-ri1)*(1/(1+i1/Kmi1)))^wi1) * ((ri2+(1-ri2)*(1/(1+i2/Kmi2)))^wi2) * ((ri3+(1-ri3)*(1/(1+i3/Kmi3)))^wi3) * (kf*(s/Kms)^ms-kr*(p/Kmp)^mp)/((1+(s/Kms))^ms+(1+(p/Kmp))^mp-1)
        end
        
        model modular_EGFR_current_128()
        
        
        // Reactions
        FreeLigand: -> L; Fa(v_0, ra_0, kf_0, kr_0, Lp, E, L, Kma_0, Kms_0, Kmp_0, wa_0, ms_0, mp_0);
        Phosphotyrosine: -> P; Fi(v_1, ri_1, kf_1, kr_1, Mig6, L, P, Kmi_1, Kms_1, Kmp_1, wi_1, ms_1, mp_1);
        Ras: -> R; Fiii(v_2, ri1_2, ri2_2, ri3_2, kf_2, kr_2, Spry2, P, E, P, R, Kmi1_2, Kmi2_2, Kmi3_2, Kms_2, Kmp_2, wi1_2, wi2_2, wi3_2, ms_2, mp_2);
        Erk: -> E; F0(v_3, kf_3, kr_3, R, E, Kms_3, Kmp_3, ms_3, mp_3);
        
        // Species IVs
        Lp = 100;
        E = 0;
        L = 1000;
        Mig6 = 100;
        P = 0;
        Spry2 = 10000;
        R = 0;
        
        // Parameter values
        v_0 = 1;
        ra_0 = 1;
        kf_0 = 1;
        kr_0 = 1;
        Kma_0 = 1;
        Kms_0 = 1;
        Kmp_0 = 1;
        wa_0 = 1;
        ms_0 = 1;
        mp_0 = 1;
        v_1 = 1;
        ri_1 = 1;
        kf_1 = 1;
        kr_1 = 1;
        Kmi_1 = 1;
        Kms_1 = 1;
        Kmp_1 = 1;
        wi_1 = 1;
        ms_1 = 1;
        mp_1 = 1;
        v_2 = 1;
        ri1_2 = 1;
        ri2_2 = 1;
        ri3_2 = 1;
        kf_2 = 1;
        kr_2 = 1;
        Kmi1_2 = 1;
        Kmi2_2 = 1;
        Kmi3_2 = 1;
        Kms_2 = 1;
        Kmp_2 = 1;
        wi1_2 = 1;
        wi2_2 = 1;
        wi3_2 = 1;
        ms_2 = 1;
        mp_2 = 1;
        v_3 = 1;
        kf_3 = 1;
        kr_3 = 1;
        Kms_3 = 1;
        Kmp_3 = 1;
        ms_3 = 1;
        mp_3 = 1;
        
        end
        ''')
        observedPath = os.path.join(DIR, "mike_bug.csv")
        fitter = ModelFitter(model, observedPath, ["v_0", "ra_0", "kf_0", "kr_0", "Kma_0", "Kms_0", "Kmp_0", "wa_0", "ms_0",
                                                      "mp_0", "v_1", "ri_1", "kf_1", "kr_1", "Kmi_1", "Kms_1", "Kmp_1", "wi_1",
                                                      "ms_1", "mp_1", "v_2", "ri1_2", "ri2_2", "ri3_2", "kf_2", "kr_2",
                                                      "Kmi1_2", "Kmi2_2", "Kmi3_2", "Kms_2", "Kmp_2", "wi1_2", "wi2_2", "wi3_2",
                                                      "ms_2", "mp_2", "v_3", "kf_3", "kr_3", "Kms_3", "Kmp_3", "ms_3", "mp_3"])
        try:
            fitter.fitModel()
        except ValueError as err:
            pass
        self.assertIsNone(fitter.residualsTS)
        

if __name__ == '__main__':
    unittest.main()
