# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

import SBstoat
import SBstoat._modelFitterCore as mf
import SBstoat._constants as cn
from SBstoat.logs import Logger
from SBstoat.modelFitter import ModelFitter
from SBstoat import _helpers
from SBstoat._modelFitterCore import ModelFitterCore
from SBstoat.namedTimeseries import NamedTimeseries, TIME
import tellurium as te
from tests import _testHelpers as th
from tests import _testConstants as tcn

import copy
import lmfit
import matplotlib
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
METHODS = [SBstoat.OptimizerMethod("leastsq", {cn.MAX_NFEV: None})]


class TestModelFitterCore(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self._init()

    def _init(self):
        self._remove()
        self.timeseries = copy.deepcopy(TIMESERIES)
        self.fitter = th.getFitter(cls=ModelFitterCore, fitterMethods=METHODS)

    def tearDown(self):
        self._remove()

    def _remove(self):
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)

    def testMakeMethods(self):
        if IGNORE_TEST:
            return
        self._init()
        METHOD = "dummy"
        def test(methods):
            result = ModelFitter.makeMethods(methods, None)
            self.assertTrue(isinstance(result, list))
            optimizerMethod = result[0]
            self.assertEqual(optimizerMethod.method, METHOD)
            self.assertTrue(isinstance(optimizerMethod.kwargs, dict))
        #
        test([METHOD])
        test([METHOD, METHOD])
        test([_helpers.OptimizerMethod(method=METHOD, kwargs={})])
        #
        methods = _helpers.OptimizerMethod(method=METHOD,
              kwargs={"a": 1})
        result = ModelFitter.makeMethods([methods, methods], None)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self._init()
        self.assertIsNone(self.fitter.roadrunnerModel)
        self.assertGreater(len(self.fitter.observedTS), 0)
        #
        for variable in self.fitter.selectedColumns:
            self.assertTrue(variable in th.VARIABLE_NAMES)

    def testrpConstruct(self):
        if IGNORE_TEST:
            return
        self._init()
        fitter = ModelFitterCore.rpConstruct()
        def updateAttr(attr):
            if not attr in fitter.__dict__.keys():
                fitter.__setattr__(attr, None)
        #
        updateAttr("roadrunnerModel")
        updateAttr("observedTS")
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
        self.fitter.initializeRoadRunnerModel()
        params = self.fitter.mkParams()
        arr = self.fitter.calcResiduals(params)
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
        params = self.fitter.mkParams(["k1"])
        self.assertTrue(isinstance(params, lmfit.Parameters))

    def testFit1(self):
        if IGNORE_TEST:
            return
        self._init()
        self.fitter.fitModel()
        dct = self.checkParameterValues()
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
        #
        for method in [cn.METHOD_LEASTSQ, cn.METHOD_BOTH,
              cn.METHOD_DIFFERENTIAL_EVOLUTION]:
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
        for method in [cn.METHOD_LEASTSQ, cn.METHOD_BOTH,
              cn.METHOD_DIFFERENTIAL_EVOLUTION]:
            test(method)

    def testFitNanValues(self):
        if IGNORE_TEST:
            return
        self._init()
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
        diff1 = calc(cn.METHOD_BOTH, probNan=0.05)
        diff2 = calc(cn.METHOD_BOTH, probNan=0.99)
        condition = (diff1 < diff2) or (np.abs(diff2 - diff1) < 1)
        self.assertTrue(condition)

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
        self._init()
        fitter = self.getFitter()
        parameterValueDct = self.fitter.getDefaultParameterValues()
        for name in parameterValueDct.keys():
            self.assertEqual(parameterValueDct[name], th.PARAMETER_DCT[name])

    def testWolfBug(self):
        if IGNORE_TEST:
            return
        trueParameterDct = {
              "J1_n": 4,
              "J4_kp": 76411,
              "J5_k": 80,
              "J6_k": 9.7,
              "J9_k": 28,
              }
        parametersToFit= [
           SBstoat.Parameter("J1_n", lower=1, value=1, upper=8),  # 4
           SBstoat.Parameter("J4_kp", lower=3600, value=36000, upper=150000),  #76411
           SBstoat.Parameter("J5_k", lower=10, value=10, upper=160),  # 80
           SBstoat.Parameter("J6_k", lower=1, value=1, upper=10),  # 9.7
           SBstoat.Parameter("J9_k", lower=1, value=50, upper=50),   # 28
           ]
        ts = NamedTimeseries(csvPath=WOLF_DATA)
        methods = []
        for optName in ["differential_evolution", "leastsq"]:
            methods.append(SBstoat.OptimizerMethod(optName,
                  {cn.MAX_NFEV: 10}))
        fitter = ModelFitter(WOLF_MODEL, ts,
              parametersToFit=parametersToFit,
              fitterMethods=methods)
        fitter.fitModel()
        for name in [p.name for p in parametersToFit]:
            expected = trueParameterDct[name]
            actual = fitter.params.valuesdict()[name]
            self.assertLess(np.abs(np.log10(expected) - np.log10(actual)), 1.5)
            self.assertTrue(name in fitter.reportFit())

    def testMikeBug(self):
        if IGNORE_TEST:
            return
        fitter = self._makeMikeModel()
        self.assertIsNotNone(fitter.params)

    def _makeMikeModel(self, **kwargs):
        """Makes a model from Mike's data."""
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
        if "fitterMethods" not in kwargs:
            methods = []
            for optName in ["differential_evolution", "leastsq"]:
                methods.append(SBstoat.OptimizerMethod(optName,
                      {cn.MAX_NFEV: 10}))
            kwargs["fitterMethods"] = methods
        observedPath = os.path.join(DIR, "mike_bug.csv")
        fitter = ModelFitter(model, observedPath, logger=Logger(logLevel=1),
         parametersToFit=[
         #"v_0", "ra_0", "kf_0", "kr_0", "Kma_0", "Kms_0", "Kmp_0", "wa_0", "ms_0",
         #"mp_0", "v_1", "ri_1", "kf_1", "kr_1", "Kmi_1", "Kms_1", "Kmp_1", "wi_1",
         #"ms_1", "mp_1", "v_2", "ri1_2", "ri2_2", "ri3_2", "kf_2", "kr_2",
         "Kmi1_2", "Kmi2_2", "Kmi3_2", "Kms_2", "Kmp_2", "wi1_2", "wi2_2", "wi3_2",
         "ms_2", "mp_2", "v_3", "kf_3", "kr_3", "Kms_3", "Kmp_3", "ms_3", "mp_3"],
         **kwargs)
        fitter.fitModel()
        return fitter

    def testOptimizerMethod(self):
        if IGNORE_TEST:
            return
        METHOD_NAME = 'leastsq'
        optimizerMethod = _helpers.OptimizerMethod(
            method=METHOD_NAME,
            kwargs={ "max_nfev": 10})
        fitter1 = self._makeMikeModel(fitterMethods=[METHOD_NAME])
        fitter2 = self._makeMikeModel(fitterMethods=[optimizerMethod])
        self.assertTrue(True) # Smoke test

    def testOptimizerRestart(self):
        if IGNORE_TEST:
            return
        METHOD_NAME = 'leastsq'
        optimizerMethod = _helpers.OptimizerMethod(
            method=METHOD_NAME,
            kwargs={ "max_nfev": 10})
        fitter1 = self._makeMikeModel(fitterMethods=[optimizerMethod])
        fitter2 = self._makeMikeModel(fitterMethods=[optimizerMethod],
              numRestart=100)
        self.assertLess(fitter2.optimizer.rssq, fitter1.optimizer.rssq)

    def testMkParameters(self):
        if IGNORE_TEST:
            return
        NAMES = ["a", "b"]
        def test(parametersToFit):
            result = ModelFitterCore.mkParameters(parametersToFit=parametersToFit)
            self.assertTrue(isinstance(result, lmfit.Parameters))
            self.assertEqual(len(result.valuesdict()), len(parametersToFit))
        #
        test(NAMES)
        parametersToFit = [SBstoat.Parameter(n, value=1) for n in NAMES]
        test(parametersToFit)

    def testSelectCompatibleIndices(self):
        if IGNORE_TEST:
            return
        SIZE = 10
        SUB_SIZE = 5
        bigTimes = np.array(range(SIZE))
        smallTimes = np.random.permutation(bigTimes)[:SUB_SIZE]
        smallTimes = np.sort(smallTimes)
        resultArr = ModelFitterCore.selectCompatibleIndices(bigTimes,
              smallTimes)
        np.testing.assert_array_equal(smallTimes, resultArr)


       


if __name__ == '__main__':
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        pass
    unittest.main()
