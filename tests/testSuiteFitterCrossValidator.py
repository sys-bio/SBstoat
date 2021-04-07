# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

import SBstoat
import SBstoat._constants as cn
from SBstoat.modelFitter import ModelFitter
from SBstoat.suiteFitter import mkSuiteFitter
from SBstoat.namedTimeseries import NamedTimeseries
from SBstoat._suiteFitterCore import SuiteFitterCore
from SBstoat._suiteFitterCrossValidator import SuiteFitterWrapper,  \
      SuiteFitterCrossValidator
from tests import _testHelpers as th

import matplotlib
import numpy as np
import lmfit
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_MODEL = 3
MODEL_NAMES = ["model_%d" % d for d in range(NUM_MODEL)]
METHODS = [SBstoat.OptimizerMethod("differential_evolution", {cn.MAX_NFEV: 100})]
METHODS = [SBstoat.OptimizerMethod("leastsq", {cn.MAX_NFEV: 100})]


def mkRepeatedList(list, repeat):
    return [list for _ in range(repeat)]


################ TEST CLASSES #############
class TestSuiteFitterWrapper(unittest.TestCase):

    def setUp(self):
        self._init()

    def _init(self, numFold=3, numModel=NUM_MODEL):
        """
        Initializes the test instance variables for a single fold.

        Parameters
        ----------
        numFold: int
        numModel: int
        """
        self.numModel = numModel
        self.observedTS = NamedTimeseries(th.TEST_DATA_PATH)
        numPoint = len(self.observedTS)
        testIdxs = [n for n in range(numPoint) if n % numFold == 0]
        trainIdxs = [n for n in range(numPoint) if n % numFold != 0]
        self.modelNames = MODEL_NAMES[0:self.numModel]
        self.trainTS = self.observedTS[trainIdxs]
        self.testTS = self.observedTS[testIdxs]
        self.modelSpecifications = mkRepeatedList(th.ANTIMONY_MODEL, self.numModel)
        self.trainTSCol = mkRepeatedList(self.testTS, self.numModel)
        self.testTSDct = {n: self.testTS for n in self.modelNames}
        self.parameterNames = list(th.PARAMETER_DCT.keys())
        self.parameterNamesCollection = mkRepeatedList(self.parameterNames,
              self.numModel)
        self.suiteFitter = mkSuiteFitter(self.modelSpecifications,
              self.trainTSCol,
              self.parameterNamesCollection, modelNames=self.modelNames)
        self.wrapper = SuiteFitterWrapper(self.suiteFitter, self.testTSDct)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.wrapper.testTSDct, dict))

    def testFit(self):
        if IGNORE_TEST:
            return
        self.wrapper.fit()
        self.assertTrue(isinstance(self.wrapper.parameters, lmfit.Parameters))

    def testScore(self):
        if IGNORE_TEST:
            return
        self.wrapper.fit()
        rsq = self.wrapper.score()
        self.assertGreater(rsq, 0.9)

    def testScoreSubset(self):
        if IGNORE_TEST:
            return
        self._init(numFold=2)
        self.wrapper.fit()
        rsq = self.wrapper.score()
        self.assertGreater(rsq, 0.9)


class TestSuiteFitterCrossValidator(unittest.TestCase):

    def _init(self, numModel=NUM_MODEL):
        self.numModel = numModel
        self.observedTS = NamedTimeseries(th.TEST_DATA_PATH)
        self.modelNames = MODEL_NAMES[0:numModel]
        self.modelSpecifications = mkRepeatedList(th.ANTIMONY_MODEL, numModel)
        self.datasets = mkRepeatedList(self.observedTS, numModel)
        self.parameterNames = list(th.PARAMETER_DCT.keys())
        self.parameterNamesCollection = mkRepeatedList(self.parameterNames,
              numModel)
        self.suiteFitter = mkSuiteFitter(
              self.modelSpecifications, self.datasets,
              self.parameterNamesCollection, modelNames=self.modelNames,
              fitterMethods=METHODS)

    def setUp(self):
        self._init()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        def getObservedTS(idx):
            return self.suiteFitter.fitterDct[MODEL_NAMES[idx]].observedTS
        #
        self.assertEqual(len(getObservedTS(0)), len(getObservedTS(1)))

    def testGetFitterGenerator(self):
        if IGNORE_TEST:
            return
        numFold = 2
        expectedObservedLen = len(self.observedTS) // 2
        modelName = MODEL_NAMES[0]
        generator = self.suiteFitter._getFitterGenerator(numFold)
        expectedLength = len(self.observedTS) // numFold
        self.assertEqual(len(self.suiteFitter.fitterDct), self.numModel)
        actualNumFold = 0
        expectedObservedLen = len(self.observedTS) // 2
        for suiteFitterWrapper in generator:
            firstModelFitter = list(
                  suiteFitterWrapper.suiteFitter.fitterDct.values())[0]
            self.assertEqual(len(firstModelFitter.observedTS), expectedLength)
            actualNumFold += 1
            self.assertTrue(isinstance(suiteFitterWrapper, SuiteFitterWrapper))
            modelFitter = suiteFitterWrapper.suiteFitter.fitterDct[modelName]
            actualLength = len(modelFitter.observedTS)
            self.assertEqual(actualLength, expectedLength)
        self.assertEqual(numFold, actualNumFold)

    def testCrossValidate(self):
        if IGNORE_TEST:
            return
        numFold = 5
        numParameter = len(th.PARAMETER_DCT)
        self.suiteFitter.crossValidate(numFold)
        self.assertEqual(len(self.suiteFitter.scoreDF), numFold)
        self.assertEqual(len(self.suiteFitter.parameterDF), numParameter)

        

if __name__ == '__main__':
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        pass
    unittest.main()
