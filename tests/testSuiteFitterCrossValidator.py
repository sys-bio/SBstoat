# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

import SBstoat
import SBstoat._constants as cn
from SBstoat.modelFitter import ModelFitter
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
        observedTS = NamedTimeseries(th.TEST_DATA_PATH)
        numPoint = len(observedTS)
        testIdxs = [n for n in range(numPoint) if n % numFold == 0]
        trainIdxs = [n for n in range(numPoint) if n % numFold != 0]
        self.modelNames = MODEL_NAMES[0:self.numModel]
        self.trainTS = observedTS[trainIdxs]
        self.testTS = observedTS[testIdxs]
        self.modelSpecifications = mkRepeatedList(th.ANTIMONY_MODEL, self.numModel)
        self.trainTSCol = mkRepeatedList(self.testTS, self.numModel)
        self.testTSDct = {n: self.testTS for n in self.modelNames}
        self.parameterNames = list(th.PARAMETER_DCT.keys())
        self.parameterNamesCollection = mkRepeatedList(self.parameterNames,
              self.numModel)
        self.suiteFitter = SuiteFitterCore(self.modelSpecifications,
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
        

if __name__ == '__main__':
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        pass
    unittest.main()
