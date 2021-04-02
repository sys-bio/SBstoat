# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

import SBstoat
from SBstoat._modelFitterCrossValidator import ModelFitterCrossValidator, Fitter
import SBstoat._constants as cn
from SBstoat.modelFitter import ModelFitter
from SBstoat import _helpers
from SBstoat.logs import Logger, LEVEL_MAX
from SBstoat._modelFitterCore import ModelFitterCore
from SBstoat.namedTimeseries import NamedTimeseries, TIME
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
NUM_FOLD = 5


class TestFitter(unittest.TestCase):

    def setUp(self):
        pass

    def _init(self):
        trainIdxs = list(range(th.NUM_POINT))
        testIdxs = list(range(th.NUM_POINT))
        self.fitter = th.getFitter(cls=Fitter, trainIdxs=trainIdxs,
              testIdxs=testIdxs)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self._init()
        self.assertTrue(self.fitter.trainTS.equals(self.fitter.testTS))

    def testScore(self):
        if IGNORE_TEST:
            return
        self._init()
        self.fitter.fit()
        rsq1 = self.fitter.score()
        self.assertGreater(rsq1, 0.9)
        #
        size = th.NUM_POINT // 6
        trainIdxs = list(range(th.NUM_POINT))[:size]
        testIdxs = list(range(th.NUM_POINT))[size:]
        fitter = th.getFitter(cls=Fitter, trainIdxs=trainIdxs,
              testIdxs=testIdxs)
        fitter.fit()
        rsq2 = fitter.score()
        self.assertGreater(rsq1, rsq2)


class TestModelFitterCrossValidator(unittest.TestCase):

    def setUp(self):
        self.validator = th.getFitter(cls=ModelFitterCrossValidator)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(len(self.validator.observedTS), th.NUM_POINT)

    def testCrossValidateParallel(self):
        if IGNORE_TEST:
            return
        numFold = 30
        self.validator.crossValidate(numFold, isParallel=True)
        self.assertEqual(numFold, len(self.validator.cvFitters))

    def testCrossValidateSequential(self):
        if IGNORE_TEST:
            return
        numFold = 30
        self.validator.crossValidate(numFold, isParallel=False)
        self.assertEqual(numFold, len(self.validator.cvFitters))

    def testScoreDF(self):
        if IGNORE_TEST:
            return
        self.validator.crossValidate(th.NUM_POINT)
        self.assertLess(min(self.validator.scoreDF[cn.SCORE]), 0.1)

    def testParameterDF(self):
        if IGNORE_TEST:
            return
        self.validator.crossValidate(th.NUM_POINT)
        df = self.validator.parameterDF
        # Parameter values should be increasing
        for parameter1, value1 in df[cn.MEAN].items():
            for parameter2, value2 in df[cn.MEAN].items():
                if int(parameter1[-1]) < int(parameter2[-1]):
                    self.assertLess(value1, value2)
        self.assertLess(min(self.validator.scoreDF[cn.SCORE]), 0.1)


if __name__ == '__main__':
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        pass
    unittest.main()
