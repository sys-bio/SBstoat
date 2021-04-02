# -*- coding: utf-8 -*-
"""
Created on Tue March 11, 2021

@author: joseph-hellerstein
"""

import SBstoat
import SBstoat._constants as cn
from SBstoat.abstractCrossValidator import AbstractFitter, \
      AbstractCrossValidator, FitterRunner

import matplotlib
import numpy as np
import lmfit
import unittest


IGNORE_TEST = False
IS_PLOT = False
LOWER = 0.0
UPPER = 10.0
VALUE = 1.0
PARAMETER_NAMES = np.array(["a", "b"])
NUM_FOLD = 5
NUM_POINT = 20

############# TEST INFRASTRUCTURE ################
class Fitter(AbstractFitter):

    def __init__(self, parameterNames=None, values=None, noiseStd=0.0):
        super().__init__()
        if parameterNames is None:
            self.parameterNames = PARAMETER_NAMES
        else:
            self.parameterNames = parameterNames
        if values is None:
            self.valueArr = np.repeat(VALUE, len(self.parameterNames))
        else:
            self.valuesArr = np.array(values)
        self.trueParameters = lmfit.Parameters()
        self.noiseStd = noiseStd
        for value, parameterName in zip(self.valueArr, self.parameterNames):
            self.trueParameters.add(parameterName,
                  min=LOWER, max=UPPER, value=value)
        self._parameters = self._mkParameters()

    @property
    def parameters(self):
        return self._parameters

    def _mkParameters(self):
        """
        Creates variations of the orginal parameter values.
        
        Returns
        -------
        lmfit.Parameters
        """
        parameters = lmfit.Parameters()
        for parameterName, value in zip(self.parameterNames, self.valueArr):
            newValue = value + np.random.normal(0, self.noiseStd)
            parameters.add(parameterName,
                  min=LOWER, max=UPPER, value=newValue)
        return parameters

    def fit(self):
        pass

    def score(self):
        parameterValueArr = np.array(list(self.parameters.valuesdict().values()))
        return 1 - np.var(parameterValueArr)


class CrossValidator(AbstractCrossValidator):

    def __init__(self, isParallel=True, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            optional arguments for Fitter
        """
        self.isParallel = isParallel
        self.kwargs = kwargs
        super().__init__()

    def _getFitterGenerator(self, numFold):
        for _ in range(numFold):
            yield Fitter(**self.kwargs)

    def crossValidate(self, numFold):
        fitterGenerator = self._getFitterGenerator(numFold)
        self._crossValidate(fitterGenerator, isParallel=self.isParallel)


class BadCrossValidator(AbstractCrossValidator):

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            optional arguments for Fitter
        """
        self.kwargs = kwargs
        super().__init__()
        

################ TEST CLASSES #############
class TestFitterRunner(unittest.TestCase):

    def setUp(self):
        self.fitter = Fitter()
        self.numWorkUnit = 3
        self.runner = FitterRunner(np.repeat(self.fitter, self.numWorkUnit))

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.runner.numWorkUnit, self.numWorkUnit)
        self.assertFalse(self.runner.isDone)

    def testRun(self):
        if IGNORE_TEST:
            return
        result = self.runner.run()
        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result[0][0], lmfit.Parameters))
        self.assertTrue(isinstance(result[0][1], float))
        

class TestAbstractFitter(unittest.TestCase):

    def setUp(self):
        self.fitter = Fitter()

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        keyArr = np.array(list(self.fitter.parameters.valuesdict().keys()))
        valueArr = np.array(list(self.fitter.parameters.valuesdict().values()))
        np.testing.assert_array_equal(PARAMETER_NAMES, keyArr)
        trues = [v == VALUE for v in valueArr]
        self.assertTrue(all(trues))

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        fitter = Fitter(noiseStd=1.0)
        keyArr = np.array(list(self.fitter.parameters.valuesdict().keys()))
        np.testing.assert_array_equal(PARAMETER_NAMES, keyArr)

    def testFit(self):
        if IGNORE_TEST:
            return
        self.fitter.fit()  #Smoke tests

    def testScore(self):
        if IGNORE_TEST:
            return
        score = self.fitter.score()
        self.assertTrue(np.isclose(score, 1))
        #
        fitter = Fitter(noiseStd=0.1)
        score = fitter.score()
        self.assertLess(score, 1.0)
 

class TestCrossValidator(unittest.TestCase):

    def setUp(self):
        self.validator = CrossValidator(noiseStd=0)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.validator.numFold, 0)

    def testCrossValidateSequential(self):
        if IGNORE_TEST:
            return
        self.validator.crossValidate(NUM_FOLD)
        for items in [self.validator.cvFitters, self.validator.cvRsqs,
              self.validator.cvParametersCollection]:
            self.assertEqual(len(items), NUM_FOLD)
        self.assertTrue(isinstance(self.validator.cvFitters[0], Fitter))
        self.assertTrue(isinstance(self.validator.cvRsqs[0], float))
        self.assertTrue(isinstance(self.validator.cvParametersCollection[0],
              lmfit.Parameters))

    def testCrossValidateParallel(self):
        if IGNORE_TEST:
            return
        numFold=10
        validator = CrossValidator(isParallel=True, noiseStd=0)
        validator.crossValidate(numFold)
        for items in [validator.cvRsqs, validator.cvParametersCollection]:
            self.assertEqual(len(items), numFold)
        self.assertTrue(isinstance(validator.cvFitters[0], Fitter))
        self.assertTrue(isinstance(validator.cvRsqs[0], float))
        self.assertTrue(isinstance(validator.cvParametersCollection[0],
              lmfit.Parameters))

    def testFoldIdxGenerator(self):
        if IGNORE_TEST:
            return
        generator = AbstractCrossValidator.getFoldIdxGenerator(
              NUM_POINT, NUM_FOLD)
        allTestIdxs = []
        foldSize = NUM_POINT // NUM_FOLD
        for trainIdxs, testIdxs in generator:
            self.assertEqual(len(testIdxs),  foldSize)
            allTestIdxs.extend(testIdxs)
            self.assertEqual(len(trainIdxs), NUM_POINT - foldSize)
        diff = set(allTestIdxs).symmetric_difference(range(NUM_POINT))
        self.assertEqual(len(diff), 0)

    def testReportParameters(self):
        if IGNORE_TEST:
            return
        validator = CrossValidator(noiseStd=0.5)
        validator.crossValidate(NUM_FOLD)
        df = validator.parameterDF
        self.assertEqual(len(df), len(PARAMETER_NAMES))
        diff = set([cn.MEAN, cn.STD, cn.COUNT]).symmetric_difference(df.columns)
        self.assertEqual(len(diff), 0)

    def testScoreDF(self):
        if IGNORE_TEST:
            return
        validator = CrossValidator(noiseStd=0.5)
        validator.crossValidate(NUM_FOLD)
        self.assertEqual(len(validator.scoreDF), NUM_FOLD)

    def testMissingOverride(self):
        if IGNORE_TEST:
            return
        validator = BadCrossValidator(noiseStd=0)
        with self.assertRaises(RuntimeError):
            validator.crossValidate(NUM_FOLD)


if __name__ == '__main__':
    unittest.main()
