"""
Created on Nov 11, 2020

@author: joseph-hellerstein
"""

from SBstoat._testHarness import TestHarness
from SBstoat.logs import Logger

import numpy as np
import os
import unittest


IGNORE_TEST = False
IS_PLOT = False
DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(DIR, "testTestHarness.log")
DATA_DIR = os.path.join(os.path.dirname(DIR), "biomodels")
PATH_PAT = os.path.join(DATA_DIR, "BIOMD0000000%03d.xml")
INPUT_PATH = PATH_PAT % 339
VARIABLE_NAMES = ["Va_Xa", "IIa_Tmod", "VIIa_TF"]
PARAMETER_NAMES = ["r27_c", "r28_c", "r29_c"]
VARIABLE_NAMES = ["Pk", "VK"]
PARAMETER_NAMES = ["d_Pk", "d_VK"]
LOGGER = Logger()

if os.path.isfile(LOG_PATH):
    os.remove(LOG_PATH)


class TestFunctions(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.harness = TestHarness(INPUT_PATH, PARAMETER_NAMES, VARIABLE_NAMES,
              logger=LOGGER)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(len(self.harness.parameterValueDct), len(PARAMETER_NAMES))

    def testConstructorInvalid(self):
        if IGNORE_TEST:
            return
        with self.assertRaises(ValueError):
            self.harness = TestHarness("dummy", VARIABLE_NAMES, PARAMETER_NAMES,
                  logger=LOGGER)

    def testEvaluate(self):
        # Works for: 200, 210
        if IGNORE_TEST:
            return
        modelNums = 200 + np.array(range(2))
        fitModelRelerrors = []
        bootstrapRelerrors = []
        erroredModels = []
        nonErroredModels = []
        for modelNum in modelNums:
            input_path = PATH_PAT % modelNum
            try:
                harness = TestHarness(input_path, logger=LOGGER)
                harness.evaluate(stdResiduals=1.0, fractionParameterDeviation=1.0,
                      relError=2.0)
                nonErroredModels.append(modelNum)
                values = [v for v in 
                      harness.fitModelResult.parameterRelErrorDct.values()]
                fitModelRelerrors.extend(values)
                values = [v for v in 
                      harness.bootstrapResult.parameterRelErrorDct.values()]
                bootstrapRelerrors.extend(values)
            except:
                erroredModels.append(modelNum)
        self.assertEqual(len(nonErroredModels), 2)
        self.assertEqual(len(erroredModels), 0)

    def testBug258(self):
        if IGNORE_TEST:
            return
        # Smoke test
        input_path = PATH_PAT % 258
        harness = TestHarness(input_path, logger=LOGGER)
        harness.evaluate(stdResiduals=1.0, fractionParameterDeviation=1.0,
              relError=2.0)

    def testBug157(self):
        # Bug with setting min and max values for parameters
        if IGNORE_TEST:
            return
        # Smoke test
        input_path = PATH_PAT % 157
        harness = TestHarness(input_path, logger=LOGGER)
        harness.evaluate(stdResiduals=1.0, fractionParameterDeviation=1.0,
              relError=2.0)

    def testBug148(self):
        # "SBML error"
        if IGNORE_TEST:
            return
        # Smoke test
        input_path = PATH_PAT % 148
        harness = TestHarness(input_path, logger=LOGGER)
        with self.assertRaises(ValueError):
            harness.evaluate(stdResiduals=1.0, fractionParameterDeviation=1.0,
                  relError=2.0)

    def testBug437(self):
        # "One time column"
        if IGNORE_TEST:
            return
        # Smoke test
        input_path = PATH_PAT % 437
        harness = TestHarness(input_path, logger=LOGGER)
        harness.evaluate(stdResiduals=1.0, fractionParameterDeviation=1.0,
              relError=2.0)



if __name__ == '__main__':
    unittest.main()
