"""
Created on Nov 11, 2020

@author: joseph-hellerstein
"""

from SBstoat._testHarness import TestHarness
from SBstoat._logger import Logger

import numpy as np
import os
import unittest


IGNORE_TEST = False
IS_PLOT = False
DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(DIR), "biomodels")
PATH_PAT = os.path.join(DATA_DIR, "BIOMD0000000%d.xml")
INPUT_PATH = os.path.join(DIR, "BIOMD0000000339.xml")
VARIABLE_NAMES = ["Va_Xa", "IIa_Tmod", "VIIa_TF"]
PARAMETER_NAMES = ["r27_c", "r28_c", "r29_c"]
VARIABLE_NAMES = ["Pk", "VK"]
PARAMETER_NAMES = ["d_Pk", "d_VK"]
BIOMD_URL_PAT = "http://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000%s?filename=BIOMD0000000%s_url.xml"
URL_603 = BIOMD_URL_PAT % ("603", "603")

class TestFunctions(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.harness = TestHarness(INPUT_PATH, PARAMETER_NAMES, VARIABLE_NAMES)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(len(self.harness.parameterValueDct), len(PARAMETER_NAMES))

    def testConstructorInvalid(self):
        if IGNORE_TEST:
            return
        with self.assertRaises(ValueError):
            self.harness = TestHarness("dummy", VARIABLE_NAMES, PARAMETER_NAMES)

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
            logger = Logger(isReport=False)
            input_path = PATH_PAT % modelNum
            try:
                harness = TestHarness(input_path, logger=logger)
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
        self.assertEqual(len(nonErroredModels), 1)
        self.assertEqual(len(erroredModels), 1)


if __name__ == '__main__':
    unittest.main()
