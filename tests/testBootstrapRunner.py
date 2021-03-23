# -*- coding: utf-8 -*-
"""
Created on Aug 19, 2020

@author: hsauro
@author: joseph-hellerstein
"""

import SBstoat
from SBstoat import _bootstrapRunner as br
from SBstoat._modelFitterCore import ModelFitterCore
from SBstoat.namedTimeseries import NamedTimeseries
from tests import _testHelpers as th

import numpy as np
import os
import pandas as pd
import unittest


IGNORE_TEST = True
IS_PLOT = True
FITTER = th.getFitter(cls=ModelFitterCore)
NUM_ITERATION = 20



class TestRunnerArgument(unittest.TestCase):

    def setUp(self):
        self.fitter = FITTER.copy()
        self.argument = br.RunnerArgument(self.fitter, numIteration=NUM_ITERATION)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(self.fitter.observedTS.equals(
              self.argument.fitter.observedTS))

class TestBootstrapRunner(unittest.TestCase):

    def setUp(self):
        self.fitter = FITTER.copy()
        self.argument = br.RunnerArgument(self.fitter, numIteration=NUM_ITERATION)
        self.runner = br.BootstrapRunner(self.argument)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.runner.numWorkUnit, NUM_ITERATION)

    def testRun(self):
        # TESTING
        self.runner.run()

    def testRun2(self):
        if IGNORE_TEST:
            return
        results = []
        while not self.runner.isDone:
            results.append(self.runner.run())
        dct = {}
        for parameter in results[0].parameters:
            dct[parameter] = [r.parameterDct[parameter][0] for r in results]
        self.assertEqual(len(results), NUM_ITERATION)
        df = pd.DataFrame(dct)
        meanDF = df.mean()
        stdDF = df.std()
        cvDF = stdDF/meanDF
        trues = [c < 0.5 for c in cvDF.values]
        self.assertTrue(all(trues))


if __name__ == '__main__':
    unittest.main()
