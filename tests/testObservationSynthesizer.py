# -*- coding: utf-8 -*-
"""
Created on Aug 30, 2020

@author: hsauro
@author: joseph-hellerstein
"""

from SBstoat import observationSynthesizer as obs
from SBstoat.namedTimeseries import NamedTimeseries, TIME

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
NUM_COL = 3
NUM_ROW = 1000
STD = 0.1
MEAN = 10

def mkTimeseries(numRow=NUM_ROW, numCol=NUM_COL, std=STD):
    colnames = ["V%d" % d for d in range(NUM_COL)]
    allColnames = list(colnames)
    allColnames.insert(0, TIME)
    timeArr = np.array(range(NUM_ROW))
    timeArr = np.reshape(timeArr, (NUM_ROW, 1))
    #
    def addTime(arr):
        newArr = np.concatenate([timeArr, arr], axis=1)
        return newArr
    #
    residualsTS = NamedTimeseries(
        array=addTime(np.random.normal(0, std, (numRow, numCol))),
        colnames=allColnames)
    fittedTS = NamedTimeseries(
        array=addTime(np.random.normal(MEAN, 10*std, (numRow, numCol))),
        colnames=allColnames)
    fittedTS[colnames] = np.floor(fittedTS[colnames])
    observedTS = fittedTS.copy()
    observedTS[colnames] += residualsTS[colnames]
    return observedTS, fittedTS, residualsTS
        

class TestObservationSynthesizer(unittest.TestCase):

    def setUp(self):
        self.observedTS, fittedTS, residualsTS = mkTimeseries()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        with self.assertRaises(TypeError):
            _ = obs.ObservationSynthesizer()
        pass

    def _testCalculate(self):
        MAX_DIFF = 0.05
        def differenceDistributions(arr1, arr2):
            return np.abs(np.mean(arr1)-np.mean(arr2)),  \
                   np.abs(np.std(arr1) - np.std(arr2))
        #
        newObservedTS = self.synthesizer.calculate()
        residualsTS = newObservedTS.copy()
        residualsTS[self.columns] -= self.fittedTS[self.columns]
        for column in self.columns:
          meanDiff, stdDiff = differenceDistributions(
              self.residualsTS[column], residualsTS[column])
          for diff in [meanDiff, stdDiff]:
              self.assertLess(diff, MAX_DIFF)

class TestObservationSynthesizerRandomizedResiduals(TestObservationSynthesizer):

    def setUp(self):
        self.observedTS, self.fittedTS, self.residualsTS = mkTimeseries()
        self.synthesizer = obs.ObservationSynthesizerRandomizedResiduals(
            observedTS=self.observedTS,
            fittedTS=self.fittedTS)
        self.columns = self.observedTS.colnames

    def testCalculate(self):
        if IGNORE_TEST:
            return
        self._testCalculate()


class TestObservationSynthesizerRandomizedErrors(TestObservationSynthesizer):

    def setUp(self):
        self.observedTS, self.fittedTS, self.residualsTS = mkTimeseries()
        self.synthesizer = obs.ObservationSynthesizerRandomErrors(self.fittedTS)
        self.columns = self.observedTS.colnames

    def testCalculate(self):
        if IGNORE_TEST:
            return
        self._testCalculate()


if __name__ == '__main__':
    unittest.main()
