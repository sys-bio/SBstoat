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
THRESHOLD_PROB = 0.5

def _mkTimeseries(numRow=NUM_ROW, numCol=NUM_COL, std=STD):
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

def _setRandomMissing(timeseries):
    for col in timeseries.colnames:
        for idx in range(len(timeseries)):
            if np.random.random() < THRESHOLD_PROB:
                timeseries[col][idx] = np.nan

class TestObservationSynthesizer(unittest.TestCase):

    def setUp(self):
        self.observedTS, fittedTS, residualsTS = _mkTimeseries()

    def testConstructor(self):
        if IGNORE_TEST:
            return
        with self.assertRaises(TypeError):
            _ = obs.ObservationSynthesizer()
        pass

    def _testCalculate(self, synthesizer=None, maxDiff=0.1):
        if synthesizer is None:
            synthesizer = self.synthesizer
        def differenceDistributions(arr1, arr2):
            return np.abs(np.mean(arr1)-np.mean(arr2)),  \
                   np.abs(np.std(arr1) - np.std(arr2))
        #
        newObservedTS = synthesizer.calculate()
        residualsTS = newObservedTS.copy()
        residualsTS[self.columns] -= self.fittedTS[self.columns]
        for column in self.columns:
          meanDiff, stdDiff = differenceDistributions(
              self.residualsTS[column], residualsTS[column])
          for diff in [meanDiff, stdDiff]:
              self.assertLess(diff, maxDiff)

class TestObservationSynthesizerRandomizedResiduals(TestObservationSynthesizer):

    def setUp(self):
        self.observedTS, self.fittedTS, self.residualsTS = _mkTimeseries()
        _setRandomMissing(self.observedTS)
        self.synthesizer = obs.ObservationSynthesizerRandomizedResiduals(
            observedTS=self.observedTS,
            fittedTS=self.fittedTS)
        self.columns = self.observedTS.colnames

    def testCalculate(self):
        if IGNORE_TEST:
            return
        self._testCalculate()

    def testCalculateWithMissingValues(self):
        if IGNORE_TEST:
            return
        self._testCalculate()

    def testCalculateWithStdthreshold(self):
        if IGNORE_TEST:
            return
        stds = []
        for maxSL in [0.9, 0.4, 0.001]:
            synthesizer = obs.ObservationSynthesizerRandomizedResiduals(
                observedTS=self.observedTS,
                fittedTS=self.fittedTS, filterSL=maxSL)
            observedTS = synthesizer.calculate()
            stds.append(np.std(observedTS.flatten() - self.fittedTS.flatten()))
        self.assertGreater(stds[1], stds[0])
        self.assertGreater(stds[2], stds[1])


class TestObservationSynthesizerRandomizedErrors(TestObservationSynthesizer):

    def setUp(self):
        self.observedTS, self.fittedTS, self.residualsTS = _mkTimeseries()
        _setRandomMissing(self.observedTS)
        self.synthesizer = obs.ObservationSynthesizerRandomErrors(
              fittedTS=self.fittedTS)
        self.columns = self.observedTS.colnames

    def testCalculate(self):
        if IGNORE_TEST:
            return
        self._testCalculate()


if __name__ == '__main__':
    unittest.main()
