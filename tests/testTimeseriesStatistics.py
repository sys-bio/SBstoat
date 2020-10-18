# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:24:09 2020

@author: joseph-hellerstein
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME
from SBstoat import timeseriesStatistic as tss
from SBstoat.timeseriesStatistic import TimeseriesStatistic

import numpy as np
import os
import pandas as pd
import tellurium as te
import typing
import unittest


# Independent constants
IGNORE_TEST = True
IS_PLOT = True
NUM_COL = 5
LENGTH = 5
UNIFORM_LEN = 1000
COLNAMES = ["S%d" % d for d in range(NUM_COL-1)]
COLNAMES.insert(0, TIME)
UNIFORM_MEAN = 0.5
UNIFORM_STD = np.sqrt(1/12.0)
SIMPLE_CNT = 5
UNIFORM_CNT = 100


def mkTimeseries(length:int, colnames:typing.List[str],
      isRandom:bool=False)->NamedTimeseries:
    """
    Creates a time series of the desired shape.
 
    Parameters
    ----------
    length: number of time periods
    colnames: names of the columns, excluding TIME
    isRandom: generate uniform random numbers
    
    Returns
    -------
    NamedTimeseries
    """
    num_col = len(colnames)
    if isRandom:
        arr = np.random.random(num_col*length)
    else:
        arr = np.array(range(num_col*length))
    matrix = np.reshape(arr, (length, len(colnames)))
    timeseries = NamedTimeseries(array=matrix, colnames=colnames)
    timeseries[TIME] = np.array(range(length))
    return timeseries

# Constructed Constants
SIMPLE_TS = mkTimeseries(LENGTH, COLNAMES)
UNIFORM_TS = mkTimeseries(UNIFORM_LEN, COLNAMES, isRandom=True)


class TestNamedTimeseries(unittest.TestCase):

    def setUp(self):
        self.timeseries = SIMPLE_TS
        self.statistic = TimeseriesStatistic(self.timeseries)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        colnames = list(COLNAMES)
        colnames.remove(TIME)
        diff = set(self.statistic.colnames).symmetric_difference(colnames)
        self.assertEqual(len(diff), 0)

    def testAccumulate(self):
        if IGNORE_TEST:
            return
        self.statistic.accumulate(SIMPLE_TS)
        self.assertTrue(self.statistic.sumTS.equals(SIMPLE_TS))

    def testCalculate1(self):
        if IGNORE_TEST:
            return
        for _ in range(SIMPLE_CNT):
            self.statistic.accumulate(SIMPLE_TS)
        self.statistic.calculate()
        self.assertEqual(self.statistic.count, SIMPLE_CNT)
        self.assertTrue(self.statistic.meanTS.equals(SIMPLE_TS))
        stdTS = SIMPLE_TS.copy(isInitialize=True)
        self.assertTrue(self.statistic.stdTS.equals(stdTS))

    def mkStatistics(self, count):
        result = []
        for _ in range(count):
            statistic = TimeseriesStatistic(UNIFORM_TS, isCollectTimeseries=True)
            for _ in range(UNIFORM_CNT):
                statistic.accumulate(
                      mkTimeseries(UNIFORM_LEN, COLNAMES, isRandom=True))
            result.append(statistic)
        return result

    def evaluateStatistic(self, statistic, count=1):
        statistic.calculate()
        self.assertEqual(statistic.count, count*UNIFORM_CNT)
        mean = np.mean(statistic.meanTS.flatten())
        self.assertLess(np.abs(mean - UNIFORM_MEAN), 0.1)
        std = np.mean(statistic.stdTS.flatten())
        self.assertLess(np.abs(std - UNIFORM_STD), 0.1)
        lower = np.mean(statistic.lowerPercentileTS.flatten())
        self.assertLess(np.abs(lower - 0.01*tss.PERCENTILES[0]), 0.01)
        upper = np.mean(statistic.upperPercentileTS.flatten())
        self.assertLess(np.abs(upper - 0.01*tss.PERCENTILES[1]), 0.01)

    def testCalculate2(self):
        if IGNORE_TEST:
            return
        statistic = self.mkStatistics(1)[0]
        self.evaluateStatistic(statistic)

    def testEquals(self):
        if IGNORE_TEST:
            return
        statistic = TimeseriesStatistic(self.timeseries)
        self.assertTrue(self.statistic.equals(statistic))
        #
        statistic.accumulate(SIMPLE_TS)
        self.assertFalse(self.statistic.equals(statistic))

    def testCopy(self):
        if IGNORE_TEST:
            return
        statistic = self.statistic.copy()
        self.assertTrue(self.statistic.equals(statistic))
        #
        statistic = self.mkStatistics(1)[0]
        self.assertTrue(statistic.equals(statistic))

    def testMerge(self):
        # TESTING
        NUM = 4
        statistics = self.mkStatistics(NUM)
        statistic = TimeseriesStatistic.merge(statistics)
        statistic.calculate()
        self.evaluateStatistic(statistic, count=NUM)
            

if __name__ == '__main__':
  unittest.main()
