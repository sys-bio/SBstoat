# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:24:09 2020

@author: joseph-hellerstein
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME
from SBstoat.timeseriesStatistics import TimeseriesStatistics

import numpy as np
import os
import pandas as pd
import tellurium as te
import unittest


IGNORE_TEST = False
NUM_COL = 3
LENGTH = 5
COLNAMES = ["S%d" % d for d in range(NUM_COL-1)]
COLNAMES.insert(0, TIME)
ARR = np.reshape(np.array(range(NUM_COL*LENGTH)), (LENGTH, NUM_COL))
TIMESERIES = NamedTimeseries(array=ARR, colnames=COLNAMES)
TIMESERIES[TIME] = np.array(range(LENGTH))
COUNT = 5


class TestNamedTimeseries(unittest.TestCase):

    def setUp(self):
        self.timeseries = TIMESERIES
        self.statistics = TimeseriesStatistics(self.timeseries)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        colnames = list(COLNAMES)
        colnames.remove(TIME)
        diff = set(self.statistics.colnames).symmetric_difference(colnames)
        self.assertEqual(len(diff), 0)

    def testAccumulate(self):
        if IGNORE_TEST:
            return
        self.statistics.accumulate(TIMESERIES)
        self.assertTrue(self.statistics.sumTS.equals(TIMESERIES))

    def testCalculate(self):
        if IGNORE_TEST:
            return
        for _ in range(COUNT):
            self.statistics.accumulate(TIMESERIES)
        self.statistics.calculate()
        self.assertEqual(self.statistics.count, COUNT)
        self.assertTrue(self.statistics.meanTS.equals(TIMESERIES))
        stdTS = TIMESERIES.copy(isInitialize=True)
        self.assertTrue(self.statistics.stdTS.equals(stdTS))
            

if __name__ == '__main__':
  unittest.main()
