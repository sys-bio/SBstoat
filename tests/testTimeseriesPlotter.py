# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:24:09 2020

@author: hsauro
@author: joseph-hellerstein
"""

from SBstoat.namedTimeseries import NamedTimeseries, mkNamedTimeseries, TIME
import SBstoat.namedTimeseries as namedTimeseries
from SBstoat import _plotOptions as po
from SBstoat.timeseriesPlotter import TimeseriesPlotter
from SBstoat import timeseriesPlotter as tp

import numpy as np
import os
import pandas as pd
import unittest
import matplotlib
import matplotlib.pyplot as plt


IGNORE_TEST = False
IS_PLOT = False
DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(DIR, "tst_data.txt")
DEFAULT_NUM_ROW = 2
DEFAULT_NUM_COL = 3
DEFAULT_NUM_PLOT = 5
        

class TestTimeseriesPlotter(unittest.TestCase):

    def setUp(self):
        self.timeseries = NamedTimeseries(csvPath=TEST_DATA_PATH)
        self.plotter = TimeseriesPlotter(isPlot=IS_PLOT)

    def testConstructor1(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.plotter.isPlot, bool))

    def testInitializeRowColumn(self):
        if IGNORE_TEST:
            return
        def test(maxCol, **kwargs):
            options = self.plotter._mkPlotOptionsMatrix(self.timeseries,
                   maxCol=maxCol, **kwargs)
            if po.NUM_ROW in kwargs:
                self.assertGreaterEqual(options.numRow, kwargs[po.NUM_ROW])
            if po.NUM_COL in kwargs:
                self.assertEqual(options.numCol, kwargs[po.NUM_COL])
        #
        test(3, **{})
        test(3, **{po.NUM_COL: 3})
        test(4, **{po.NUM_ROW: 2})
        test(5, **{po.NUM_ROW: 2})

    def testPlotSingle1(self):
        if IGNORE_TEST:
            return
        self.plotter.plotTimeSingle(self.timeseries,
              timeseries2=self.timeseries,
              numCol=4,
              marker=[None, '*'], alpha=[0.1, 0.8], color=["red", "g"],
              titlePosition=[0.8, 0.5], titleFontsize=10)
        self.plotter.plotTimeSingle(self.timeseries, numCol=4,
              marker=[None, '*'])
        self.plotter.plotTimeSingle(self.timeseries, numCol=4,
              subplotWidthSpace=0.2, yticklabels=[])
        self.plotter.plotTimeSingle(self.timeseries, columns=["S1", "S2", "S3"], numRow=2)
        self.plotter.plotTimeSingle(self.timeseries, numCol=4)
        self.plotter.plotTimeSingle(self.timeseries, numCol=2)
        self.plotter.plotTimeSingle(self.timeseries, numRow=2, numCol=3, ylabel="xxx")
        self.plotter.plotTimeSingle(self.timeseries, columns=["S1", "S2"])

    def testPlotSingle5(self):
        if IGNORE_TEST:
            return
        timeseries = self.timeseries.subsetColumns(["S1"])
        dct = {}
        indices = [i for i in range(len(timeseries)) if i % 4 == 0]
        for col in timeseries.allColnames:
            dct[col] = timeseries[col][indices]
        df = pd.DataFrame(dct)
        meanTS = NamedTimeseries(dataframe=df)
        meanTS[meanTS.colnames] = 1.1*meanTS[meanTS.colnames]
        stdTS = meanTS.copy()
        for col in meanTS.colnames:
            stdTS[col] = 1
        #
        self.plotter.plotTimeSingle(timeseries, timeseries2=self.timeseries,
              meanTS=meanTS, stdTS=stdTS, marker=[None, 'o', "^"],
              color=["b", "r", "g"])
        #
        self.plotter.plotTimeSingle(timeseries, meanTS=meanTS, stdTS=stdTS)
        #
        self.plotter.plotTimeSingle(timeseries, timeseries2=self.timeseries,
              marker='*')

    def testPlotSingle6(self):
        if IGNORE_TEST:
            return
        numCol = 3
        numRow = 2
        fig, axes = plt.subplots(numRow, numCol)
        self.plotter.isPlot = False
        for idx in range(numCol*numRow):
            if idx < numCol:
                row = 0
                col = idx
            else:
                row = 1
                col = idx - numCol
            position = [row, col]
            ax = axes[row, col]
            ts = self.timeseries.subsetColumns(self.timeseries.colnames[idx])
            self.plotter.plotTimeSingle(ts, ax_spec=ax, position=position,
                  numRow=2)
        if IS_PLOT:
            plt.show()

    def testPlotSingle7(self):
        if IGNORE_TEST:
            return
        def setTS(ts, frac):
            ts = self.timeseries.copy()
            for col in ts.colnames:
                ts[col] = frac*ts[col]
            return ts
        #
        numCol = 3
        numRow = 2
        fig, axes = plt.subplots(numRow, numCol)
        self.plotter.isPlot = False
        tsLower = setTS(self.timeseries, 0.7)
        tsUpper = setTS(self.timeseries, 1.5)
        for idx in range(numCol*numRow):
            if idx < numCol:
                row = 0
                col = idx
            else:
                row = 1
                col = idx - numCol
            position = [row, col]
            ax = axes[row, col]
            ts = self.timeseries.subsetColumns(self.timeseries.colnames[idx])
            self.plotter.plotTimeSingle(ts, ax_spec=ax, position=position,
                  bandLowTS=tsLower, bandHighTS=tsUpper,
                  numRow=2)
        if IS_PLOT:
            plt.show()

    def testPlotSingle8(self):
        # Plot with nan values
        if IGNORE_TEST:
            return
        def setTS(ts, mult, numNan):
            ts = self.timeseries.copy()
            for col in ts.colnames:
                ts[col] = mult*ts[col]
                ts[col][:numNan] = np.nan
            return ts
        #
        numCol = 3
        numRow = 2
        timeseries2 = setTS(self.timeseries, 2, 3)
        self.plotter.plotTimeSingle(self.timeseries, timeseries2=timeseries2, columns=["S1"],
              numRow=numRow, numCol=numCol)
        if IS_PLOT:
            plt.show()

    def mkTimeseries(self):
        ts2 = self.timeseries.copy()
        ts2[ts2.colnames] = ts2[ts2.colnames] + np.multiply(ts2[ts2.colnames], ts2[ts2.colnames])
        return ts2

    def testPlotSingle2(self):
        if IGNORE_TEST:
            return
        ts2 = self.mkTimeseries()
        self.plotter.plotTimeSingle(self.timeseries, timeseries2=ts2, columns=["S1", "S2"])
        self.plotter.plotTimeSingle(self.timeseries, timeseries2=ts2)
        self.plotter.plotTimeSingle(self.timeseries, timeseries2=ts2, numRow=2, numCol=3)

    def testPlotSingle3(self):
        if IGNORE_TEST:
            return
        self.plotter.plotTimeSingle(self.timeseries, ylabel="MISSING")

    def testPlotSingle4(self):
        if IGNORE_TEST:
            return
        ts2 = self.mkTimeseries()
        self.plotter.plotTimeSingle(self.timeseries, markersize=[2, 5],
              timeseries2=ts2, numRow=2, numCol=3, marker=[None, "o"])
        self.plotter.plotTimeSingle(self.timeseries, timeseries2=ts2,
               numRow=2, numCol=3, marker=[None, "o"], alpha=0.1)

    def testPlotMultiple1(self):
        if IGNORE_TEST:
            return
        ts2 = self.mkTimeseries()
        self.plotter.plotTimeMultiple(self.timeseries, timeseries2=ts2,
              suptitle="Testing", marker=[None, 'o', None, None, None, None],
              color=['r', 'g', 'b', 'brown', 'g', 'pink'],
              alpha=0.3)
        self.plotter.plotTimeMultiple(self.timeseries, timeseries2=ts2, suptitle="Testing", 
              numRow=1, numCol=1,
              marker="o")
        self.plotter.plotTimeMultiple(self.timeseries, timeseries2=ts2, suptitle="Testing", 
              numRow=2,
              marker="o")
        self.plotter.plotTimeMultiple(self.timeseries, suptitle="Testing")

    def testValuePairs(self):
        if IGNORE_TEST:
            return
        ts2 = self.mkTimeseries()
        self.plotter.plotValuePairs(self.timeseries, 
              [("S1", "S2"), ("S2", "S3"), ("S4", "S5")],
              numCol=2, numRow=2, alpha=0.3)
        self.plotter.plotValuePairs(self.timeseries, [("S1", "S2"), ("S2", "S3")], numRow=2)
        self.plotter.plotValuePairs(self.timeseries, [("S1", "S2")])

    def testPlotHistograms(self):
        if IGNORE_TEST:
            return
        self.plotter.plotHistograms(self.timeseries, numCol=2, alpha=0.3)

    def testPlotValuePairsBug(self):
        if IGNORE_TEST:
            return
        self.plotter.plotValuePairs(self.timeseries,
              pairs=[("S1", "S2"), ("S1", "S6"), ("S2", "S3")], numCol=3)

    def testPlotCompare(self):
        if IGNORE_TEST:
            return
        self.plotter.plotCompare(self.timeseries,
              self.timeseries, numCol=3)

    def testPlotAutoCorrelations(self):
        if IGNORE_TEST:
            return
        self.plotter.plotAutoCorrelations(self.timeseries, numCol=3,
              color=["black", "grey", "grey"],
              linestyle=[None, "dashed", "dashed"],
              alpha=[None, 0.5, 0.5])

    def testPlotCrossCorrelations(self):
        if IGNORE_TEST:
            return
        self.plotter.plotCrossCorrelations(self.timeseries,
              titleFontsize=8, titlePosition=(0.8, 0.8),
              suptitle="Cross Correlations",
              color=["black", "g", "g"], figsize=(12,10))



if __name__ == '__main__':
    matplotlib.use('TkAgg')
    unittest.main()
