"""Calculation of statistics for time series"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME

import numpy as np
import typing


class TimeseriesStatistic(object):

    def __init__(self, prototypeTS:NamedTimeseries,
          confidenceLimits:typing.Tuple[float,float]=(5.0, 95.0),
          isCollectTimeseries:bool=True):
        """
        Parameters
        ----------
        prototypeTS: same length and columns as desired
        confidenceLimits: Lower and upper limits of confidence limits
            for the timeseries accumulated)
        isCollectTimeseries: Must be enabled to calculate confidence limits
        """
        self.colnames = prototypeTS.colnames
        self.sumTS = prototypeTS.copy(isInitialize=True)
        self.ssqTS = prototypeTS.copy(isInitialize=True)
        self.confidenceLimits = confidenceLimits
        self._timeseries_list = []
        self._isCollectTimeseries = isCollectTimeseries
        # Statistics
        self.count = 0  # Count of timeseries accumulated
        # Means
        self.meanTS = prototypeTS.copy(isInitialize=True) # means
        # Standard deviations
        self.stdTS = prototypeTS.copy(isInitialize=True)  # standard deviations
        # Lower bound of confidence interval
        self.lowerConfidenceTS = prototypeTS.copy(isInitialize=True)
        # Upper bound of confidence interval
        self.upperConfidenceTS = prototypeTS.copy(isInitialize=True)

    def accumulate(self, newTS:NamedTimeseries):
        """
        Accumulates statistics for a new timeseries.
        """
        self.sumTS[self.colnames] = self.sumTS[self.colnames]  \
               + newTS[self.colnames]
        self.ssqTS[self.colnames] = self.ssqTS[self.colnames]  \
               + newTS[self.colnames]**2
        if self._isCollectTimeseries:
            self._timeseries_list.append(newTS)
        self.count += 1

    def calculate(self):
        """
        Calculates statistics.
        """
        if self.count <= 1:
            raise ValueError("Must accumulate at leat 2 timeseries before calculating statistics.")
        self._calculateMean()
        self._calculateStd()
        self._calculateConfidenceLimits()

    def _calculateMean(self):
        for col in self.colnames:
            self.meanTS[col] = (1.0*self.sumTS[col]) / self.count

    def _calculateStd(self):
        for col in self.colnames:
            self.stdTS[col] = self.ssqTS[col] - self.count*self.meanTS[col]**2
            self.stdTS[col] = self.stdTS[col] / (self.count - 1)
            self.stdTS[col] = np.sqrt(self.stdTS[col])

    def _calculateConfidenceLimits(self):
        if len(self._timeseries_list) == 0:
            print("***Cannot generate confidence limits unless isCollectTimeseries == True.")
            return
        for col in self.colnames:
            lowers = []
            uppers = []
            for i in range(len(self.lowerConfidenceTS)):
                values = [t[col][i]  for t in self._timeseries_list]
                lowers.append(np.percentile(values, self.confidenceLimits[0]))
                uppers.append(np.percentile(values, self.confidenceLimits[1]))
            self.lowerConfidenceTS[col] = np.array(lowers)
            self.upperConfidenceTS[col] = np.array(uppers)

    @classmethod
    def merge(cls, others):
        """
        Merges a colection of TimeseriesStatistic for the same
        shape of timeseries.

        Parameters
        ----------
        others: list-TimeseriesStatistic

        Returns
        -------
        TimeseriesStatistic
        """
        ts = others[0].self.sumTS
        result = TimeseriesStatistic(ts, confidenceLImits=self.confidenceLimits,
              isCollectTimeseries=self._isCollectTimeseries)
        for other in others:
            result.count += other.count
            result.sumTS += other.sumTS
            result.ssqTS += other.ssqTS
            if result.isCollectTimeseries:
                result.timeseries_list.extend(other.timeseries_list)
        return result
