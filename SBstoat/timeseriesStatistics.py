"""Calculation of statistics for time series"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME

import numpy as np


class TimeseriesStatistic(object):

    def __init__(self, prototypeTS:NamedTimeseries):
        """
        Parameters
        ----------
        prototypeTS: same length and columns as desired
        """
        self.colnames = prototypeTS.colnames
        self.sumTS = prototypeTS.copy(isInitialize=True)
        self.ssqTS = prototypeTS.copy(isInitialize=True)
        # Statistic
        self.count = 0  # Count of timeseries accumulated
        self.meanTS = prototypeTS.copy(isInitialize=True) # means
        self.stdTS = prototypeTS.copy(isInitialize=True)  # standard deviations

    def accumulate(self, newTS:NamedTimeseries):
        """
        Accumulates statistics for a new timeseries.
        """
        self.sumTS[self.colnames] = self.sumTS[self.colnames]  \
               + newTS[self.colnames]
        self.ssqTS[self.colnames] = self.ssqTS[self.colnames]  \
               + newTS[self.colnames]**2
        self.count += 1

    def calculate(self):
        """
        Calculates statistics.
        """
        if self.count <= 1:
            raise ValueError("Must accumulate at leat 2 timeseries before calculating statistics.")
        # Calculate mean
        for col in self.colnames:
            self.meanTS[col] = (1.0*self.sumTS[col]) / self.count
        # Calculate standard deviations
        for col in self.colnames:
            self.stdTS[col] = self.ssqTS[col] - self.count*self.meanTS[col]**2
            self.stdTS[col] = self.stdTS[col] / (self.count - 1)
            self.stdTS[col] = np.sqrt(self.stdTS[col])

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
        result = TimeseriesStatistic(ts)
        for other in others:
            result.count += other.count
            result.sumTS += other.sumTS
            result.ssqTS += other.ssqTS
        return result
        
