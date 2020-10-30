"""
Calculation of statistics for time series. Usage:
    statistic = TimeseriesStatistic(baseTimeseries)
    statistic.accumulate(timeseries_1)
    statistic.accumulate(timeseries_2)
    statistic.calculate()
    print(statistic.count)  # Number of timeseries accumulated
    print(statistic.meanTS)  # Timeseries of mean values
    print(statistic.stdTS)  # Timeseries of std values
    print(statistic.lowerPercentileTS)  # Timeseries of lower 5% at each time
    print(statistic.upperPercentileTS)  # Timeseries of lower 95% at each time
"""

from SBstoat.namedTimeseries import NamedTimeseries
from SBstoat import rpickle

import copy
import numpy as np
import typing


PERCENTILES = [5.0, 50.0, 95.0]  # Percentiles calculated


class TimeseriesStatistic(rpickle.RPickler):

    def __init__(self, prototypeTS:NamedTimeseries,
          percentiles:list=PERCENTILES):
        """
        Parameters
        ----------
        prototypeTS: same length and columns as desired
        percentiles: percentiles to calculate for accumulated Timeseries
        """
        # Statistics
        if prototypeTS is not None:
            self.count = 0  # Count of timeseries accumulated
            self.prototypeTS = prototypeTS
            self.colnames = self.prototypeTS.colnames
            self.sumTS = self.prototypeTS.copy(isInitialize=True)
            self.ssqTS = self.prototypeTS.copy(isInitialize=True)
            self.percentiles = percentiles
            self._timeseries_list = []  # List of timeseries accumulated
            # Means
            self.meanTS = prototypeTS.copy(isInitialize=True) # means
            # Standard deviations
            self.stdTS = prototypeTS.copy(isInitialize=True)  # standard deviations
            # Percentiles
            self.percentileDct = {}  # Key: percentile; Value: Timeseries
        else:
            # rpConstruct initializations.
            pass
    
    @classmethod
    def rpConstruct(cls):
        """
        Overrides rpickler.rpConstruct to create a method that
        constructs an instance without arguments.
        
        Returns
        -------
        Instance of cls
        """
        return cls(None)

    def copy(self):
        """
        Makes a copy of the object, including internal state.
        
        Returns
        -------
        TimeseriesStatistic
        """
        newStatistic = TimeseriesStatistic(self.prototypeTS,
              percentiles=self.percentiles)
        # Update internal state
        for attr in self.__dict__.keys():
            if "__" in attr:
                continue
            expression = "dir(self.%s)" % attr
            if "copy" in eval(expression):
                statement = "newStatistic.%s = self.%s.copy()" % (attr, attr)
                exec(statement)
            else:
                statement = "newStatistic.%s = copy.deepcopy(self.%s)" % (attr, attr)
                exec(statement)
        return newStatistic

    def equals(self, other):
        """
        Checks if the two instances have the same values.

        Parameters
        ----------
        other: TimeseriesStatistic
        
        Returns
        -------
        bool
        """
        isEqual = True
        for attr in self.__dict__.keys():
            if "__" in attr:
                continue
            expression = "dir(self.%s)" % attr
            if "equals" in eval(expression):
                expression = "self.%s.equals(other.%s)" % (attr, attr)
            else:
                expression = "self.%s == other.%s" % (attr, attr)
            isEqual = isEqual and eval(expression)
        return isEqual

    def accumulate(self, newTS:NamedTimeseries):
        """
        Accumulates statistics for a new timeseries.
        """
        self.sumTS[self.colnames] = self.sumTS[self.colnames]  \
               + newTS[self.colnames]
        self.ssqTS[self.colnames] = self.ssqTS[self.colnames]  \
               + newTS[self.colnames]**2
        if len(self.percentiles) > 0:
            self._timeseries_list.append(newTS)
        self.count += 1

    def calculate(self):
        """
        Calculates statistics.
        """
        self._calculateMean()
        if self.count > 1:
            self._calculateStd()
        self._calculatePercentiles()

    def _calculateMean(self):
        for col in self.colnames:
            self.meanTS[col] = (1.0*self.sumTS[col]) / self.count

    def _calculateStd(self):
        for col in self.colnames:
            if self.count > 1:
                self.stdTS[col] = self.ssqTS[col] - self.count*self.meanTS[col]**2
                self.stdTS[col] = self.stdTS[col] / (self.count - 1)
                self.stdTS[col] = np.array([
                      0.0 if np.isclose(v, 0.0) else v for v in self.stdTS[col]])
                self.stdTS[col] = np.sqrt(self.stdTS[col])
            else:
                pass

    def _calculatePercentiles(self):
        if len(self.percentiles) == 0:
            return
        self.percentileDct = {}
        refTS = self._timeseries_list[0]
        indices = range(len(refTS))
        for percentile in self.percentiles:
            percentileTS = refTS.copy(isInitialize=True)
            for col in self.colnames:
                col_values = []
                for idx in indices:
                    time_values = [t[col][idx]  for t in self._timeseries_list]
                    col_values.append(np.percentile(time_values, percentile))
                percentileTS[col] = np.array(col_values)
            self.percentileDct[percentile] = percentileTS

    def _merge(self, others):
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
        result = self.copy()
        for other in others:
            result.count += other.count
            result.sumTS[result.colnames] += other.sumTS[result.colnames]
            result.ssqTS[result.colnames] += other.ssqTS[result.colnames]
            result._timeseries_list.extend(other._timeseries_list)
        return result

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
        statistic = others[0]
        newOthers = list(others[1:])
        return statistic._merge(newOthers)
