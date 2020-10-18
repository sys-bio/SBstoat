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

from SBstoat.namedTimeseries import NamedTimeseries, TIME

import copy
import numpy as np
import typing


PERCENTILES = [5.0, 95.0]  # Percentiles calculated: lower, upper


class TimeseriesStatistic(object):

    def __init__(self, prototypeTS:NamedTimeseries,
          confidenceLimits:typing.Tuple[float,float]=PERCENTILES,
          isCollectTimeseries:bool=True):
        """
        Parameters
        ----------
        prototypeTS: same length and columns as desired
        confidenceLimits: Lower and upper limits of confidence limits
            for the timeseries accumulated)
        isCollectTimeseries: Must be enabled to calculate confidence limits
        """
        self.prototypeTS = prototypeTS
        self.colnames = self.prototypeTS.colnames
        self.sumTS = self.prototypeTS.copy(isInitialize=True)
        self.ssqTS = self.prototypeTS.copy(isInitialize=True)
        self.confidenceLimits = confidenceLimits
        self._isCollectTimeseries = isCollectTimeseries
        self._timeseries_list = []
        # Statistics
        self.count = 0  # Count of timeseries accumulated
        # Means
        self.meanTS = prototypeTS.copy(isInitialize=True) # means
        # Standard deviations
        self.stdTS = prototypeTS.copy(isInitialize=True)  # standard deviations
        # Lower bound of confidence interval
        self.lowerPercentileTS = prototypeTS.copy(isInitialize=True)
        # Upper bound of confidence interval
        self.upperPercentileTS = prototypeTS.copy(isInitialize=True)

    def copy(self):
        """
        Makes a copy of the object, including internal state.
        
        Returns
        -------
        TimeseriesStatistic
        """
        newStatistic = TimeseriesStatistic(self.prototypeTS,
              confidenceLimits=self.confidenceLimits,
              isCollectTimeseries=self._isCollectTimeseries)
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
        self._calculatePercentileLimits()

    def _calculateMean(self):
        for col in self.colnames:
            self.meanTS[col] = (1.0*self.sumTS[col]) / self.count

    def _calculateStd(self):
        for col in self.colnames:
            self.stdTS[col] = self.ssqTS[col] - self.count*self.meanTS[col]**2
            self.stdTS[col] = self.stdTS[col] / (self.count - 1)
            self.stdTS[col] = np.sqrt(self.stdTS[col])

    def _calculatePercentileLimits(self):
        if len(self._timeseries_list) == 0:
            print("***Cannot generate confidence limits unless isCollectTimeseries == True.")
            return
        for col in self.colnames:
            lowers = []
            uppers = []
            for i in range(len(self.lowerPercentileTS)):
                values = [t[col][i]  for t in self._timeseries_list]
                lowers.append(np.percentile(values, self.confidenceLimits[0]))
                uppers.append(np.percentile(values, self.confidenceLimits[1]))
            self.lowerPercentileTS[col] = np.array(lowers)
            self.upperPercentileTS[col] = np.array(uppers)

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
            if result._isCollectTimeseries:
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
