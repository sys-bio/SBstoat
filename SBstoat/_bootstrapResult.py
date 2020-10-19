# -*- coding: utf-8 -*-
"""
 Created on August 18, 2020

@author: joseph-hellerstein

Container for the results of bootstrapping. Provides
metrics that are calculated from the results.
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME, mkNamedTimeseries
from SBstoat.timeseriesStatistic import TimeseriesStatistic
import SBstoat._modelFitterCore as mfc
from SBstoat import _helpers

import lmfit
import numpy as np
import pandas as pd
import typing

PERCENTILES = [2.5, 97.55]  # Percentile for confidence limits
# Timeseries columns
COL_SUM = "sum"  # sum of fitted values
COL_SSQ = "ssq"  # sum of squares


######### FUNCTIONS ###############
def sampleDct(dct:dict, numSample:int):
    """
    Samples the values in a dictionary with lists of the same length.

    Parameters
    ----------
    dct: dictionary of values to sample
    numSample: number of samples returned
    
    Returns
    -------
    dict
    """
    df = pd.DataFrame(dct)
    df_sample = df.sample(numSample, replace=True, axis=0)
    return df_sample.to_dict()


######### CLASSES ###############
class BootstrapResult():

    def __init__(self, fitter, numIteration: int, parameterDct:dict,
          timeseriesStatistic: TimeseriesStatistic):
        """
        Results from bootstrap

        Parameters
        ----------
        numIteration: number of successful iterations
        parameterDct: dict
            key: parameter name
            value: list of values
        timeseriesStatistic: statistics for fitted timeseries
        """
        self.fitter = fitter.copy()
        self.numIteration = numIteration
        # population of parameter values
        self.parameterDct = dict(parameterDct)
        # list of parameters
        self.parameters = list(self.parameterDct.keys())
        # Number of simulations
        self.numSimulation =  \
              len(self.parameterDct[self.parameters[0]])
        # means of parameter values
        self.meanDct = {p: np.mean(parameterDct[p])
              for p in self.parameters}
        # standard deviation of parameter values
        self.stdDct = {p: np.std(parameterDct[p])
              for p in self.parameters}
        # 95% Confidence limits for parameter values
        self.percentileDct = {
              p: np.percentile(self.parameterDct[p],
              PERCENTILES) for p in self.parameterDct}
        # Timeseries statistics for fits
        self.timeseriesStatistic = timeseriesStatistic
        ### PRIVATE
        # Fitting parameters from result
        self._params = None
        self._meanBootstrapFittedTS = None
        self._stdBootstrapFittedTS = None

    def __str__(self) -> str:
        """
        Bootstrap report.       
        """
        report = _helpers.Report()
        report.addHeader("Bootstrap Report.")
        report.addTerm("Total iterations", self.numIteration)
        report.addTerm("Total simulation", self.numSimulation)
        for par in self.parameters:
            report.addHeader(par)
            report.indent(1)
            report.addTerm("mean", self.meanDct[par])
            report.addTerm("std", self.stdDct[par])
            report.addTerm("%s Percentiles" % str(PERCENTILES),
                  self.percentileDct[par])
            report.indent(-1)
        return report.get()

    @property
    def params(self)->lmfit.Parameters:
        """
        Constructs parameters from bootstrap result.
        
        Returns
        -------
        """
        if not "_params" in self.__dict__.keys():
            self._params = None
        if self._params is None:
            self._params = lmfit.Parameters()
            for name in self.meanDct.keys():
                value = self.meanDct[name]
                self._params.add(name, value=value, min=value*0.99,
                      max=value*1.01)
        return self._params

    @property
    def meanBootstrapFittedTS(self)->NamedTimeseries:
        """
        Mean of fitted values.
        """
        if self._meanBootstrapFittedTS is None:
            self.timeseriesStatistic.calculate()
            self._meanBootsrapFittedTS = self.timeseriesStatistic.meanTS
            self._stdBootsrapFittedTS = self.timeseriesStatistic.stdTS
        return self._meanBootstrapFittedTS

    def getSimulatedStatistic(self, numSample:int=1000,
           numPoint:int=100)->TimeseriesStatistic:
        """
        Standard deviation of fitted values using the bootstrap estimates.
        This is obtained by doing simulations with different sampled parameters.
        
        Parameters
        ----------
        numSample: number of fitted parameters to sample
        numPoint: number of points in the simulation
        
        Returns
        -------
        TimeseriesStatistic
        """
        timeseriesStatistic = TimeseriesStatistic(self.fitter.observedTS,
              isCollectTimeseries=True)
        for idx in range(numSample):
            fittedTS = mfb.simulate(params=params, numPoint=numPoint)
            timeseriesStatistic.accumulate(fittedTS)
        timeseriesStatistic.calculate()
        return timeseriesStatistic

    @property
    def stdBootstrapFittedTS(self)->NamedTimeseries:
        """
        Standard deviation of fitted values.
        """
        if self._stdBootstrapFittedTS is None:
            self.timeseriesStatistic.calculate()
            self._stdBootsrapFittedTS = self.timeseriesStatistic.stdTS
            self._meanBootsrapFittedTS = self.timeseriesStatistic.meanTS
        return self._stdBootstrapFittedTS

    @classmethod
    def merge(cls, bootstrapResults):
        """
        Combines a list of BootstrapResult.

        Parameter
        ---------
        bootstrapResults: list-BootstrapResult

        Return
        ------
        BootstrapResult
        """
        if len(bootstrapResults) == 0:
            raise ValueError("Must provide a non-empty list")
        statisticDct = bootstrapResults[0].statisticDct
        fitter = bootstrapResults[0].fitter
        numIteration = sum([r.numIteration for r in bootstrapResults])
        # Accumulate the results
        parameterDct = {p: [] for p in bootstrapResults[0].parameterDct}
        zeroTS = statisticDct[COL_SUM].copy(isInitialize=True)
        colnames = zeroTS.colnames
        statisticDct = {c: zeroTS.copy() for c in [COL_SUM, COL_SSQ]}       
        for bootstrapResult in bootstrapResults:
            for parameter in parameterDct.keys():
                parameterDct[parameter].extend(
                      bootstrapResult.parameterDct[parameter])
        timeseriesStatistic = TimeseriesStatistic.merge(
              [r.timeseriesStatistic for r in bootstrapResults])
        return BootstrapResult(fitter, numIteration, parameterDct, timeseriesStatistic)
