# -*- coding: utf-8 -*-
"""
 Created on August 18, 2020

@author: joseph-hellerstein

Container for the results of bootstrapping. Provides
metrics that are calculated from the results.
"""

from SBstoat.timeseriesStatistic import TimeseriesStatistic
from SBstoat.logs import Logger
from SBstoat import rpickle
from SBstoat import _helpers
from SBstoat import _modelFitterCore as mfc

import copy
import lmfit
import numpy as np
import pandas as pd

PERCENTILES = [2.5, 50, 97.55]  # Percentile for confidence limits
MIN_COUNT_PERCENTILE = 100  # Minimum number of values required to get percentiles
MIN_VALUE = 1e-3


######### CLASSES ###############
class BootstrapResult(rpickle.RPickler):

    def __init__(self, fitter, numIteration: int, parameterDct:dict,
          fittedStatistic: TimeseriesStatistic, bootstrapError=0):
        """
        Results from bootstrap

        Parameters
        ----------
        numIteration: number of successful iterations
        parameterDct: dict
            key: parameter name
            value: list of values
        fittedStatistic: statistics for fitted timeseries
        err: Error encountered
        """
        self.fitter = fitter
        self.numIteration = numIteration
        self.parameterDct = parameterDct
        self.bootstrapError = bootstrapError
        # Timeseries statistics for fits
        self.fittedStatistic = fittedStatistic
        if fitter is not None:
            self.fitter = self.fitter.copy()
            # population of parameter values
            self.parameterDct = dict(self.parameterDct)
            # list of parameters
            self.parameters = list(self.parameterDct.keys())
            # Number of simulations
            if len(self.parameters) > 0:
                self.numSimulation =  \
                      len(self.parameterDct[self.parameters[0]])
            else:
                self.numSimulation = 0
            if self.numSimulation > 1:
                # means of parameter values
                self.parameterMeanDct = {p: np.mean(parameterDct[p])
                      for p in self.parameters}
                # standard deviation of parameter values
                self.parameterStdDct = {p: np.std(parameterDct[p])
                      for p in self.parameters}
                # Confidence limits for parameter values
                self.percentileDct = {p: [] for p in self.parameterDct}
                for name, values in self.parameterDct.items():
                    if len(values) > MIN_COUNT_PERCENTILE:
                        self.percentileDct[name] = np.percentile(values, PERCENTILES)
            else:
                # means of parameter values
                self.parameterMeanDct = {p: np.nan for p in self.parameters}
                # standard deviation of parameter values
                self.parameterStdDct = {p: np.nan for p in self.parameters}
                # Confidence limits for parameter values
                self.percentileDct = {p: np.nan for p in self.parameters}
        ### PRIVATE
        # Fitting parameters from result
        self._params = None

    @classmethod
    def rpConstruct(cls):
        """
        Overrides rpickler.rpConstruct to create a method that
        constructs an instance without arguments.

        Returns
        -------
        Instance of cls
        """
        return cls(None, None, None, None)

    def copy(self):
        return copy.deepcopy(self)

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
            report.addTerm("mean", self.parameterMeanDct[par])
            report.addTerm("std", self.parameterStdDct[par])
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
        if "_params" not in self.__dict__.keys():
            self._params = None
        if self._params is None:
            self._params = lmfit.Parameters()
            for name in self.parameterMeanDct.keys():
                value = self.parameterMeanDct[name]
                if np.isclose(value, 0):
                    minValue = -MIN_VALUE
                    maxValue = MIN_VALUE
                else:
                    values = (value*0.99, value*1.01)
                    minValue = min(values)
                    maxValue = max(values)
                self._params.add(name, value=value, min=minValue, max=maxValue)
        return self._params

    def simulate(self, numSample:int=1000, numPoint:int=100)->TimeseriesStatistic:
        """
        Runs a simulation using the parameters from the bootstrap.

        Parameters
        ----------
        numSample: number of fitted parameters to sample
        numPoint: number of points in the simulation

        Returns
        -------
        TimeseriesStatistic
        """
        params_list = self._sampleParams(numSample)
        fittedTS = self.fitter.simulate(params=params_list[0], numPoint=numPoint)
        timeseriesStatistic = TimeseriesStatistic(fittedTS,
              percentiles=PERCENTILES)
        timeseriesStatistic.accumulate(fittedTS)
        # Do the simulation
        for params in params_list[1:]:
            fittedTS = self.fitter.simulate(params=params, numPoint=numPoint)
            timeseriesStatistic.accumulate(fittedTS)
        timeseriesStatistic.calculate()
        return timeseriesStatistic

    def _sampleParams(self, numSample:int):
        """
        Samples the parameters obtained in bootstrapping.

        Parameters
        ----------
        numSample: number of samples returned

        Returns
        -------
        list-lmfit.Parameters
        """
        df = pd.DataFrame(self.parameterDct)
        df_sample = df.sample(numSample, replace=True, axis=0)
        dcts = df_sample.to_dict('records')
        results = []
        for dct in dcts:
            parameterDct = {}
            for name in dct.keys():
                value = dct[name]
                parameterDct[name] = mfc.ParameterSpecification(
                      lower=value*0.9, value=value, upper=value*1.1)
            params = self.fitter.mkParams(parameterDct=parameterDct)
            results.append(params)
        return results

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
        fitter = bootstrapResults[0].fitter
        parameterDct = {p: [] for p in bootstrapResults[0].parameterDct}
        numIteration = sum([r.numIteration for r in bootstrapResults])
        bootstrapError = sum([b.bootstrapError for b in bootstrapResults])
        fittedStatistic = None
        if numIteration > 0:
            # Merge the fitter logs
            fitter.logger = Logger.merge([b.fitter.logger
                  for b in bootstrapResults])
            # Merge the statistics for fitted timeseries
            fittedStatistics = [b.fittedStatistic for b in bootstrapResults]
            fittedStatistic = TimeseriesStatistic.merge(fittedStatistics)
            # Accumulate the results
            for bootstrapResult in bootstrapResults:
                for parameter in parameterDct.keys():
                    parameterDct[parameter].extend(
                          bootstrapResult.parameterDct[parameter])
            #
        return BootstrapResult(fitter, numIteration, parameterDct,
              fittedStatistic, bootstrapError=bootstrapError)
