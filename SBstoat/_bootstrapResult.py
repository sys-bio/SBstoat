# -*- coding: utf-8 -*-
"""
 Created on August 18, 2020

@author: joseph-hellerstein

Container for the results of bootstrapping. Provides
metrics that are calculated from the results.
"""

import SBstoat
from SBstoat.timeseriesStatistic import TimeseriesStatistic
from SBstoat.logs import Logger
from SBstoat import rpickle
from SBstoat import _helpers

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
        fitter: Fitter
        numIteration: number of successful iterations
        parameterDct: dict
            key: parameter name
            value: list of values
        fittedStatistic: statistics for fitted timeseries
        err: Error encountered
        """
        self.fitter = None
        self.numIteration = numIteration
        self.parameterDct = dict(parameterDct)
        self.bootstrapError = bootstrapError
        self.numSimulation = 0
        self.parameterMeanDct = {}
        # Timeseries statistics for fits
        self.fittedStatistic = fittedStatistic
        if fitter is None:
            self.logger = Logger()
        else:
            self.logger = fitter.logger
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

    def setFitter(self, fitter):
        self.fitter = fitter
        self.fitter.logger = self.logger

    def copyFitter(self, fitter):
        """
        Creates a copy of the model fitter for bootstrap result.

        Parameters
        ----------
        fitter: ModelFitter

        Returns
        -------
        ModelFitter
        """
        newModelFitter = fitter.__class__(
              fitter.modelSpecification,
              fitter.observedTS,
              fitter.parametersToFit,
              selectedColumns=fitter.selectedColumns,
              fitterMethods=fitter._fitterMethods,
              bootstrapMethods=fitter._bootstrapMethods,
              parameterLowerBound=fitter.parameterLowerBound,
              parameterUpperBound=fitter.parameterUpperBound,
              logger=fitter.logger,
              isPlot=fitter._isPlot)
        return newModelFitter

    def copy(self, isCopyParams=False):
        """
        Create a pickle-able copy.

        Returns
        -------
        BootstrapResult
        """
        newResult = BootstrapResult(None, self.numIteration, self.parameterDct,
          self.fittedStatistic, bootstrapError=self.bootstrapError)
        if self._params is not None:
            if isCopyParams:
                newResult._params = self._params.copy()
        return newResult

    @classmethod
    def rpConstruct(cls):
        """
        Overrides rpickler.rpConstruct to create a method that
        constructs an instance without arguments.

        Returns
        -------
        Instance of cls
        """
        return cls(None, 0, {}, None)

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
        if self.fitter is None:
            raise RuntimeError("Must use setFitter before running simulation.")
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
        if self.fitter is None:
            raise RuntimeError("Must use setFitter before running simulation.")
        df = pd.DataFrame(self.parameterDct)
        df_sample = df.sample(numSample, replace=True, axis=0)
        dcts = df_sample.to_dict('records')
        results = []
        for dct in dcts:
            parametersToFit = []
            for name in dct.keys():
                value = dct[name]
                parametersToFit.append(SBstoat.Parameter(name,
                      lower=value*0.9, value=value, upper=value*1.1))
            params = self.fitter.mkParams(parametersToFit)
            results.append(params)
        return results

    @classmethod
    def merge(cls, bootstrapResults, fitter=None):
        """
        Combines a list of BootstrapResult. Sets a fitter.

        Parameter
        ---------
        bootstrapResults: list-BootstrapResult
        fitter: ModelFitter

        Return
        ------
        BootstrapResult
        """
        if len(bootstrapResults) == 0:
            raise ValueError("Must provide a non-empty list")
        parameterDct = {p: [] for p in bootstrapResults[0].parameterDct}
        numIteration = sum([r.numIteration for r in bootstrapResults])
        bootstrapError = sum([b.bootstrapError for b in bootstrapResults])
        fittedStatistic = None
        if numIteration > 0:
            # Merge the logs
            logger = Logger.merge([b.logger for b in bootstrapResults])
            # Merge the statistics for fitted timeseries
            fittedStatistics = [b.fittedStatistic for b in bootstrapResults
                  if b.fittedStatistic is not None]
            fittedStatistic = TimeseriesStatistic.merge(fittedStatistics)
            # Accumulate the results
            for bootstrapResult in bootstrapResults:
                for parameter in parameterDct.keys():
                    parameterDct[parameter].extend(
                          bootstrapResult.parameterDct[parameter])
            #
        fitter.logger = Logger.merge([fitter.logger, logger])
        bootstrapResult = BootstrapResult(fitter, numIteration, parameterDct,
              fittedStatistic, bootstrapError=bootstrapError)
        bootstrapResult.setFitter(fitter)
        return bootstrapResult
