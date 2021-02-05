# -*- coding: utf-8 -*-
"""
 Created on Tue Jul  7 14:24:09 2020

@author: hsauro
@author: joseph-hellerstein

A ModelFitter estimates parameters of a roadrunner model by using observed values
of floating species concentrations to construct fitted values with
small residuals (the difference between fitted and observed values).

Properties of interest are:
    fittedTS - estimated parameter values
    roadrunnerModel - roadrunner model object
    observedTS - observed values of floating species concentrations
    residualsTS - observedTS - fittedTS

Usage
-----
   # The constructor takes either a roadrunner or antimony model
   f = ModelFitter(model, "mydata.txt",
         parametersToFit=["k1", "k2"])
   # Fit the model parameters and view parameters
   f.fitModel()
   print(f.getFittedParameters())
   # Print observed, fitted and residual values
   print(f.observedTS)
   print(f.fittedTS)
   print(f.residualsTS)

The code is arranged as a hierarchy of classes that use the previous class:
    _modelFitterCore.ModelFitterCore - model fitting
    _modelFitterBootstrap.ModelFitterBootstrap - bootstrapping
    _modelFitterReport.ModelFitterReport - reports on results of fitting and bootstrapping
    modelFitter - plot routines
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME
import SBstoat._bootstrapResult as bsr
import SBstoat._plotOptions as po
from SBstoat._modelFitterReport import ModelFitterReport
from SBstoat.residualsAnalyzer import ResidualsAnalyzer

from docstring_expander.expander import Expander
import lmfit
import numpy as np
import pandas as pd


LOW_PERCENTILE = bsr.PERCENTILES[0]
HIGH_PERCENTILE = bsr.PERCENTILES[-1]


class ModelFitter(ModelFitterReport):

    # NOTE : The docstring for this method should contain all of the options
    #       in PlotOptions
    def plot(self, kind:str, params=None, numPoint=None, **kwargs):
        """
        High level plot routine.
        A figure may contain 1 or more plot, and each plot may contain
        1 or more curve. Figure (f), plot (p), and curve (c) are the possible
        scope of an option.  If the option is singled valued, then it applies
        to all instances of its scope. If it has multiple values, then the
        index of the value corresponds to the instance within the scope.
        A figure scope is always single valued.

        Parameters
        ----------
        kind: name of plot type
            FitAll: plots of fitted, observed data over time
            ParameterEstimatePairs: Pairwise plots of bootstrap estimates
            ParameterHistograms: histographs of bootstrap parameter values
            Residuals: plot residuals over time
            ResidualsAll: do all residuals plots
            ResidualsHistograms: histogram of residual values
            ResidualAutoCorrelations: Autocorrelations of residuals
            ResidualCrossCorrelations: Cross correlations of residuals
        alpha: (c) transparency; in [0, 1] (float)
        bins: (f) number of bins in a histogram plot (int)
        color: (c) color of the line (str, default="blue")
        columns: (f) list of columns to plot (list, default=[])
        figsize: (f) (horizontal width, vertical height) (list, default=[8, 6])
        legend: (p) tuple of str for legend (list)
        linestyle: (c) line style (str)
        marker: (c) marker for line (str)
        markersize: (c) size of marker for the line; >0 (float)
        num_row: (f) rows of plots (int)
        num_col: (f) columns of plots (int)
        subplot_width_space: (f) horizontal space between plots (float)
        timeseries2: (f) second timeseries
        title: (p) plot title (str)
        title_fontsize: (p) point size for title (float)
        title_position: (p) (x, y) relative position; x,y in [0, 1]
        suptitle: (f) figure title (str)
        xlabel: (f) x axis title (str)
        xlim: (f) order pair of lower and upper (list)
        xticklabels: (f) list of labels for x ticks (list)
        ylabel: (f) label for x axis (str)
        ylim: (f) order pair of lower and upper (str)
        yticklabels: (f) list of labels for y ticks (list)
        """
        MODULE_KINDS = [
            'ResidualsAll',
            'Residuals',
            'FitAll',
            'ParameterEstimatePairs',
            'ParameterHistograms',
            ]
        RESIDUAL_MODULE_KINDS = [
            'ResidualsHistograms',
            'ResidualAutoCorrelations',
            'ResidualCrossCorrelations',
            ]
        kinds = list(MODULE_KINDS)
        kinds.extend(RESIDUAL_MODULE_KINDS)
        if not kind in kinds:
            raise ValueError("Invalid plot kind. Options are: %s"
                             % str(kinds))
        if kind in MODULE_KINDS:
            statement = "self.plot%s(**kwargs)" % kind
            exec(statement)
        else:
            self._updateFit(params, numPoint)
            analyzer = ResidualsAnalyzer(self.observedTS, self._plotFittedTS,
                  meanFittedTS=self.bootstrapResult.fittedStatistic.meanTS,
                  stdFittedTS=self.bootstrapResult.fittedStatistic.stdTS,
                  residualsTS=self.residualsTS,
                  isPlot=self._isPlot)
            statement = "analyzer.plot%s(**kwargs)" % kind
            exec(statement)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8,
          header=po.HEADER)
    def plotResidualsAll(self,
          params:lmfit.Parameters=None, numPoint:int=None, **kwargs):
        """
        Plots a set of residual plots

        Parameters
        ----------
        #@expand
        """
        self._updateFit(params, numPoint)
        analyzer = ResidualsAnalyzer(self.observedTS, self._plotFittedTS,
              residualsTS=self.residualsTS,
              isPlot=self._isPlot)
        analyzer.plotAll(**kwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8,
          header=po.HEADER)
    def plotResiduals(self,
          params:lmfit.Parameters=None, numPoint:int=None, **kwargs):
        """
        Plots residuals of a fit over time.

        Parameters
        ----------
        #@expand
        """
        self._updateFit(params, numPoint)
        analyzer = ResidualsAnalyzer(self.observedTS, self._plotFittedTS,
              residualsTS=self.residualsTS,
              isPlot=self._isPlot)
        analyzer.plotResidualsOverTime(**kwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8, header=po.HEADER)
    def plotFitAll(self,
          params:lmfit.Parameters=None, numPoint:int=None, **kwargs):
        """
        Plots the fitted with observed data over time.

        Parameters
        ----------
        #@expand
        """
        self._updateFit(params, numPoint)
        bandLowTS = None
        bandHighTS = None
        if self.bootstrapResult is not None:
            statistic = self.bootstrapResult.simulate(numPoint=100, numSample=1000)
            self._plotFittedTS = statistic.meanTS
            bandLowTS = statistic.percentileDct[LOW_PERCENTILE]
            bandHighTS = statistic.percentileDct[HIGH_PERCENTILE]
        analyzer = ResidualsAnalyzer(self.observedTS, self._plotFittedTS,
              residualsTS=self.residualsTS,
              bandLowTS=bandLowTS, bandHighTS=bandHighTS,
              isPlot=self._isPlot)
        self._addKeyword(kwargs, po.NUM_ROW, 2)
        analyzer.plotFittedObservedOverTime(**kwargs)

    def _addKeyword(self, kwargs, key, value):
        if not key in kwargs:
            kwargs[key] = value

    def _mkParameterDF(self, parameters=None):
        df = pd.DataFrame(self.bootstrapResult.parameterDct)
        if parameters is not None:
            df = df[parameters]
        df.index.name = TIME
        return NamedTimeseries(dataframe=df)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8,
          header=po.HEADER)
    def plotParameterEstimatePairs(self, parameters=None, **kwargs):
        """
        Does pairwise plots of parameter estimates.

        Parameters
        ----------
        parameters: list-str
            List of parameters to do pairwise plots
        #@expand
        """
        if self.bootstrapResult is None:
            raise ValueError("Must run bootstrap before plotting parameter estimates.")
        ts = self._mkParameterDF(parameters=parameters)
        # Construct pairs
        names = list(self.bootstrapResult.parameterDct.keys())
        pairs = []
        compares = list(names)
        for name in names:
            compares.remove(name)
            pairs.extend([(name, c) for c in compares])
        #
        self._plotter.plotValuePairs(ts, pairs,
              isLowerTriangular=True, **kwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, includes=[po.BINS], indent=8,
          header=po.HEADER)
    def plotParameterHistograms(self, parameters=None, **kwargs):
        """
        Plots histographs of parameter values from a bootstrap.

        Parameters
        ----------
        parameters: list-str
            List of parameters to do pairwise plots
        #@expand
        """
        self._checkBootstrap()
        ts = self._mkParameterDF(parameters=parameters)
        self._plotter.plotHistograms(ts, **kwargs)

    def _updateFit(self, params:lmfit.Parameters, numPoint:int):
        """
        Checks and updates the fitted values useing the specified parameters
        and number of poitns.

        Parameters
        ----------

        Returns
        -------
        """
        self._checkFit()
        if self.fittedTS is None:
            self.fittedTS = self.simulate(
                  params=params, numPoint=len(self.observedTS))
            self.residualsTS = None
        if self.residualsTS is None:
            self.residualsTS = self.fittedTS.copy()
            cols = self.observedTS.colnames
            self.residualsTS[cols] = self.observedTS[cols] - self.fittedTS[cols]
            self.residualsTS[cols]  \
                  = np.nan_to_num(self.residualsTS[cols], nan=0.0)
        self._plotFittedTS = self.simulate(params=params, numPoint=numPoint)
        self._plotFittedTS = self._plotFittedTS.subsetColumns(self.selectedColumns)
