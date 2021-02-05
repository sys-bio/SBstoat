"""
 Created on Aug 20, 2020

@author: joseph-hellerstein

Codes that provide various analyses of residuals.

There are 3 types of timeseries: observed, fitted, and residuals
(observed - fitted).

Plots are organized by the timeseries and the characteristic analyzed. These
characteristics are: (a) over time, (b) histogram.
"""

from SBstoat.namedTimeseries import NamedTimeseries
import SBstoat._plotOptions as po
from SBstoat import timeseriesPlotter as tp

from docstring_expander.expander import Expander
import numpy as np

PLOT = "plot"


class ResidualsAnalyzer():

    def __init__(self, observedTS:NamedTimeseries, fittedTS:NamedTimeseries,
              residualsTS:NamedTimeseries=None, meanFittedTS=None,
              stdFittedTS=None,
              bandLowTS:NamedTimeseries=None, bandHighTS:NamedTimeseries=None,
              isPlot:bool=True):
        """
        Parameters
        ----------
        observedTS: Observed values
        residualsTS: same time values as observedTS
        fittedTS: fitted values
            may have different times than observedTS
        meanFittedTS: fitted values with same times as observedTS
        stdFittedTS: fitted values with same times as observedTS
        bandLowTS: timeseries that describes the lower band for timeseries1
        bandhighTS: timeseries that describes the higher band for timeseries1
        """
        self.observedTS = observedTS
        self.fittedTS = fittedTS
        self.meanFittedTS = meanFittedTS
        self.stdFittedTS = stdFittedTS
        self.bandLowTS = bandLowTS
        self.bandHighTS = bandHighTS
        if residualsTS is None:
            self.residualsTS = self.observedTS.copy()
            cols = self.residualsTS.colnames
            self.residualsTS[cols] -= self.fittedTS[cols]
            self.residualsTS[cols]  \
                  = np.nan_to_num(self.residualsTS[cols], nan=0.0)
        else:
            self.residualsTS = residualsTS.copy()
        ### Plotter
        self._plotter = tp.TimeseriesPlotter(isPlot=isPlot)

    @staticmethod
    def _addKeyword(kwargs:dict, key:str, value:object):
        if not key in kwargs:
            kwargs[key] = value

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8,
          header=po.HEADER)
    def plotAll(self, **kwargs:dict):
        """
        Does all residual plots.

        Parameters
        ----------
        #@expand
        """
        for name in dir(self):
            if name == "plotAll":
                continue
            if name[0:4] == PLOT:
                statement = "self.%s(**kwargs)" % name
                exec(statement)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8,
          header=po.HEADER)
    def plotResidualsOverTime(self, **kwargs:dict):
        """
        Plots residuals of a fit over time.

        Parameters
        ----------
        #@expand
        """
        ResidualsAnalyzer._addKeyword(kwargs, po.MARKER, "o")
        ResidualsAnalyzer._addKeyword(kwargs, po.SUPTITLE, "Residuals Over Time")
        self._plotter.plotTimeSingle(self.residualsTS, **kwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8,
          header=po.HEADER)
    def plotFittedObservedOverTime(self, **kwargs:dict):
        """
        Plots the fit with observed data over time.

        Parameters
        ----------
        #@expand
        """
        title = "Observed vs. fitted"
        if self.bandLowTS is not None:
            title += " (with shading for 95th percentile)"
        ResidualsAnalyzer._addKeyword(kwargs, po.SUPTITLE, title)
        ResidualsAnalyzer._addKeyword(kwargs, po.MARKER, [None, "o", "^"])
        legends = ["fitted", "observed"]
        if self.meanFittedTS is not None:
            legends.append("bootstrap fitted")
        ResidualsAnalyzer._addKeyword(kwargs, po.LEGEND, legends)
        ResidualsAnalyzer._addKeyword(kwargs, po.COLOR, ["b", "b", "r"])
        self._plotter.plotTimeSingle(
              self.fittedTS,
              timeseries2=self.observedTS,
              meanTS=self.meanFittedTS, stdTS=self.stdFittedTS,
              bandLowTS=self.bandLowTS,
              bandHighTS=self.bandHighTS,
              **kwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, includes=[po.BINS], indent=8,
          header=po.HEADER)
    def plotResidualsHistograms(self, **kwargs:dict):
        """
        Plots histographs of parameter values from a bootstrap.

        Parameters
        ----------
        parameters: List of parameters to do pairwise plots
        #@expand
        """
        ResidualsAnalyzer._addKeyword(kwargs, po.SUPTITLE, "Residual Distributions")
        self._plotter.plotHistograms(self.residualsTS, **kwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8,
          header=po.HEADER)
    def plotResidualsAutoCorrelations(self, **kwargs:dict):
        """
        Plots auto correlations between residuals of columns.

        Parameters
        ----------
        parameters: List of parameters to do pairwise plots
        #@expand
        """
        ResidualsAnalyzer._addKeyword(kwargs, po.SUPTITLE, "Residual Autocorrelations")
        self._plotter.plotAutoCorrelations(self.residualsTS, **kwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8,
          header=po.HEADER)
    def plotResidualCrossCorrelations(self, **kwargs:dict):
        """
        Plots cross correlations between residuals of columns.

        Parameters
        ----------
        parameters: List of parameters to do pairwise plots
        #@expand
        """
        ResidualsAnalyzer._addKeyword(kwargs, po.SUPTITLE, "Residual Cross Correlations")
        self._plotter.plotCrossCorrelations(self.residualsTS, **kwargs)
