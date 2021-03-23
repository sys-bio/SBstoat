# -*- coding: utf-8 -*-
"""
 Created on August 30, 2020

@author: joseph-hellerstein

Creates synthetic observations.

TODO:
  1. Remove outliers via std criteria
  2. Handling missing observerations in observedTS
"""

from SBstoat.namedTimeseries import NamedTimeseries
from SBstoat import _helpers

import abc
import numpy as np
import pandas as pd


class ObservationSynthesizer(abc.ABC):


    def __init__(self, observedTS:NamedTimeseries=None,
                 fittedTS:NamedTimeseries=None,
                 residualsTS:NamedTimeseries=None):
        """
        Child classes must specify two of the three timeseries.

        Parameters
        ----------
        observedTS: Observations
        fittedTS: Fitted values
        residualsTS: Residual values
        """
        def getTS(data):
            if isinstance(data, pd.DataFrame):
                return NamedTimeseries(dataframe=data)
            return data
        #
        self._observedTS = getTS(observedTS)
        self._fittedTS = getTS(fittedTS)
        self._residualsTS = getTS(residualsTS)
        if self._observedTS is not None:
            self._observedTS = self._observedTS.copy()
        if self._fittedTS is not None:
            self._fittedTS = self._fittedTS.copy()
        if self._residualsTS is not None:
            self._residualsTS = self._residualsTS.copy()
        # Set the columns based on the information provided
        for ts in [self._observedTS, self._fittedTS, self._residualsTS]:
            if ts is not None:
                self.columns = ts.colnames
                break
        np.random.seed()  # Ensure different sequences on each invocation

    @property
    def observedTS(self):
        if self._observedTS is None:
            self._observedTS = self._fittedTS.copy()
            self._observedTS[self.columns] += self._residualsTS[self.columns]
        return self._observedTS

    @property
    def fittedTS(self):
        if self._fittedTS is None:
            self._fittedTS = self._observedTS.copy()
            self._fittedTS[self.columns] -= self._residualsTS
        return self._fittedTS

    @property
    def residualsTS(self):
        if self._residualsTS is None:
            self._residualsTS = self._observedTS.copy()
            self._residualsTS[self.columns] -=  \
                self._fittedTS[self.columns]
            self._residualsTS[self.columns]  \
                  = np.nan_to_num(self._residualsTS[self.columns], nan=0.0)
        return self._residualsTS

    @abc.abstractmethod
    def calculate(self):
        """
        Calculates a set of synthetic observations
        """
        pass


class ObservationSynthesizerRandomizedResiduals(ObservationSynthesizer):

    def __init__(self, observedTS:NamedTimeseries=None,
                 fittedTS:NamedTimeseries=None,
                 residualsTS:NamedTimeseries=None,
                 filterSL:float=0.01):
        """
        Parameters
        ----------
        filterSL: maximum significance level used in filtering residuals

        Notes
        -----
        observedTS and fittedTS must be non-Null.
        """
        super().__init__(observedTS=observedTS, fittedTS=fittedTS,
                 residualsTS=residualsTS)
        self._filterSL = filterSL
        self._filteredResidualsDct = self._calcFilteredResidualsDct()

    def _calcFilteredResidualsDct(self):
        # Constructs dictionary of residuals meeting std criteria
        dct = {}
        for col in self._observedTS.colnames:
            if self._filterSL is None:
                dct[col] = self.residualsTS[col]
            else:
                dct[col] = _helpers.filterOutliersFromZero(
                      self.residualsTS[col], self._filterSL)
        return dct

    def calculate(self):
        """
        Calculates synthetic observations by randomly rearranging residuals.
        """
        numRow = len(self.observedTS)
        newObservedTS = self.fittedTS.copy()
        allIdxs = np.random.randint(0, numRow, len(self.columns)*numRow)
        for idx, column in enumerate(self.columns):
            if len(self._filteredResidualsDct[column]) == 0:
                msg = "No residuals left after filtering."
                msg += " Make filter less restrictive."
                raise ValueError(msg)
            newObservedTS[column] += np.random.choice(
                self._filteredResidualsDct[column], numRow, replace=True)
        return newObservedTS


class ObservationSynthesizerRandomErrors(ObservationSynthesizer):
    """
    Calculates residuals using randomized errors from a normal
    distribution.
    """

    def __init__(self, observedTS=None,
              fittedTS:NamedTimeseries=None, std:float=0.1):
        """
        Parameters
        ----------
        std: standard deviation for random numbers
        """
        if fittedTS is None:
            raise ValueError("Must supply fitted timeseries")
        self.std = std
        super().__init__(fittedTS=fittedTS)

    def calculate(self):
        """
        Calculates synthetic observations by randomly rearranging residuals.
        """
        numRow = len(self.fittedTS)
        self._residualsTS = self._fittedTS.copy()
        for col in self.columns:
            self._residualsTS[col] = np.random.normal(
                0, self.std, numRow)
        return self.observedTS
