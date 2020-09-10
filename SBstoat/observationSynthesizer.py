# -*- coding: utf-8 -*-
"""
 Created on August 30, 2020

@author: joseph-hellerstein

Creates synthetic observations.
"""

from SBstoat.namedTimeseries import NamedTimeseries

import abc
import numpy as np
import typing


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
        self._observedTS = observedTS
        self._fittedTS = fittedTS
        self._residualsTS = residualsTS
        for ts in [self._observedTS, self._fittedTS, self._residualsTS]:
            if ts is not None:
                self.columns = ts.colnames

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
        return self._residualsTS

    @abc.abstractmethod
    def calculate(self):
        """
        Calculates a set of synthetic observations
        """
        pass


class ObservationSynthesizerRandomizedResiduals(ObservationSynthesizer):

    def calculate(self):
        """
        Calculates synthetic observations by randomly rearranging residuals.
        """
        numRow = len(self.observedTS)
        newObservedTS = self.fittedTS.copy()
        for column in self.columns:
            newObservedTS[column] += np.random.choice(
                self.residualsTS[column],numRow, replace=True)
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
