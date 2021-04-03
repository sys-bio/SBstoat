#!/usr/bin/env python
# coding: utf-8
"""
Functions and Classes used for cross validation.
Usage:
    fitter = SuiteFitter(modelSpecifications, dataSources, parametersCollection)
    fitter.crossValidate(5)  # Do cross validation with 5 folds
    fitter.scoreDF  # Dataframe with the scores by fold
    fitter.parameterDF  # Dataframe with parameter mean, std
"""

import SBstoat._constants as cn
from SBstoat.abstractCrossValidator import AbstractCrossValidator, AbstractFitter
from SBstoat._suiteFitterCore import SuiteFitterCore

import copy
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


##################### CLASSES ###########################
class Fitter(AbstractFitter):

    def __init__(self, modelSpecification, observedTS, parametersToFit,
          trainIdxs=None, testIdxs=None,  **kwargs):
        """
        Parameters
        ----------
        modelSpecification: str
            antimony model
        observedTS: NamedTimeseries
        parametersToFit: list-str/SBstoat.Parameter/None
            parameters in the model that you want to fit
            if None, no parameters are fit
        trainIdxs: list-int
            rows of observedData used for estimating parameters
        testIdxs: list-int
            rows of observedData used for scoring fit
        """
        super().__init__()
        self.trainIdxs = trainIdxs
        self.testIdxs = testIdxs
        self.observedTS = observedTS
        if self.trainIdxs is None:
            self.trainIdxs = list(range(len(self.observedTS)))
        if self.testIdxs is None:
            self.testIdxs = list(range(len(self.observedTS)))
        self.trainTS = observedTS[self.trainIdxs]
        self.testTS = observedTS[self.testIdxs]
        # FIXME
        #self.modelFitter = ModelFitterCore(modelSpecification, self.trainTS,
        #      parametersToFit, **kwargs)
        self.columns = self.modelFitter.selectedColumns
        self.testObservedArr = self.testTS[self.columns]
    
    @property
    def parameters(self):
        """
        Returns
        -------
        lmfit.Parameters
        """
        return self.modelFitter.params

    def fit(self):
        """
        Estimates parameters.
        Parameters
        ----------
        
        Returns
        -------
        """
        self.modelFitter.fitModel()

    def score(self):
        """
        Returns an R^2 for predicting testData
        Parameters
        ----------
        
        Returns
        -------
        float
        """
        fullFittedTS = self.modelFitter.runSimulation(
              parameters=self.modelFitter.params,
              modelSpecification=self.modelFitter.modelSpecification,
              endTime=self.observedTS.end,
              numPoint=len(self.observedTS),
              returnDataFrame=False,
              _logger=self.modelFitter.logger,
              _loggerPrefix=self.modelFitter._loggerPrefix,
              )
        testFittedArr = fullFittedTS[self.modelFitter.selectedColumns]
        testFittedArr = testFittedArr[self.testIdxs, :]
        residualsArr = self.testObservedArr - testFittedArr
        rsq = 1 - np.var(residualsArr.flatten())  \
              / np.var(self.testObservedArr.flatten())
        return rsq


class SuiteFitterCrossValidator(SuiteFitterCore, AbstractCrossValidator):
    """Cross validation for Model Fitter"""
    pass
