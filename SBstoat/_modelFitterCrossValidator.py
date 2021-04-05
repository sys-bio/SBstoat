#!/usr/bin/env python
# coding: utf-8
"""
Functions and Classes used for cross validation.
Usage:
    fitter = ModelFitter(modelSpecification, dataSOurce, parametersToFit)
    fitter.crossValidate(5)  # Do cross validation with 5 folds
    fitter.scoreDF  # Dataframe with the scores by fold
    fitter.parameterDF  # Dataframe with parameter mean, std
"""

import copy
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import SBstoat._constants as cn
from SBstoat.abstractCrossValidator import AbstractCrossValidator, AbstractFitter
from SBstoat._modelFitterCore import ModelFitterCore


##################### CLASSES ###########################
class ModelFitterWrapper(AbstractFitter):

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
        self.modelFitter = ModelFitterCore(modelSpecification, self.trainTS,
              parametersToFit, **kwargs)
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


class ModelFitterCrossValidator(ModelFitterCore, AbstractCrossValidator):
    """Cross validation for ModelFitter"""

    def __init__(self, *args, **kwargs):
        # Run the constructors for the parent classes
        ModelFitterCore.__init__(self, *args, **kwargs)
        AbstractCrossValidator.__init__(self)
        # Handle case of deserialize
        if "observedTS" in self.__dict__.keys():
            self.numPoint = len(self.observedTS)

    def _getFitterGenerator(self, numFold): 
        """
        Constructs fitters for each fold.

        Parameters
        ----------
        numFold: int
            number of folds in the cross validation
        
        Returns
        -------
        Generator
            iter-ModelFitterWrapper
        """
        foldIdxGenerator = self.__class__.getFoldIdxGenerator(
                  self.numPoint, numFold)
        for trainIdxs, testIdxs in foldIdxGenerator:
            yield ModelFitterWrapper(self.modelSpecification,
                  self.observedTS, self.parametersToFit, trainIdxs, testIdxs,
                  bootstrapMethods=self._bootstrapMethods,
                  endTime=self.endTime,
                  fitterMethods=self._fitterMethods,
                  logger=self.logger,
                  _loggerPrefix=self._loggerPrefix,
                  isPlot=self._isPlot,
                  maxProcess=self._maxProcess,
                  numFitRepeat=self._numFitRepeat,
                  numIteration=self.bootstrapKwargs["numIteration"],
                  numPoint=self.numPoint,
                  numRestart=self._numRestart,
                  parameterLowerBound=self.lowerBound,
                  parameterUpperBound=self.upperBound,
                  selectedColumns=self.selectedColumns,
                  serializePath=self.bootstrapKwargs["serializePath"],
                  )

    def crossValidate(self, numFold, **kwargs):
        """
        Do cross validation.

        Parameters
        ----------
        numFold: int
        kwargs: dict
             optional parameters for _crossValidate
        """
        fitterGenerator = self._getModelFitter(numFold)
        self._crossValidate(fitterGenerator, **kwargs)
