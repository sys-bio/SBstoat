#!/usr/bin/env python
# coding: utf-8

"""Functions and Classes used for cross validation."""

import copy
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import SBstoat._constants as cn
from SBstoat.abstractCrossValidator import AbstractCrossValidator, AbstractFitter
from SBstoat._modelFitterCore import ModelFitterCore


##################### CLASSES ###########################

class Fitter(AbstractFitter):

    def __init__(self, modelSpecification, observedTS, parametersToFit,
          trainIdxs=None, testIdxs=None,  **kwargs):
        """
        Parameters
        ----------
        modelSpecification: ExtendedRoadRunner/str
            roadrunner model or antimony model
        observedTS: NamedTimeseries
        parametersToFit: list-str/SBstoat.Parameter/None
            parameters in the model that you want to fit
            if None, no parameters are fit
        trainIdxs: list-int
            rows of observedData used for estimating parameters
        testIdxs: list-int
            rows of observedData used for scoring fit
        """
        self.observedTS = observedTS.copy()
        self.trainIdxs = trainIdxs
        self.testIdxs = testIdxs
        if self.trainIdxs is None:
            self.trainIdxs = list(range(len(self.observedTS)))
        if self.testIdxs is None:
            self.testIdxs = list(range(len(self.observedTS)))
        self.trainTS = self.observedTS[self.trainIdxs]
        self.testTS = self.observedTS[self.testIdxs]
        self.fitter = ModelFitterCore(modelSpecification, self.trainTS,
              parametersToFit, **kwargs)
    
    @property
    def parameters(self):
        """
        Returns
        -------
        lmfit.Parameters
        """
        return self.fitter.params

    def fit(self):
        """
        Estimates parameters.
        Parameters
        ----------
        
        Returns
        -------
        """
        self.fitter.fitModel()

    def score(self):
        """
        Returns an R^2 for predicting testData
        Parameters
        ----------
        
        Returns
        -------
        float
        """
        fullFittedTS = ModelFitterCore.runSimulation(
              roadrunner=self.fitter.roadrunnerModel,
              parameters=self.parameters,
              startTime=self.observedTS.start,
              endTime=self.observedTS.end,
              numPoint=len(self.observedTS),
              _logger=self.fitter.logger,
              returnDataFrame=False,
              )
        columns = self.fitter.selectedColumns
        testObservedArr = self.testTS[columns]
        testFittedArr = fullFittedTS[columns]
        testFittedArr = testFittedArr[self.testIdxs, :]
        residualsArr = testObservedArr - testFittedArr
        rsq = 1 - np.var(residualsArr.flatten())/np.var(testObservedArr.flatten())
        return rsq


class ModelFitterCrossValidator(ModelFitterCore, AbstractCrossValidator):
    """Cross validation for Model Fitter"""

    def __init__(self, *args, **kwargs):
        super(ModelFitterCore, self).__init__(*args, **kwargs)
        super(AbstractCrossValidator, self).__init__()

    def _nextFitter(self, numPoint, numFold): 
        """
        Constructs fitters for each fold.

        Parameters
        ----------
        numPoint: int
            number of points in the data
        numFold: int
            number of folds in the cross validation
        
        Returns
        -------
        Generator
            iter-ModelFitterCore
        """
        generator = self.__class__.getFoldIdxGenerator(numPoint, numFold)
        for trainIdxs, testIdxs in generator:
            # FIXME: Add keyword arguments
            import pdb; pdb.set_trace()
            yield Fitter(self.modelSpecification,
                  self.observedTS, self.parametersToFit, trainIdxs,
                  testIdxs)
