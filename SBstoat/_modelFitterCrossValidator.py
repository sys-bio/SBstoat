#!/usr/bin/env python
# coding: utf-8

"""Functions and Classes used for cross validation."""

import copy
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import SBstoat._constants as cn
from SBstoat.crossValidator CrossValidator


##################### CLASSES ###########################
class ModelFitterCrossValidator(CrossValidator):
   """Cross validation for Model Fitter"""

    class Fitter(AbstractFitter):

        def __init__(self, model, observedTS, parametersToFit,
              trainIdxs, testIdxs,  **kwargs):
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
        self.trainTS = self.observedTS[trainIdxs]
        self.testTS = self.observedTS[testIdxs]
        self.fitter = ModelFitter(model, observedTS, parametersToFit, **kwargs)
        
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
            fullFittedTS = ModelFitter.runSimulation(
                  roadrunner=fitter.roadrunnerModel,
                  parameters=self.parameters,
                  startTime=self.observedTS.start,
                  endTime=self.observedTS.end,
                  numPoint=len(self.observedTS),
                  _logger=self.fitter._logger,
                  )
            columns = self.fitter.columns
            testObservedTS = self.observedTS[self.testIdxs]
            testFittedTS = self.fitted[self.testIdxs]
            residualsArr = testObservedTS[columns] - testFittedTS[columns]
            rsq = 1 - np.var(residualsArr)/np.var(testObservedTS)
            return rsq

    def __init__(self, modelSpecification, observedData,
          parametersToFit, numFold, **kwargs):
        """
        modelSpecification: ExtendedRoadRunner/str
            roadrunner model or antimony model
        observedData: NamedTimeseries/str
            str: path to CSV file
        parametersToFit: list-str/SBstoat.Parameter/None
            parameters in the model that you want to fit
            if None, no parameters are fit
        numFold: int
        kwargs: dict
            keyword arguments passed to ModelFitter
        """
        self.modelSpecification = modelSpecification
        self.observedData = observedData
        self.parametersToFit = parametersToFit
        self.numFold = numFold
        self.kwargs = kwargs
        #
        fitter = ModelFitter(modelSPecification, observedData, parametersToFit,
              **kwargs)
        #
        self._numPoint = len(fitter.observedTS)
        self._generator = CrossModelFitter.foldIdxGenerator(self._numPoint,
              self.numFold)

    def _nextFitter(self): 
        for trainIdxs, testIdxs in self._generator:
            yield ModelFitterCrossValidator.Fitter(self,
                 model, observedTS, parametersToFit, trainIdxs, testIdxs,
                 **kwargs)
