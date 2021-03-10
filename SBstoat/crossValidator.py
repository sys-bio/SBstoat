#!/usr/bin/env python
# coding: utf-8

"""Functions and Classes used for cross validation."""

import copy
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import SBstoat._constants as cn
from SBstoat.modelFitter import ModelFitter
from SBstoat.observationSynthesizer import ObservationSynthesizerRandomErrors
from SBstoat.namedTimeseries import NamedTimeseries, TIME


##################### CLASSES ###########################
# FIXME: 1. Doesn't work for SuiteFitter
#        2. Why is getFoldGenerator in AbstractFitter?
class CrossFitter(object):
    """
    Fitting function for CrossValidation. Constructed with
    its training and test data.
    """
    @property
    def parameters(self):
        """
        Returns
        -------
        lmfit.Parameters
        """
        raise RuntimeError("Must implement method %s in class %s",
              ("parameters", str(self.__class__))

    def fit(self):
        """
        Estimates parameters.
        Parameters
        ----------
        
        Returns
        -------
        """
        raise RuntimeError("Must implement method %s in class %s",
              ("fit", str(self.__class__))

    def score(self):
        """
        Returns an R^2 for predicting testData
        Parameters
        ----------
        
        Returns
        -------
        float
        """
        raise RuntimeError("Must implement method %s in class %s",
              ("score", str(self.__class__))


class CrossValidator(object):
    """
    Base clase for performing cross validation using parameter fitting.
    Must override:
        __iter__
    """
    
    def __init__(self):
        # Results
        self.fitters = []
        self.rsqs = []
        self.parametersCollection = []

    def _nextFitter(self):
        """
        Returns
        -------
        crossFitter
        """
        raise RuntimeError("Must implement method %s in class %s",
              ("_nextFitter", str(self.__class__))

    @staticmethod
    def foldIdxGenerator(numPoint, numFold):
        """
        Generates pairs of trainining and test indices.
        
        Parameters:
        ----------
        numPoint: int
            number of time points
        numFold: int
            number of pairs of testIndices and trainIndices
        
        Returns:
        --------
        list of pairs of train indices, test indices
        """
        indices = range(numPoint)
        for remainder in range(numFold):
            testIndices = []
            for idx in indices:
                if idx % numFold == remainder:
                    testIndices.append(idx)
            trainIndices = list(set(indices).difference(testIndices))
            yield trainIndices, testIndices
    
        def execute(self):
            generator = self._nextFitter()
            for fitter in generator:
                fitter.fit()
                self.fitters.append(fitter)
                self.rsqs.append(newFitter.score())
                self.parametersCollection.append(newFitter.parameters)

    # FIXME: Include estimates of parameter variance
    def reportParameters(self):
        """
        Constructs a report for the parameter values by fold.
        
        Returns
        -------
        pd.DataFrame
        """
        if self.trueParameterDct is  None:
            raise ValueError("Must construct CrossValidator with trueParameterDct")
        # Construct parameter information
        keys = [CrossValidator.FOLD, CrossValidator.TRUE,
              CrossValidator.PREDICTED, CrossValidator.PARAMETER]
        dct = {}
        for key in keys:
            dct[key] = []
        for fold in range(self.numFold):
            valuesDct = self.fitters[fold].params.valuesdict()
            for parameterName in self.parameterNames:
                dct[CrossValidator.FOLD].append(fold)
                dct[CrossValidator.PARAMETER].append(parameterName)
                dct[CrossValidator.TRUE].append(
                      self.trueParameterDct[parameterName])
                dct[CrossValidator.PREDICTED].append(valuesDct[parameterName])
        reportDF = pd.DataFrame(dct)
        #
        return reportDF
    
    def reportRsq(self):
        rssqs = [f.optimizer.rssq for f in self.fitters]
        return pd.DataFrame({CrossValidator.RSQ: rssqs})


############# CrossModelFitter #################
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
