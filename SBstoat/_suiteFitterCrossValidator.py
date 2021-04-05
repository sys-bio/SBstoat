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
class SuiteFitterWrapper(AbstractFitter):

    def __init__(self, suiteFitter, testTSDct):
        """
        Parameters
        ----------
        suiteFitters: list-SuiteFitter
        testTSDct: dict: key is modelName; value is NamedTimeseries of test data
        """
        super().__init__()
        self.suiteFitter = suiteFitter
        self.testTSDct = testTSDct
    
    @property
    def parameters(self):
        """
        Returns
        -------
        lmfit.Parameters
        """
        return self.suiteFitter.params

    def fit(self):
        """
        Estimates parameters.
        """
        self.suiteFitter.fitSuite()

    def score(self):
        """
        Returns an R^2 for predicting test data
        
        Returns
        -------
        float
        """
        manager = self.suiteFitter.parameterManager
        residualsArrs = []
        observedArrs = []
        for modelName, modelFitter in self.suiteFitter.fitterDct.items():
            fullFittedTS = modelFitter.runSimulation(
                  parameters=manager.mkParameters(modelName=modelName),
                  modelSpecification=modelFitter.modelSpecification,
                  endTime=modelFitter.observedTS.end,
                  numPoint=modelFitter.numPoint,
                  returnDataFrame=False,
                  _logger=modelFitter.logger,
                  _loggerPrefix=modelFitter._loggerPrefix,
                  )
            observedTS = self.testTSDct[modelName]
            observedTimes = observedTS[cn.TIME]
            idxs = modelFitter.selectCompatibleIndices(fullFittedTS[cn.TIME],
                  observedTimes)
            fittedArr = fullFittedTS[modelFitter.selectedColumns]
            fittedArr = fittedArr[idxs, :]
            fittedArr = fittedArr.flatten()
            observedArr = observedTS[modelFitter.selectedColumns].flatten()
            residualsArrs.append(observedArr - fittedArr)
            observedArrs.append(observedArr)
        residualsArr = np.concatenate(residualsArrs)
        observedArr = np.concatenate(observedArrs)
        rsq = 1 - np.var(residualsArr) / np.var(observedArr)
        return rsq


class SuiteFitterCrossValidator(SuiteFitterCore, AbstractCrossValidator):
    """Cross validation for Model Fitter"""

    def __init__(self, *args, **kwargs):
        # Run the constructors for the parent classes
        SuiteFitterCore.__init__(self, *args, **kwargs)
        AbstractCrossValidator.__init__(self)

    def _getFitterGenerator(self, numFold): 
        """
        Constructs fitters for each fold.

        Parameters
        ----------
        numFold: int
            number of folds in the cross validation
        
        Returns
        -------
        Generator for SuiteFitterWrapper
        """
        foldIdxGenerators = {m: self.getFoldIdxGenerator(f.numPoint, numFold)
              for m, f in self.fitterDct.items()}
        modelSpecifications = [f.modelSpecifications
              for f in self.fitterDct.values()]
        parametersCol = [f.parametersToFit for f in self.fitterDct.values()]
        # Create the arguments for processing each fold
        for _ in range(numFold):
            testTSDct = {}
            datasets = []
            # Create the test and training data
            for modelName, modelFitter in self.fitterDct.items():
                trainIdxs, testIdxs = foldIdxGenerators[modelName].__next__()
                testTSDct[model] = modelFitter.observedTS[testIdxs]
                datasets.append(modelFitter.observedTS[trainIdxs])
            # Construct the SuiteFitterWrapper
            suiteFitterWrapper = SuiteFitterWrapper(modelSPecifications, datasets,
                  parametersCol,
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
            yield SuiteFitterWrapper(suiteFitterWrapper, testTSDct)

    def crossValidate(self, numFold, isParallel=True):
        """
        Do cross validation.

        Parameters
        ----------
        numFold: int
        kwargs: dict
             optional parameters for _crossValidate
        isParallel: bool
             run each fold in parallel
        """
        fitterGenerator = self._getFitterGenerator(numFold)
        self._crossValidate(fitterGenerator, isParallel=isParallel)
