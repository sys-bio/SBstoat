#!/usr/bin/env python
# coding: utf-8

"""Functions and Classes used for cross validation."""

import copy
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import SBstoat._constants as cn


class AbstractFitter(object):
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
              ("parameters", str(self.__class__)))

    def fit(self):
        """
        Estimates parameters.
        Parameters
        ----------
        
        Returns
        -------
        """
        raise RuntimeError("Must implement method %s in class %s",
              ("fit", str(self.__class__)))

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
              ("score", str(self.__class__)))


class AbstractCrossValidator(object):
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
              ("_nextFitter", str(self.__class__)))

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
        """
        Calculates parameters for folds.
        """
        generator = self._nextFitter()
        for fitter in generator:
            fitter.fit()
            self.fitters.append(fitter)
            self.rsqs.append(fitter.score())
            self.parametersCollection.append(fitter.parameters)

    def reportParameters(self):
        """
        Constructs a report for the parameter values by fold.
        
        Returns
        -------
        pd.DataFrame
            Columns: MEAN, STD
            index: parameter
        """
        keys = [cn.FOLD, cn.PARAMETER, cn.VALUE]
        dct = {}
        for key in keys:
            dct[key] = []
        for fold in range(self.numFold):
            valuesDct = self.fitters[fold].parameters.valuesdict()
            for parameterName in valuesDct.keys():
                dct[cn.FOLD].append(fold)
                dct[cn.PARAMETER].append(parameterName)
                dct[cn.VALUE].append(valuesDct[parameterName])
        df = pd.DataFrame(dct)
        #
        reportDF = pd.DataFrame(df.groupby(cn.PARAMETER).mean())
        reportDF = reportDF.rename(columns={cn.VALUE: cn.MEAN})
        del reportDF[cn.FOLD]
        stdDF = pd.DataFrame(df.groupby(cn.PARAMETER).std())
        reportDF[cn.STD] = stdDF[cn.VALUE]
        return reportDF
    
    def reportScores(self):
        scores = [f.score() for f in self.fitters]
        return pd.DataFrame({cn.SCORE: scores})
