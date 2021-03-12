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
        raise RuntimeError("Must implement method %s in class %s" %
              ("parameters", str(self.__class__)))

    def fit(self):
        """
        Estimates parameters.
        Parameters
        ----------
        
        Returns
        -------
        """
        raise RuntimeError("Must implement method %s in class %s" %
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
        raise RuntimeError("Must implement method %s in class %s" %
              ("score", str(self.__class__)))


class AbstractCrossValidator(object):
    """
    Base clase for performing cross validation using parameter fitting.
    """

    def __init__(self):
        self.cvFitters = []
        self.cvRsqs = []
        self.cvParametersCollection = []

    @property
    def numFold(self):
        return len(self.cvFitters)

    def _getFitterGenerator(self, numFold):
        """
        Generator for Fitters.A

        Parameters
        ----------
        numFold: int
            number of folds to generate       
  
        Returns
        -------
        crossFitter
        """
        raise RuntimeError("Must implement method %s in class %s" %
              ("_getFitterGenerator", str(self.__class__)))

    @staticmethod
    def getFoldIdxGenerator(numPoint, numFold):
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
    
    def _crossValidate(self, fitterGenerator):
        """
        Calculates parameters for folds.

        Parameters
        ----------
        fitterGenerator: generator
        """
        for fitter in fitterGenerator:
            fitter.fit()
            self.cvFitters.append(fitter)
            self.cvRsqs.append(fitter.score())
            self.cvParametersCollection.append(fitter.parameters)
    
    def crossValidate(self, numFold):
        """
        External interface to crossvalidation.

        Parameters
        ----------
        numFold: int
        """
        raise RuntimeError("Must implement method %s in class %s" %
              ("crossValidate", str(self.__class__)))

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
            valuesDct = self.cvFitters[fold].parameters.valuesdict()
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
        scores = [f.score() for f in self.cvFitters]
        return pd.DataFrame({cn.SCORE: scores})
