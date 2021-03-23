#!/usr/bin/env python
# coding: utf-8

"""Functions and Classes used for cross validation."""

import copy
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import SBstoat._constants as cn

################ FUNCTIONS ################
def _runCrossValidate(fitter):
    """
    Top level function that runs crossvalidation for a fitter.
    Used in parallel processing.

    Parameters
    ----------
    AbstractFitter
    
    Returns
    -------
    lmfit.Parameters, score
    """
    fitter.fit()
    score = fitter.score()
    return fitter.parameters, score


################ CLASSES ################
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
        iter-AbstractFitter
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
    
    def _crossValidate(self, fitterGenerator, isParallel=False):
        """
        Calculates parameters for folds.

        Parameters
        ----------
        fitterGenerator: generator
        """
        self.cvFitters = [f for f in fitterGenerator]
        results = []
        if isParallel:
            raise RuntimeError("Not implemented.")
        else:
            for fitter in self.cvFitters:
                results.append(_runCrossValidate(fitter))
        # Extract the fields
        [self.cvParametersCollection.append(r[0]) for r in results]
        [self.cvRsqs.append(r[1]) for r in results]
    
    def crossValidate(self, numFold):
        """
        External interface to crossvalidation.

        Parameters
        ----------
        numFold: int
        """
        raise RuntimeError("Must implement method %s in class %s" %
              ("crossValidate", str(self.__class__)))

    @property
    def parameterDF(self):
        """
        Constructs a DataFrame for the mean, std of parameter values.
        
        Returns
        -------
        pd.DataFrame
            Columns: MEAN, STD, COUNT (# folds)
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
        reportDF[cn.COUNT] = self.numFold
        return reportDF
   
    @property 
    def scoreDF(self):
        scores = [f.score() for f in self.cvFitters]
        return pd.DataFrame({cn.SCORE: scores})
