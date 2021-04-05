#!/usr/bin/env python
# coding: utf-8

"""
Functions and Classes used for cross validation.

_runCrossValidate: top level function for executing a fitter.
    returns parameters and a score

AbstractFitter: Wrapper for a fitter. Must be pickleable.
    User must subclass this.
    Must override:
      @property: parameters - returns lmfit.parameters
      function: fit. After running fit, parameters are available.
      function: score. Returns a value between 0 (low) and 1 (high)


AbstractCrossValidator: Controls cross validation.
    User must subclass this.
    Must overide:
      function: _getFitterGenerator: generator for fitters
      function: crossValidate: loop for cross validation

"""
import SBstoat._constants as cn
from SBstoat._parallelRunner import AbstractRunner, ParallelRunner

import pandas as pd


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
class FitterRunner(AbstractRunner):
    """
    Wrapper for user-provided code that is run in parallel.
    An AbstractRunner has a run method that returns a list of
    work unit results.
    """

    def __init__(self, fitters):
        self.fitters = fitters
        self._fittersProcessed = 0

    @property
    def numWorkUnit(self):
        """
        Returns
        -------
        int: number of work units to be processed by runner
        """
        return len(self.fitters)

    @property
    def isDone(self):
        """
        Returns
        -------
        bool: all work has been processed
        """
        return self._fittersProcessed == self.numWorkUnit

    def run(self):
        """
        Interface for repeated running of work units.

        Returns
        -------
        Object
            list of work unit results
        """
        if not self.isDone:
            results = self.fitters[self._fittersProcessed].run()
            self._fittersProcessed += 1
            return [results]
        return []


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

    def __init__(self, maxProcess=None):
        """
        Parameters
        ----------
        maxProcess: int
            maximum number of processes if running in parallel
        """
        self.maxProcess = maxProcess
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

        Parameters
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

    def _crossValidate(self, fitterGenerator, isParallel=True):
        """
        Calculates parameters for folds.

        Parameters
        ----------
        fitterGenerator: generator
        isParallel: bool
             run each fold in parallel
        """
        self.cvFitters = list(fitterGenerator)
        runner = ParallelRunner(FitterRunner, desc="Folds",
              maxProcess=self.maxProcess)
        argumentsCol = runner._mkArgumentsCollections(self.cvFitters)
        initialResults = runner.runSync(argumentsCol,
              isParallel=isParallel, isProgressBar=True)
        results = []
        _ = [results.extend(r) for r in initialResults]
        # Extract the fields
        _ = [self.cvParametersCollection.append(r[0]) for r in results]
        _ = [self.cvRsqs.append(r[1]) for r in results]

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
        for fold, parameters in enumerate(self.cvParametersCollection):
            valuesDct = parameters.valuesdict()
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
