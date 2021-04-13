# -*- coding: utf-8 -*-
"""
 Created on August 18, 2020

@author: joseph-hellerstein

Bootstrapping. Provides a single external: bootstrap().
Several considerations are made;
  1. Extensibility of the bootstrap results by returning
     instances of BootstrapResult.
  2. Multiprocessing. This requires a top-level (static)
     method. Arguments are packaged in _Arguments.
"""

from SBstoat._bootstrapRunner import BootstrapRunner, RunnerArgument
from SBstoat._bootstrapResult import BootstrapResult
from SBstoat import _modelFitterCrossValidator as mfc
from SBstoat._parallelRunner import ParallelRunner
from SBstoat import _helpers

import multiprocessing
import time
import typing

MAX_CHISQ_MULT = 5
PERCENTILES = [2.5, 97.55]  # Percentile for confidence limits
IS_REPORT = True
ITERATION_MULTIPLIER = 10  # Multiplier to calculate max bootsrap iterations
ITERATION_PER_PROCESS = 5  # Numer of iterations handled by a process
MAX_TRIES = 10  # Maximum number of tries to fit
MAX_ITERATION_TIME = 10.0


class ModelFitterBootstrap(mfc.ModelFitterCrossValidator):

    def bootstrap(self, isParallel=True,
          # The following must be kept in sync with ModelFitterCore.__init__
          numIteration:int=None,
          maxProcess:int=None,
          serializePath:str=None,
          **kwargs: dict):
        """
        Constructs a bootstrap estimate of parameter values.

        Parameters
        ----------
        isParallel: bool
            run in parallel
        numIteration: number of bootstrap iterations
        maxProcess: Maximum number of processes to use. Default: numCPU
        serializePath: Where to serialize the fitter after bootstrap
        kwargs: arguments passed to ObservationSynthesizer

        Example
        -------
            f.bootstrap()
            f.getParameterMeans()  # Mean values
            f.getParameterStds()  # standard deviations

        Notes
            1. Arguments can be overriden by the constructor using
               the keyword argument bootstrapKwargs.
        ----
        """
        def getValue(name, value, defaultValue=None):
            if value is not None:
                return value
            # Handle arguments specified in constructor
            if name in self.bootstrapKwargs:
                if self.bootstrapKwargs[name] is not None:
                    return self.bootstrapKwargs[name]
            if name in self.__dict__.keys():
                return self.__dict__[name]
            # None specified
            return defaultValue
        #
        # Initialization
        numIteration = getValue("numIteration", numIteration)
        isParallel = getValue("_isParallel", isParallel)
        isProgressBar = getValue("_isProgressBar", None, defaultValue=True)
        if maxProcess is None:
            maxProcess = self._maxProcess
        if maxProcess is None:
            maxProcess = multiprocessing.cpu_count()
        serializePath = getValue("serializePath", serializePath)
        # Ensure that there is a fitted model
        if self.minimizerResult is None:
            self.fitModel()
        # Construct arguments collection
        numProcess = min(maxProcess, numIteration)
        batchSize = numIteration // numProcess
        argumentsCol = [RunnerArgument(self,
              numIteration=batchSize,
              _loggerPrefix="bootstrap",
              **kwargs) for i in range(numProcess)]
        # Run separate processes for each batch
        runner = ParallelRunner(BootstrapRunner,
              desc="iteration", maxProcess=numProcess)
        results = runner.runSync(argumentsCol, isParallel=isParallel,
            isProgressBar=isProgressBar)
        # Check the results
        if len(results) == 0:
            msg = "modelFitterBootstrap/timeout in solving model."
            msg = "\nConsider increasing per timeout."
            msg = "\nCurent value: %f" % MAX_ITERATION_TIME
            self.logger.result(msg)
        else:
            self.bootstrapResult = BootstrapResult.merge(results, self)
            # Update the logger in place
            _ = _helpers.copyObject(self.bootstrapResult.fitter.logger,
                  self.logger)
            self.bootstrapResult.fittedStatistic.calculate()
            self.logger.result("%d bootstrap estimates of parameters."
                  % self.bootstrapResult.numSimulation)
            if serializePath is not None:
                self.serialize(serializePath)

    def getParameterMeans(self)->typing.List[float]:
        """
        Returns a list of values mean values of parameters from bootstrap.

        Return
        ------
        NamedTimeseries

        Example
        -------
              f.getParameterMeans()
        """
        if self._checkBootstrap(isError=False):
            return self.bootstrapResult.parameterMeanDct.values()
        return [self.params[str(p)].value for p in self.parametersToFit]

    def getParameterStds(self)->typing.List[float]:
        """
        Returns a list of values std values of parameters from bootstrap.

        Return
        ------
        NamedTimeseries

        Example
        -------
              f.getParameterStds()
        """
        if self._checkBootstrap(isError=False):
            return list(self.bootstrapResult.parameterStdDct.values())
        raise ValueError("***Must run bootstrap to get parameter stds.")

    def _checkBootstrap(self, isError:bool=True)->bool:
        """
        Verifies that bootstrap has been done.
        """
        self._checkFit()
        if self.bootstrapResult is None:
            if isError:
                raise ValueError("Must use bootstrap first.")
            return False
        return True
