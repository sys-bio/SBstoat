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

from SBstoat.timeseriesStatistic import TimeseriesStatistic
from SBstoat._bootstrapResult import BootstrapResult
from SBstoat.observationSynthesizer import  \
      ObservationSynthesizerRandomizedResiduals
from SBstoat import _modelFitterCrossValidator as mfc
from SBstoat._parallelRunner import ParallelRunner
from SBstoat import _helpers
from SBstoat.logs import Logger

import multiprocessing
import numpy as np
import sys
import typing

MAX_CHISQ_MULT = 5
PERCENTILES = [2.5, 97.55]  # Percentile for confidence limits
IS_REPORT = True
ITERATION_MULTIPLIER = 10  # Multiplier to calculate max bootsrap iterations
ITERATION_PER_PROCESS = 5  # Numer of iterations handled by a process
MAX_TRIES = 10  # Maximum number of tries to fit
MAX_ITERATION_TIME = 10.0


###############  HELPER CLASSES ###############
class _Arguments():
    """ Arguments passed to _runBootstrap. """

    def __init__(self, fitter, 
                 numIteration:int=10,
                 synthesizerClass=ObservationSynthesizerRandomizedResiduals,
                 _loggerPrefix="",
                 **kwargs: dict):
        # Same the antimony model, not roadrunner bcause of Pickle
        self.fitter = fitter.copy(isKeepLogger=True)
        self.numIteration  = numIteration
        self.synthesizerClass = synthesizerClass
        self._loggerPrefix = _loggerPrefix
        self.kwargs = kwargs


################# FUNCTIONS ####################
def _runBootstrap(arguments:_Arguments)->BootstrapResult:
    """
    Executes bootstrapping.

    Parameters
    ----------
    arguments: inputs to bootstrap

    Notes
    -----
    1. Only the first process generates progress reports.
    2. Uses METHOD_LEASTSQ for fitModel iterations.
    """
    fitter = arguments.fitter
    logger = fitter.logger
    mainBlock = Logger.join(arguments._loggerPrefix, "_runBootstrap")
    mainGuid = logger.startBlock(mainBlock)
    # Unapack arguments
    isSuccess = False
    lastErr = ""
    # Do an initial fit
    for _ in range(MAX_TRIES):
        try:
            fitter.fitModel()  # Initialize model
            isSuccess = True
            break
        except Exception as err:
            lastErr = err
    # Set up logging for this process
    fd = logger.getFileDescriptor()
    if fd is not None:
        sys.stderr = logger.getFileDescriptor()
        sys.stdout = logger.getFileDescriptor()
    iterationGuid = None
    if not isSuccess:
        msg = "modelFitterBootstrip/_runBootstrap"
        logger.error(msg,  lastErr)
        fittedStatistic = TimeseriesStatistic(fitter.observedTS, percentiles=[])
        bootstrapResult = BootstrapResult(fitter, 0, {},
              fittedStatistic)
    else:
        numIteration = arguments.numIteration
        cols = fitter.selectedColumns
        synthesizer = arguments.synthesizerClass(
              observedTS=fitter.observedTS.subsetColumns(cols),
              fittedTS=fitter.fittedTS.subsetColumns(cols),
              **arguments.kwargs)
        # Initialize
        parameterDct = {str(p): [] for p in fitter.parametersToFit}
        numSuccessIteration = 0
        if fitter.minimizerResult is None:
            fitter.fitModel()
        baseChisq = fitter.minimizerResult.redchi
        # Do the bootstrap iterations
        bootstrapError = 0
        iterationBlock = Logger.join(mainBlock, "Iteration")
        for iteration in range(numIteration*ITERATION_MULTIPLIER):
            if iterationGuid is not None:
                logger.endBlock(iterationGuid)
            iterationGuid = logger.startBlock(iterationBlock)
            newObservedTS = synthesizer.calculate()
            fittingSetupBlock = Logger.join(iterationBlock, "fittingSetup")
            fittingSetupGuid = logger.startBlock(fittingSetupBlock)
            newFitter = ModelFitterBootstrap(fitter.roadrunnerModel,
                  newObservedTS,
                  parametersToFit=fitter.parametersToFit,
                  selectedColumns=fitter.selectedColumns,
                  # Use bootstrap methods for fitting
                  fitterMethods=fitter._bootstrapMethods,
                  parameterLowerBound=fitter.lowerBound,
                  parameterUpperBound=fitter.upperBound,
                  logger=logger,
                  _loggerPrefix=iterationBlock,
                  isPlot=fitter._isPlot)
            fittedStatistic = TimeseriesStatistic(newFitter.observedTS,
                  percentiles=[])
            logger.endBlock(fittingSetupGuid)
            try:
                if numSuccessIteration >= numIteration:
                    # Performed the iterations
                    break
                tryBlock = Logger.join(iterationBlock, "try")
                tryGuid = logger.startBlock(tryBlock)
                try:
                    tryFitterBlock = Logger.join(tryBlock, "Fitter")
                    tryFitterGuid = logger.startBlock(tryFitterBlock)
                    newFitter.fitModel(params=fitter.params)
                    logger.endBlock(tryFitterGuid)
                except Exception as err:
                    # Problem with the fit.
                    msg = "modelFitterBootstrap. Fit failed on iteration %d."  \
                          % iteration
                    fitter.logger.error(msg, err)
                    logger.endBlock(tryGuid)
                    continue
                if newFitter.minimizerResult.redchi > MAX_CHISQ_MULT*baseChisq:
                    if IS_REPORT:
                        msg = "Fit has high chisq: %2.2f on iteration %d."
                        fitter.logger.exception(msg 
                              % (newFitter.minimizerResult.redchi, iteration))
                    logger.endBlock(tryGuid)
                    continue
                if newFitter.params is None:
                    continue
                numSuccessIteration += 1
                dct = newFitter.params.valuesdict()
                _ = [parameterDct[str(p)].append(dct[str(p)]) for
                      p in fitter.parametersToFit]
                cols = newFitter.fittedTS.colnames
                fittedStatistic.accumulate(newFitter.fittedTS)
                newFitter.observedTS = synthesizer.calculate()
                logger.endBlock(tryGuid)
            except Exception as err:
                msg = "modelFitterBootstrap"
                msg += " Error on iteration %d."  % iteration
                fitter.logger.error(msg, err)
                bootstrapError += 1
        bootstrapResult = BootstrapResult(fitter, numSuccessIteration, parameterDct,
              fittedStatistic, bootstrapError=bootstrapError)
    if iterationGuid is not None:
        logger.endBlock(iterationGuid)
    logger.endBlock(mainGuid)
    if fd is not None:
        if not fd.closed:
            fd.close()
    return bootstrapResult


##################### CLASSES #########################
class ModelFitterBootstrap(mfc.ModelFitterCrossValidator):

    def bootstrap(self, isParallel=True,
          # The following must be kept in sync with ModelFitterCore.__init__
          numIteration:int=None,
          synthesizerClass=ObservationSynthesizerRandomizedResiduals,
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
        synthesizerClass: object that synthesizes new observations
            Must subclass ObservationSynthesizer
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
        def getValue(name, value):
            if value is not None:
                return value
            # Handle arguments specified in constructor
            if name in self.bootstrapKwargs:
                if self.bootstrapKwargs[name] is not None:
                    return self.bootstrapKwargs[name]
            # None specified
            return None
        #
        # Initialization
        numIteration = getValue("numIteration", numIteration)
        synthesizerClass = getValue("synthesizerClass", synthesizerClass)
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
        argumentsCol = [_Arguments(self,
              numIteration=batchSize,
              synthesizerClass=synthesizerClass,
              _loggerPrefix="bootstrap",
              **kwargs) for i in range(numProcess)]
        # Run separate processes for each batch
        runner = ParallelRunner(_runBootstrap, desc="iteration",
              maxProcess=numProcess, batchSize=batchSize)
        results = runner.runSync(argumentsCol, isParallel=isParallel)
        # Check the results
        if len(results) == 0:
            msg = "modelFitterBootstrap/timeout in solving model."
            msg = "\nConsider increasing per timeout."
            msg = "\nCurent value: %f" % MAX_ITERATION_TIME
            self.logger.result(msg)
        else:
            self.bootstrapResult = BootstrapResult.merge(results)
            # Update the logger in place
            _ = _helpers.copyObject(self.bootstrapResult.fitter.logger,
                  self.logger)
            if self.bootstrapResult.fittedStatistic is not None:
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
