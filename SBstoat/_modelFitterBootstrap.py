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
from SBstoat import _modelFitterCore as mfc
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
ITERATION_PER_PROCESS = 20  # Numer of iterations handled by a process
MAX_TRIES = 10  # Maximum number of tries to fit
MAX_ITERATION_TIME = 10.0


###############  HELPER CLASSES ###############
class _Arguments():
    """ Arguments passed to _runBootstrap. """

    def __init__(self, fitter, numProcess:int, processIdx:int,
                 numIteration:int=10,
                 reportInterval:int=-1,
                 synthesizerClass=  \
                 ObservationSynthesizerRandomizedResiduals,
                 _loggerPrefix="",
                 **kwargs: dict):
        # Same the antimony model, not roadrunner bcause of Pickle
        self.fitter = fitter.copy(isKeepLogger=True)
        self.numIteration  = numIteration
        self.reportInterval  = reportInterval
        self.numProcess = numProcess
        self.processIdx = processIdx
        self.synthesizerClass = synthesizerClass
        self._loggerPrefix = _loggerPrefix
        self.kwargs = kwargs


################# FUNCTIONS ####################
def _runBootstrap(arguments:_Arguments, queue=None)->BootstrapResult:
    """
    Executes bootstrapping.

    Parameters
    ----------
    arguments: inputs to bootstrap
    queue: multiprocessing.Queue

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
    processIdx = arguments.processIdx
    if fd is not None:
        sys.stderr = logger.getFileDescriptor()
        sys.stdout = logger.getFileDescriptor()
    iterationGuid = None
    if not isSuccess:
        msg = "Process %d/modelFitterBootstrip/_runBootstrap" % processIdx
        logger.error(msg,  lastErr)
        fittedStatistic = TimeseriesStatistic(fitter.observedTS, percentiles=[])
        bootstrapResult = BootstrapResult(fitter, 0, {},
              fittedStatistic)
    else:
        numIteration = arguments.numIteration
        reportInterval = arguments.reportInterval
        processingRate = min(arguments.numProcess,
                             multiprocessing.cpu_count())
        cols = fitter.selectedColumns
        synthesizer = arguments.synthesizerClass(
              observedTS=fitter.observedTS.subsetColumns(cols),
              fittedTS=fitter.fittedTS.subsetColumns(cols),
              **arguments.kwargs)
        # Initialize
        parameterDct = {p: [] for p in fitter.parametersToFit}
        numSuccessIteration = 0
        lastReport = 0
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
                  fitter.parametersToFit,
                  selectedColumns=fitter.selectedColumns,
                  # Use bootstrap methods for fitting
                  fitterMethods=fitter._bootstrapMethods,
                  parameterLowerBound=fitter.lowerBound,
                  parameterUpperBound=fitter.upperBound,
                  fittedDataTransformDct=fitter.fittedDataTransformDct,
                  logger=logger,
                  _loggerPrefix=iterationBlock,
                  isPlot=fitter._isPlot)
            fittedStatistic = TimeseriesStatistic(newFitter.observedTS,
                  percentiles=[])
            logger.endBlock(fittingSetupGuid)
            try:
                if (iteration > 0) and (iteration != lastReport)  \
                        and (processIdx == 0):
                    totalSuccessIteration = numSuccessIteration*processingRate
                    totalIteration = iteration*processingRate
                    if totalIteration % reportInterval == 0:
                        msg = "Bootstrap completed %d total iterations "
                        msg += "with %d successes."
                        msg = msg % (totalIteration, totalSuccessIteration)
                        fitter.logger.status(msg)
                        lastReport = numSuccessIteration
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
                    # Problem with the fit. Don't numSuccessIteration it.
                    msg = "Process %d/modelFitterBootstrap" % processIdx
                    msg += " Fit failed on iteration %d."  % iteration
                    fitter.logger.error(msg, err)
                    logger.endBlock(tryGuid)
                    continue
                if newFitter.minimizerResult.redchi > MAX_CHISQ_MULT*baseChisq:
                    if IS_REPORT:
                        msg = "Process %d: Fit has high chisq: %2.2f on iteration %d."
                        fitter.logger.exception(msg % (processIdx,
                              newFitter.minimizerResult.redchi, iteration))
                    logger.endBlock(tryGuid)
                    continue
                if newFitter.params is None:
                    continue
                numSuccessIteration += 1
                dct = newFitter.params.valuesdict()
                [parameterDct[p].append(dct[p]) for p in fitter.parametersToFit]
                cols = newFitter.fittedTS.colnames
                fittedStatistic.accumulate(newFitter.fittedTS)
                newFitter.observedTS = synthesizer.calculate()
                logger.endBlock(tryGuid)
            except Exception as err:
                msg = "Process %d/modelFitterBootstrap" % processIdx
                msg += " Error on iteration %d."  % iteration
                fitter.logger.error(msg, err)
                bootstrapError += 1
        fitter.logger.status("Process %d: completed bootstrap." % (processIdx + 1))
        bootstrapResult = BootstrapResult(fitter, numSuccessIteration, parameterDct,
              fittedStatistic, bootstrapError=bootstrapError)
    if iterationGuid is not None:
        logger.endBlock(iterationGuid)
    logger.endBlock(mainGuid)
    if fd is not None:
        if not fd.closed:
            fd.close()
    if queue is None:
        return bootstrapResult
    queue.put(bootstrapResult)


##################### CLASSES #########################
class ModelFitterBootstrap(mfc.ModelFitterCore):

    def bootstrap(self,
          # The following must be kept in sync with ModelFitterCore.__init__
          numIteration:int=None,
          reportInterval:int=None,
          synthesizerClass=ObservationSynthesizerRandomizedResiduals,
          maxProcess:int=None,
          serializePath:str=None,
          **kwargs: dict):
        """
        Constructs a bootstrap estimate of parameter values.

        Parameters
        ----------
        numIteration: number of bootstrap iterations
        reportInterval: number of iterations between progress reports
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
        def get(name, value):
            if value is not None:
                return value
            # Handle arguments specified in constructor
            if name in self.bootstrapKwargs:
                if self.bootstrapKwargs[name] is not None:
                    return self.bootstrapKwargs[name]
            # None specified
            return None
        # Handle overrides of arguments specified in constructor
        numIteration = get("numIteration", numIteration)
        reportInterval = get("reportInterval", reportInterval)
        synthesizerClass = get("synthesizerClass", synthesizerClass)
        maxProcess = get("maxProcess", maxProcess)
        serializePath = get("serializePath", serializePath)
        # Other initializations
        if maxProcess is None:
            maxProcess = multiprocessing.cpu_count()
        if self.minimizerResult is None:
            self.fitModel()
        # Run processes
        numProcess = max(int(numIteration/ITERATION_PER_PROCESS), 1)
        numProcess = min(numProcess, maxProcess)
        numProcessIteration = int(np.ceil(numIteration/numProcess))
        msg = "Running bootstrap for %d successful iterations " % numIteration
        msg += "with %d processes." % numProcess
        self.logger.activity(msg)
        # Run separate processes for each bootstrap
        processes = []
        queue = multiprocessing.Queue()
        results = []
        # Set to False for debug so not doing multiple processes
        if True:
            args_list = [_Arguments(self, numProcess, i,
                  numIteration=numProcessIteration,
                  reportInterval=reportInterval,
                  synthesizerClass=synthesizerClass,
                  _loggerPrefix="bootstrap",
                  **kwargs) for i in range(numProcess)]
            for args in args_list:
                p = multiprocessing.Process(target=_runBootstrap,
                      args=(args, queue,))
                p.start()
                processes.append(p)
            timeout = MAX_ITERATION_TIME*numProcessIteration
            try:
                # Get rid of possible zombies
                for _ in range(len(processes)):
                    results.append(queue.get(timeout=timeout))
                # Get rid of possible zombies
                for process in processes:
                    process.terminate()
            except Exception as err:
                msg = "modelFitterBootstrap/Error in process management"
                self.logger.error(msg, err)
            finally:
                pass
        else:
            # Keep to debug _runBootstrap single threaded
            thisNumProcess = 1
            thisProcessIdx = 0
            args = _Arguments(self, thisNumProcess, thisProcessIdx,
                  numIteration=numIteration,
                  reportInterval=reportInterval,
                  synthesizerClass=synthesizerClass,
                  _loggerPrefix="bootstrap",
                  **kwargs)
            results = [_runBootstrap(args)]
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
        return [self.params[p].value for p in self.parametersToFit]

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
