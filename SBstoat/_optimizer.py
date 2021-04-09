"""Abstraction for optimization.

The abstraction extends lmfit optimizations by:
1. Ensuring that the parameters chosen have the lowest residuals sum of squares
2. Providing for a sequence of optimization methods
3. Providing an option to repeat a method sequence with different randomly
   chosen initial parameter values (numRandomRestart).

"""

from SBstoat.logs import Logger
from SBstoat import _helpers
from SBstoat import _constants as cn

import copy
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time



class _FunctionWrapper():
    """Wraps a function used for optimization."""

    def __init__(self, function, isCollect=False):
        """
        Parameters
        ----------
        function: function
            function callable by lmfit.Minimizer
               argument
                   lmfit.Parameter
                   isRawData - boolean to indicate return total SSQ
               returns: np.array
        isCollect: bool
            collect performance statistics on function execution
        """
        self._function = function
        self._isCollect = isCollect
        # Results
        self.perfStatistics = []  # durations of function executions
        self.rssqStatistics = []  # residual sum of squares, a quality measure
        self.rssq = 10e10
        self.bestParamDct = None

    @staticmethod
    def _calcSSQ(arr):
        return sum(arr**2)

    def execute(self, params, **kwargs):
        if self._isCollect:
            startTime = time.time()
        result = self._function(params, **kwargs)
        if self._isCollect:
            duration = time.time() - startTime
        rssq = _FunctionWrapper._calcSSQ(result)
        if rssq < self.rssq:
            self.rssq = rssq
            self.bestParamDct = dict(params.valuesdict())
        if self._isCollect:
            self.perfStatistics.append(duration - startTime)
            self.rssqStatistics.append(rssq)
        return result


class Optimizer():
    """
    Implements an interface to optimizers with abstractions
    for multiple methods and performance reporting.
    The class also handles an oddity with lmfit that the final parameters
    returned may not be the best.

    Usage
    -----
    optimizer = Optimizer(calcResiduals, params, [cn.METHOD_LEASTSQ])
    optimizer.execute()
    """

    def __init__(self, function, initialParams, methods, logger=None,
          isCollect=False):
        """
        Parameters
        ----------
        function: Funtion
           Arguments
            lmfit.parameters
            isInitialze (bool). True on first call the
            isGetBest (bool). True to retrieve best parameters
           returns residuals (if bool arguments are false)
        initialParams: lmfit.parameters
        methods: list-_helpers.OptimizerMethod
        isCollect: bool
           Collects performance statistcs
        """
        self._function = function
        self._methods = methods
        self._initialParams = initialParams
        self._isCollect = isCollect
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        # Outputs
        self.performanceStats = []  # list of performance results
        self.qualityStats = []  # relative rssq
        self.params = None
        self.minimizerResult = None
        self.rssq = None

    def copyResults(self):
        """
        Copies of the results of the optimization.

        Returns
        -------
        Optimizer
        """
        newOptimizer = Optimizer(self._function, self._initialParams.copy(),
              self._methods, logger=self.logger, isCollect=self._isCollect)
        newOptimizer._function = None  # Not serializable
        #
        newOptimizer.performanceStats = copy.deepcopy(self.performanceStats)
        newOptimizer.qualityStats = copy.deepcopy(self.qualityStats)
        newOptimizer.minimizerResult = copy.deepcopy(self.minimizerResult)
        newOptimizer.params = None
        if self.params is not None:
            newOptimizer.params = self.params.copy()
        newOptimizer.rssq = self.rssq
        return newOptimizer

    @staticmethod
    def _setRandomValue(params):
        """
        Sets value to a uniformly distributed random number between min and max.

        Parameters
        ----------
        params: lmfit.Parameters
        
        Returns
        -------
        lmfit.Parameters
        """
        newParameters = lmfit.Parameters()
        for name, parameter in params.items():
            newValue = np.random.uniform(parameter.min, parameter.max)
            newParameters.add(name, min=parameter.min, max=parameter.max,
                  value=newValue)
        return newParameters
       
    def execute(self):
        """
        Performs the optimization on the function.
        Result is self.params
        """
        lastExcp = None
        self.params = self._initialParams.copy()
        minimizer = None
        for optimizerMethod in self._methods:
            method = optimizerMethod.method
            kwargs = optimizerMethod.kwargs
            wrapperFunction = _FunctionWrapper(self._function,
                  isCollect=self._isCollect)
            minimizer = lmfit.Minimizer(wrapperFunction.execute, self.params)
            try:
                self.minimizerResult = minimizer.minimize(method=method, **kwargs)
            except Exception as excp:
                lastExcp = excp
                msg = "Error minimizing for method: %s" % method
                self.logger.error(msg, excp)
                continue
            # Update the parameters
            valuesDct = self.params.valuesdict()
            for parameterName in valuesDct.keys():
                self.params[parameterName].set(
                      value=wrapperFunction.bestParamDct[parameterName])
            # Update other statistics
            self.rssq = wrapperFunction.rssq
            self.performanceStats.append(list(wrapperFunction.perfStatistics))
            self.qualityStats.append(list(wrapperFunction.rssqStatistics))
        if minimizer is None:
            msg = "*** Optimization failed."
            self.logger.error(msg, lastExcp)

    def report(self):
        """
        Reports the result of an optimization.

        Returns
        -------
        str
        """
        VARIABLE_STG = "[[Variables]]"
        CORRELATION_STG = "[[Correlations]]"
        if self.minimizerResult is None:
            raise ValueError("Must do fitModel before reportFit.")
        valuesDct = self.params.valuesdict()
        valuesStg = _helpers.ppDict(dict(valuesDct), indent=4)
        reportSplit = str(lmfit.fit_report(self.minimizerResult)).split("\n")
        # Eliminate Variables section
        inVariableSection = False
        trimmedReportSplit = []
        for line in reportSplit:
            if VARIABLE_STG in line:
                inVariableSection = True
            if CORRELATION_STG in line:
                inVariableSection = False
            if inVariableSection:
                continue
            trimmedReportSplit.append(line)
        # Construct the report
        newReportSplit = [VARIABLE_STG]
        newReportSplit.extend(valuesStg.split("\n"))
        newReportSplit.extend(trimmedReportSplit)
        return "\n".join(newReportSplit)

    def plotPerformance(self, isPlot=True):
        """
        Plots the statistics for running the objective function.
        """
        if not self._isCollect:
            msg = "Must construct with isCollect = True "
            msg += "to get performance plot."
            raise ValueError(msg)
        # Compute statistics
        TOT = "Tot"
        CNT = "Cnt"
        AVG = "Avg"
        IDX = "Idx"
        totalTimes = [sum(v) for v in self.performanceStats]
        counts = [len(v) for v in self.performanceStats]
        averages = [np.mean(v) for v in self.performanceStats]
        df = pd.DataFrame({
            IDX: range(len(self.performanceStats)),
            TOT: totalTimes,
            CNT: counts,
            AVG: averages,
            })
        #
        _, axes = plt.subplots(1, 3)
        df.plot.bar(x=IDX, y=TOT, ax=axes[0], title="Total time",
              xlabel="method")
        df.plot.bar(x=IDX, y=AVG, ax=axes[1], title="Average time",
              xlabel="method")
        df.plot.bar(x=IDX, y=CNT, ax=axes[2], title="Number calls",
              xlabel="method")
        if isPlot:
            plt.show()

    def plotQuality(self, isPlot=True):
        """
        Plots the quality results
        """
        if not self._isCollect:
            msg = "Must construct with isCollect = True "
            msg += "to get quality plots."
            raise ValueError(msg)
        ITERATION = "iteration"
        _, axes = plt.subplots(len(self._methods))
        minLength = min([len(v) for v in self.qualityStats])
        # Compute statistics
        dct = {self._methods[i].method: self.qualityStats[i][:minLength]
            for i in range(len(self._methods))}
        df = pd.DataFrame(dct)
        df[ITERATION] = range(minLength)
        #
        for idx, method in enumerate(self._methods):
            if "AxesSubplot" in str(type(axes)):
                ax = axes
            else:
                ax = axes[idx]
            df.plot.line(x=ITERATION, y=method.method, ax=ax, xlabel="")
            ax.set_ylabel("SSQ")
            if idx == len(self._methods) - 1:
                ax.set_xlabel(ITERATION)
        if isPlot:
            plt.show()

    @staticmethod
    def mkOptimizerMethod(methodNames=None, methodKwargs=None,
          maxFev=cn.MAX_NFEV_DFT):
        """
        Constructs an OptimizerMethod
        Parameters
        ----------
        methodNames: list-str/str
        methodKwargs: list-dict/dict

        Returns
        -------
        list-OptimizerMethod
        """
        if methodNames is None:
            methodNames = [cn.METHOD_LEASTSQ]
        if isinstance(methodNames, str):
            methodNames = [methodNames]
        if methodKwargs is None:
            methodKwargs = {}
        # Ensure that there is a limit of function evaluations
        newMethodKwargs = dict(methodKwargs)
        if cn.MAX_NFEV not in newMethodKwargs.keys():
            newMethodKwargs[cn.MAX_NFEV] = maxFev
        elif maxFev is None:
            del newMethodKwargs[cn.MAX_NFEV]
        methodKwargs = np.repeat(newMethodKwargs, len(methodNames))
        #
        result = [_helpers.OptimizerMethod(n, k) for n, k  \
              in zip(methodNames, methodKwargs)]
        return result

    @classmethod
    def optimize(cls, function, initialParams, methods, numRestart=0, **kwargs):
        """
        Parameters
        ----------
        function: Funtion
           Arguments
            lmfit.parameters
            isInitialze (bool). True on first call the
            isGetBest (bool). True to retrieve best parameters
           returns residuals (if bool arguments are false)
        initialParams: lmfit.parameters
        methods: list-_helpers.OptimizerMethod
        numRestart: int
            Number of restarts with randomly chosen initial values

        Returns
        -------
        Optimizer
        """
        bestOptimizer = cls(function, initialParams, methods, **kwargs)
        bestOptimizer.execute()
        #
        for _ in range(numRestart):
            newInitialParams = Optimizer._setRandomValue(initialParams)
            newOptimizer = cls(function, newInitialParams, methods, **kwargs)
            newOptimizer.execute()
            if newOptimizer.rssq < bestOptimizer.rssq:
                bestOptimizer = newOptimizer
        return bestOptimizer
