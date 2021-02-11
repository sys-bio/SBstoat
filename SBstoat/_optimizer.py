from SBstoat.logs import Logger
from SBstoat import _helpers
from SBstoat import _constants as cn

import collections
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


_BestParameters = collections.namedtuple("_BestParameters",
      "params rssq")  #  parameters, residuals sum of squares
_ParameterDescriptor = collections.namedtuple("_ParameterDescriptor",
      "params method rssq kwargs minimizer minimizerResult")


class _FunctionWrapper(object):
    """Wraps a function used for optimization."""

    def __init__(self, function, isPerf=False):
        """
        Parameters
        ----------
        function: function
            function callable by lmfit.Minimizer
               argument: lmfit.Parameter
               returns: np.array
        isPerf: bool
            collect performance statistics on function execution
        """
        self._function = function
        self._isPerf = isPerf
        # Results
        self.statistics = []  # durations of function executions
        self.rssq = 10e10
        self.bestParams = None

    def execute(self, params, **kwargs):
        if self._isPerf:
            startTime = time.time()
        result = self._function(params, **kwargs)
        if self._isPerf:
            self.statisticss.append(time.time() - startTime)
        rssq = sum(result**2)
        if rssq < self.rssq:
            self.rssq = rssq
            self.bestParams = params.copy()
        return result
        

class Optimizer(object):
    """
    Implements an interface to optimizers with abstractions
    for multiple methods and performance reporting.
    The class also handles an oddity with lmfit that the final parameters
    returned may not be the best.
 
    Usage
    -----
    optimizer = Optimizer(calcResiduals, params, [cn.METHOD_LEASTSQ])
    optimizer.optimize()
    """

    def __init__(self, function, initialParams, methods, logger=None,
          isPerf=False):
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
        isPerf: bool
           Collects performance statistcs
        
        Returns
        -------
        """
        self._function = function
        self._methods = methods
        self._initialParams = initialParams
        self._isPerf = isPerf
        if logger is None:
            self.logger = Logger()
        # Purely internal state
        self._bestParameters = None
        self._currentMethodIndex = None
        # Outputs
        self.statisticss = []  # list of performance results
        self.params = None
        self.minimizer = None
        self.minimizerResult = None

    def optimize(self): 
        """
        Performs the optimization on the function.
        Result is self.params
        """
        descriptors = []
        lastExcp = None
        for idx, optimizerMethod in enumerate(self._methods):
            self._currentMethodIndex = idx
            method = optimizerMethod.method
            kwargs = optimizerMethod.kwargs
            params = self._initialParams.copy()
            wrapperFunction = _FunctionWrapper(self._function, isPerf=self._isPerf)
            minimizer = lmfit.Minimizer(wrapperFunction.execute, params)
            try:
                minimizerResult = minimizer.minimize(
                      method=method, **kwargs)
            except Exception as excp:
                lastExcp = excp
                msg = "Error minimizing for method: %s" % method
                self.logger.error(msg, excp)
                continue
            parameterDescriptor = _ParameterDescriptor(
                  params=wrapperFunction.bestParams.copy(),
                  method=method,
                  rssq=wrapperFunction.rssq,
                  kwargs=dict(kwargs),
                  minimizer=minimizer,
                  minimizerResult=minimizerResult,
                  )
            self.statisticss.append(wrapperFunction.statistics)
            descriptors.append(parameterDescriptor)
        if len(descriptors) == 0:
            msg = "*** Optimization failed."
            self.logger.error(msg, lastExcp)
        else:
            # Select the result that has the smallest residuals
            sortedMethods = sorted(descriptors, key=lambda r: r.rssq)
            bestMethod = sortedMethods[0]
            self.params = bestMethod.params
            self.minimizer= bestMethod.minimizer
            self.minimizerResult = bestMethod.minimizerResult

    def report(self):
        """
        Reports the result of an optimization.
        
        Returns
        -------
        str
        """
        VARIABLE_STG = "[[Variables]]"
        CORRELATION_STG = "[[Correlations]]"
        if self.minimizer is None:
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
            else:
                trimmedReportSplit.append(line)
        # Construct the report
        newReportSplit = [VARIABLE_STG]
        newReportSplit.extend(valuesStg.split("\n"))
        newReportSplit.extend(trimmedReportSplit)
        return "\n".join(newReportSplit)

    def plotPerformance(self):
        """
        Plots the statistics for running the objective function.
        """
        if not self._isPerfReport:
            msg = "Must construct with isPerfReport = True "
            msg += "to get performance report."
            raise ValueError(msg)
        # Compute statistics
        TOT = "Tot"
        CNT = "Cnt"
        AVG = "Avg"
        IDX = "Idx"
        totalTimes = [sum(v) for v in self.statisticss]
        counts = [len(v) for v in self.statisticss]
        averages = [np.mean(v) for v in self.statisticss]
        df = pd.DataFrame({
            IDX: range(len(self.statisticss)),
            TOT: totalTimes,
            CNT: counts,
            AVG: averages,
            })
        #
        fig, axes = plt.subplots(1, 3)
        df.plot.bar(x=IDX, y=TOT, ax=axes[0], title="Total time")
        df.plot.bar(x=xvalues, y=AVG, ax=axes[1], title="Average time")
        df.plot.bar(x=xvalues, y=CNT, ax=axes[2], title="Number calls")
        plt.show()

    @staticmethod
    def mkOptimizerMethod(methodNames=None, methodKwargs=None, maxFev=100):
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
        methodKwargs = np.repeat(newMethodKwargs, len(methodNames))
        #
        result = [_helpers.OptimizerMethod(n, k) for n, k  \
              in zip(methodNames, methodKwargs)]
        return result
