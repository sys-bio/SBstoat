"""Implements an interface to optimizers."""

from SBstoat.logs import Logger
from SBstoat import _helpers
from SBstoat import _constants as cn

import collections
import lmfit
import numpy as np

_BestParameters = collections.namedtuple("_BestParameters",
      "params rssq")  #  parameters, residuals sum of squares
ParameterDescriptor = collections.namedtuple("ParameterDescriptor",
      "params method rssq kwargs minimizer minimizerResult")

class Optimizer(object):

    def __init__(self, function, initialParams, methods, logger=None):
        """
        Parameters
        ----------
        function: Funtion
           Function called repeatedly with arguments lmfit.parameters
           returns residuals
        initialParams: lmfit.parameters
        methods: list-_helpers.OptimizerMethod
        
        Returns
        -------
        """
        self._function = function
        self._methods = methods
        self._initialParams = initialParms
        if logger is None:
            self.logger = Logger()
        # Purely internal state
        self._bestParameters = None
        #
        self.params = None
        self.minimizer = None
        self.minimizerResult = None

    def optimize(self): 
        """
        Performs the optimization on the function.
        Result is self.params
        """
        paramResults = []
        lastExcp = None
        for idx, optimizerMethod in enumerate(self._methods):
            method = optimizerMethod.method
            kwargs = optimizerMethod.kwargs
            self._bestParameters = _BestParameters(params=None, rssq=None)
            minimizer = lmfit.Minimizer(self._function, params)
            try:
                minimizerResult = minimizer.minimize(
                      method=method, **kwargs)
            except Exception as excp:
                lastExcp = excp
                msg = "Error minimizing for method: %s" % method
                self.logger.error(msg, excp)
                continue
            params = self._bestParameters.params.copy()
            rssq = np.sum(self.function(params)**2)
            if len(paramResults) > idx:
                if rssq >= paramResults[idx].rssq:
                    continue
            parameterDescriptor = ParameterDescriptor(
                  params=params,
                  method=method,
                  rssq=rssq,
                  kwargs=dict(kwargs),
                  minimizer=minimizer,
                  minimizerResult=minimizerResult,
                  )
            paramResults.append(parameterDescriptor)
        if len(paramResults) == 0:
            msg = "*** Optimization failed."
            self.logger.error(msg, lastExcp)
        else:
            # Select the result that has the smallest residuals
            sortedMethods = sorted(paramResults, key=lambda r: r.rssq)
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
