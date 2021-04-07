"""
Class that does fitting for a suite of related models.

A parameter has a lower bound, upper bound, and value.
Parameters are an lmfit collection of parameter.
A parameter collection is a collection of parameters.
"""

from SBstoat import _constants as cn
from SBstoat.modelFitter import ModelFitter
from SBstoat._optimizer import Optimizer
from SBstoat.logs import Logger
from SBstoat._serverManager import AbstractServer, ServerManager

import matplotlib.pyplot as plt
import numpy as np
import lmfit


class _Parameter():

    def __init__(self, name, lower, upper, value):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.value = value

    def updateLower(self, value):
        self.lower = min(self.lower, value)

    def updateUpper(self, value):
        self.upper = max(self.upper, value)


class _ParameterManager():
    """Manages overlapping parameters for models."""
    ALL = "#all#"  # Model for all parameters

    def __init__(self, modelNames, parameterCollection):
        """
        Parameters
        ----------
        parameterCollection: collection-lmfit.Parameters
        modelNames: list-str
            name of models corresponding to parameters
        """
        self.parametersCollection = parameterCollection
        self.modelNames = modelNames
        self.modelDct, self.parameterDct = self._mkDcts()
        pass

    def _mkDcts(self):
        """
        Constructs dictionaries for parameters and models.
        Constructs the ALL model.

        Returns
        -------
        parameterDct: dict
            key: str (parameter name)
            value: _Parameter
        modelDct: dict
            key: str (model name)
            value: list-str (parameter names)
        """
        parameterDct = {}
        modelDct = {}
        for modelName, parameters in zip(self.modelNames,
              self.parametersCollection):
            modelDct[modelName] = []
            for parameterName, parameter in parameters.items():
                modelDct[modelName].append(parameterName)
                if parameterName in parameterDct.keys():
                    parameterDct[parameterName].updateLower(parameter.min)
                    parameterDct[parameterName].updateUpper(parameter.max)
                else:
                    parameterDct[parameterName] = _Parameter(parameterName,
                          lower=parameter.min,
                          upper=parameter.max, value=parameter.value)
        modelDct[_ParameterManager.ALL] = list(parameterDct.keys())
        return modelDct, parameterDct

    def updateValues(self, parameters):
        """
        Updates parameter values.

        Parameters
        ----------
        parameters: lmfit.Parameters
        """
        for name, parameter in parameters.items():
            self.parameterDct[name].value = parameter.value

    def mkParameters(self, modelName=None):
        """
        Makes lmfit.Parameters for the model. If none, then constructs one
        for all parameters.

        Parameters
        ----------
        model: str

        Returns
        -------
        lmfit.Parameters
        """
        if modelName is None:
            modelName = _ParameterManager.ALL
        parameters = lmfit.Parameters()
        for parameterName in self.modelDct[modelName]:
            parameter = self.parameterDct[parameterName]
            parameters.add(parameter.name,
                  min=parameter.lower, max=parameter.upper, value=parameter.value)
        return parameters


class ResidualsServer(AbstractServer):

    """
    server = ResidualsServer(fitter, inputQ, outputQ)
    server.run()
    while not done:
        inputQ.put(params)
        residualsArr = outputQ.get()
    server.terminate()
    """

    def __init__(self, fitter, inputQ, outputQ, logger=Logger()):
        """
        Parameters
        ----------
        fitter: ModelFitter
            cannot have swig objects (e.g., roadrunner)
        inputQ: multiprocessing.queue
        outputQ: multiprocessing.queue
        logger: Logger
        """
        super().__init__(fitter, inputQ, outputQ, logger=logger)
        self.fitter = fitter
   
    def runFunction(self, params):
        """
        
        Parameters
        ----------
        params: lmfit.parameters
            Parameters for this simulation
        
        Returns
        -------
        np.array: residuals
        """
        self.fitter.initializeRoadRunnerModel()
        residuals = self.fitter.calcResiduals(params)
        return residuals/np.size(residuals)


class SuiteFitterCore():

    def __init__(self, modelFitters, modelNames=None, modelWeights=None,
          fitterMethods=None, numRestart=0, isParallel=False, logger=Logger()):
        """
        Parameters
        ----------
        modelFitters: list-modelFiter
        modelWeights: list-float
            how models are weighted in least squares
        modelNames: list-str
        fitterMethods: list-optimization methods
        numRestart: int
            number of times the minimization is restarted with random
            initial values for parameters to fit.
        isParallel: bool
            runs each fitter in parallel
        logger: Logger

        Raises
        ------
        ValueError: len(modelNames) == len(modelFitters)
        """
        self.numModel = len(modelFitters)
        self.modelWeights = modelWeights
        if self.modelWeights is None:
            self.modelWeights = np.repeat(1, self.numModel)
        self.modelNames = modelNames
        if self.modelNames is None:
            self.modelNames = [str(v) for v in range(len(modelSpecifications))]
        self._numRestart = numRestart
        self._isParallel = isParallel
        self.logger = logger
        # Validation checks
        if self.numModel != len(self.modelNames):
            msg = "Length of modelNames must be same as number of modelFitters."
            raise ValueError(msg)
        #
        self.fitterDct = {n: f for n, f in zip(modelNames, modelFitters)}
        # Construct tha parameters for each model
        self.parametersCollection = [f.params for f in self.fitterDct.values()]
        self.parameterManager = _ParameterManager(self.modelNames,
              self.parametersCollection)
        self._fitterMethods = ModelFitter.makeMethods(
              fitterMethods, cn.METHOD_FITTER_DEFAULTS)
        # Residuals calculations
        self.residualsServers = [ResidualsServer(f, None, None,
              logger=self.logger) for f in self.fitterDct.values()]
        self.manager = None
        # Results
        self.optimizer = None

    @property
    def params(self):
        if self.optimizer is not None:
            return self.optimizer.params
        return None

    def clean(self):
        if self.manager is not None:
            self.manager.stop()

    def _calcResiduals(self, parameters):
        """
        Calculates the residuals for models in the suite. The residuals are the
        a concatenation of the residuals for each model. Residuals are normalized by
        the number of elements in the model residuals.

        Parameters
        ----------
        parameters: lmfit.Parameters

        Returns
        -------
        list-np.ndarray
            residuals for each model
        """
        self.parameterManager.updateValues(parameters)
        residualsCollection = []
        parametersList = [self.parameterManager.mkParameters(modelName=n)
              for n in self.fitterDct.keys()]
        if self._isParallel and (self.manager is not None):
            residualsCollection = self.manager.submit(parametersList)
        else:
            residualsCollection = [s.runFunction(p) for s, p in 
                  zip(self.residualsServers, parametersList)]
        normalizedCollection = [w*a for w, a in zip(self.modelWeights,
              residualsCollection)]
        return normalizedCollection

    def calcResiduals(self, parameters):
        """
        Calculates the residuals for models in the suite. The residuals are the
        a concatenation of the residuals for each model. Residuals are normalized by
        the number of elements in the model residuals.

        Parameters
        ----------
        parameters: lmfit.Parameters

        Returns
        -------
        np.ndarray
        """
        return np.concatenate(self._calcResiduals(parameters))

    def fitSuite(self, params:lmfit.Parameters=None):
        """
        Fits the model by adjusting values of parameters based on
        differences between simulated and provided values of
        floating species.

        Parameters
        ----------
        params: lmfit.parameters
            starting values of parameters
        max_nfev: int
            Maximum number of iterations for an evaluation

        Example
        -------
        f.fitSuite()
        """
        if params is None:
            initialParameters = self.parameterManager.mkParameters()
        else:
            initialParameters = params.copy()
        # Setup parallel servers if needed
        if self._isParallel:
            fitters = [f.copy() for f in self.fitterDct.values()]
            self.manager = ServerManager(ResidualsServer, fitters,
                  logger=self.logger)
        # Do the optimization
        self.optimizer = Optimizer.optimize(self.calcResiduals, initialParameters,
              self._fitterMethods, logger=self.logger, isCollect=True,
              numRestart=self._numRestart)
        # Assign fitter results to each model
        self.parameterManager.updateValues(self.params)
        for modelName, fitter in self.fitterDct.items():
            fitter.suiteFitterParams = self.parameterManager.mkParameters(
                  modelName=modelName)
        # Clean up
        if self._isParallel:
            self.manager.stop()
            self.manager = None

    def reportFit(self):
        """

        Parameters
        ----------

        Returns
        -------
        """
        return ModelFitter.reportTheFit(self.optimizer.minimizerResult,
              self.params)

    def plotResidualsSSQ(self, isPlot=True):
        """
        Plots residuals SSQ for ihe models.

        Parameters
        ----------
        isPlot: bool
        """
        fig, ax = plt.subplots(1)
        rssqs = [np.sum(v**2) for v in self._calcResiduals(self.params)]
        totalRssq = np.sum(rssqs)
        fracs = rssqs/totalRssq
        ax.bar(self.modelNames, fracs)
        ax.set_xlabel("model")
        ax.set_ylabel("fraction of total sum of squares")
        ax.set_title("Total Sum of Squares: %f" % totalRssq)
        #
        if isPlot:
            plt.show()

    def plotFitAll(self, isPlot=True, **kwargs):
        """
        Plots fits for all models

        Parameters
        ----------
        isPlot: bool
        kwargs: dict
            keyword arguments pased to plotFitAll for fitters
        """
        if not isPlot:
            return
        for model, fitter in self.fitterDct.items():
            if fitter is None:
                print("\n\n%s\n %s: Could not fit model." % model)
            else:
                newKwargs = dict(kwargs)
                if not cn.PARAMS in newKwargs:
                    newKwargs[cn.PARAMS] = fitter.suiteFitterParams
                print("\n\n%s\n" % model)
                fitter.plotFitAll(**newKwargs)
