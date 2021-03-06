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


class SuiteFitter():

    def __init__(self, modelSpecifications, datasets, parameterNamesCollection,
          modelNames=None, modelWeights=None, fitterMethods=None,
          **kwargs):
        """
        Parameters
        ----------
        models: list-modelSpecification argument for ModelFitter
        datasets: list-observedData argument for ModelFitter
        parameterNamesCollection: list-iparametersToFit argument for modelFitter
        modelWeights: list-float
            how models are weighted in least squares
        modelNames: list-str
        kwargs: dict
            constructor arguments passed to fitter for a model

        Raises
        ------
        ValueError: len(models) == len(parametersCollection) == len(datasets)
                     == len(modelNames)
        """
        # Mandatory parameters
        self.modelSpecifications = modelSpecifications
        self.datasets = datasets
        self.parameterNamesCollection = parameterNamesCollection
        #
        self.modelWeights = modelWeights
        if self.modelWeights is None:
            self.modelWeights = np.repeat(1, len(self.modelSpecifications))
        self.modelNames = modelNames
        if self.modelNames is None:
            self.modelNames = [str(v) for v in range(len(modelSpecifications))]
        # Derived values
        self.numModel = len(self.modelSpecifications)
        # Validation checks
        if self.numModel != len(self.datasets):
            raise ValueError("Number of datasets must equal number of models")
        if self.numModel != len(self.parameterNamesCollection):
            msg = "Number of parametersNameCollection must equal number of models"
            raise ValueError(msg)
        if self.numModel != len(self.modelWeights):
            raise ValueError("Number of modelWeights must equal number of models")
        if self.numModel != len(self.modelNames):
            raise ValueError("Number of modelNames must equal number of models")
        #
        self.fitterDct = {}  # key: model name, value: ModelFitter
        self.logger = Logger()
        if cn.LOGGER not in kwargs.keys():
            kwargs[cn.LOGGER] = self.logger
        for modelSpecification, dataset, parametersToFit, modelName in   \
              zip(self.modelSpecifications, self.datasets,
              self.parameterNamesCollection, self.modelNames):
            self.fitterDct[modelName] = ModelFitter(modelSpecification, dataset,
                  parametersToFit=parametersToFit,
                  **kwargs)
        # Construct tha parameters for each model
        self.parametersCollection = [f.mkParams(c) for f, c
              in zip(self.fitterDct.values(), self.parameterNamesCollection)]
        self.parameterManager = _ParameterManager(self.modelNames,
              self.parametersCollection)
        self._fitterMethods = ModelFitter.makeMethods(
              fitterMethods, cn.METHOD_FITTER_DEFAULTS)
        # Results
        self.optimizer = None

    @property
    def params(self):
        if self.optimizer is not None:
            return self.optimizer.params
        return None

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
        for modelName, fitter in self.fitterDct.items():
            fitter.initializeRoadRunnerModel()
            parameters = self.parameterManager.mkParameters(modelName=modelName)
            residuals = fitter.calcResiduals(parameters)
            residuals = residuals/np.size(residuals)
            residualsCollection.append(residuals)
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
        self.optimizer = Optimizer(self.calcResiduals, initialParameters,
              self._fitterMethods, logger=self.logger, isCollect=True)
        self.optimizer.optimize()
        # Assign fitter results to each model
        self.parameterManager.updateValues(self.params)
        for modelName, fitter in self.fitterDct.items():
            fitter.params = self.parameterManager.mkParameters(
                  modelName=modelName)

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
            print("\n\n%s\n" % model)
            fitter.plotFitAll(**kwargs)
