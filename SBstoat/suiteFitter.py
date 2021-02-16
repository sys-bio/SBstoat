"""Class that does fitting for a suite of related models."""

"""
A parameter has a lower bound, upper bound, and value.
Parameters are an lmfit collection of parameter.
A parameter collection is a collection of parameters.
"""

import SBstoat._constants as cn
from SBstoat.modelFitter import modelFitter
from SBstoat.logs import Logger

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
        self.lower = max(self.upper, value)


class _ParameterManager():
    """Manages overlapping parameters for models."""
    ALL = "#all#"  # Model for all parameters

    def __init__(self, models, parameterCollection):
        """
        Parameters
        ----------
        parameterCollection: collection-lmfit.Parameters
        models: list-str
            name of models corresponding to parameters
        """
        self.parameterCollection = parameterCollection
        self.models = models
        self.modelDct, self.parameterDct = self._mkDcts()

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
        for model, parameters in zip(self.models, self.parameterCollection):
            modelDct[model] = []
            for name, parameter in parameters.items():
               modelDct[model].append(name)
               if name in dct.keys():
                   parameterDct[name].updateLower(parameter.min)
                   parameterDct[name].updateUpper(parameter.max)
               else:
                   parameterDct[name] = _Parameter(name, parameter.min,
                         parameter.max, parameter.value)
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

    def mkParameters(model=None):
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
        if model is None:
            model = _ParameterManager.ALL
        parameters = lmfit.Parameters()
        for parameter in self.modelDct[model]:
            parameters.add(parameter.name,
                  min=parameter.lower, max=parameter.upper, value=parameter.value)
        return parameters
        

class SuiteFitter(object):

    def __init__(self, models, datasets, parametersSpecifications,
          modelNames=None, modelWeights=None, fitterMethods=None,
          **kwargs):
        """
        Parameters
        ----------
        models: list-modelSPecification argument for ModelFitter
        datasets: list-observedData argument for ModelFitter
        parametersSpecifications: list-iparametersToFit argument for modelFitter
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
        self.models = models
        self.datasets = datasets
        self.parametersCollection = [ModelFitter.mkParameters(p)
              for p in parametersDescriptions]
        #
        self.modelWeights = modelWeights
        if self.modelWeights is None:
            self.modelWeights = np.repeat(1, len(models))
        self.modelNames = modelNames
        if self.modelNames is None:
            self.modelNames = [str(v) for v in np.repeat(1, len(models)]
        # Validation checks
        if len(models) != len(datasets):
            raise ValueError("Number of datasets must equal number of models")
        if len(models) != len(parametersCollection):
            msg = "Number of parametersCollection must equal number of models"
            raise ValueError(msg)
        if len(models) != len(modelWeights):
            raise ValueError("Number of modelWeights must equal number of models")
        if len(models) != len(modelNames):
            raise ValueError("Number of modelNames must equal number of models")
        #
        self.fitterDct = {}  # key: model name, value: ModelFitter
        self.logger = Logger()
        if not cn.LOGGER in kwargs.keys():
            kwargs[cn.LOGGER] = self.logger
        for model, dataset, parametersToFit, modelName in   \
              zip(models, datatsets, parametersCollection, modelNames):
            self.fitterDct[modelName] = ModelFitter(model, dataset,
                  parametersToFit=parametersToFit,
                  **kwargs)
        self.parameterManager = _ParameterManager(self.models,
              self.parametersCollection)
        self._fitterMethods = fitterMethods
        # Results
        self.params = None
        self.minimizerResult = None
        self.minimizer = None

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
        self.parameterManager.update(parameters)
        residualsCollection = []
        for model, fitter in self.fitterDct.items():
            parameters = self.parameterManager.mkParameters(model=model)
            residuals = fitter.calcResiduals(parameters)
            residuals = residuals/np.size(residuals)
            residualsCollection.append(residuals)
        return np.concatenate(residualsCollection)

    def fitSuite(self, params:lmfit.Parameters=None, max_nfev:int=100):
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
        initalParameters = self.parameterManager.mkParameters()
        optimizer = Optimizer(self.calcResiduals, params, self._fitterMethods,
              logger=self.logger)
        optimizer.optimize()
        self.params = optimizer.params.copy()
        self.minimizer = optimizer.minimizer
        self.minimizerResult = optimizer.minimizerResult
    
    def reportFit(self):    
        """
        
        Parameters
        ----------
        
        Returns
        -------
        """

    def plotResidualsSSQ(self)
        """
        Plots residuals SSQ for ihe models.
        """
