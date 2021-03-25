"""
Manages parameters for models.

A parameter has a lower bound, upper bound, and value.
Parameters are an lmfit collection of parameter.
A parameter collection is a collection of parameters.
"""

from SBstoat import _constants as cn

import matplotlib.pyplot as plt
import numpy as np
import lmfit

LOWER_PARAMETER_MULT = 0.95
UPPER_PARAMETER_MULT = 1.05


class Parameter():

    def __init__(self, name, lower=cn.PARAMETER_LOWER_BOUND,
              value=None, upper=cn.PARAMETER_UPPER_BOUND):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.value = value
        if value is None:
            self.value = (lower + upper)/2.0
        if self.value <= self.lower:
            self.lower = LOWER_PARAMETER_MULT*self.value
        if self.value >= self.upper:
            self.upper = UPPER_PARAMETER_MULT*self.value
        if np.isclose(self.lower, 0.0):
            self.lower = -0.001
        if np.isclose(self.upper, 0.0):
            self.upper = 0.001

    def __str__(self):
        return self.name

    def copy(self, name=None):
        if name is None:
            name = self.name
        return Parameter(name, lower=self.lower, upper=self.upper,
              value=self.value)

    def updateLower(self, value):
        self.lower = min(self.lower, value)

    def updateUpper(self, value):
        self.upper = max(self.upper, value)


class ParameterManager():
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
            value: Parameter
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
                    parameterDct[parameterName] = Parameter(parameterName,
                          lower=parameter.min,
                          upper=parameter.max, value=parameter.value)
        modelDct[ParameterManager.ALL] = list(parameterDct.keys())
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
            modelName = ParameterManager.ALL
        parameters = lmfit.Parameters()
        for parameterName in self.modelDct[modelName]:
            parameter = self.parameterDct[parameterName]
            parameters.add(parameter.name,
                  min=parameter.lower, max=parameter.upper, value=parameter.value)
        return parameters
