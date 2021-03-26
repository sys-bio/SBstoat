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
ALL = "#all#"  # Model for all parameters


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

    @staticmethod
    def mkParameter(parameter):
        """
        Creates an lmfit parameter

        Parameters
        ----------
        parameter: Parameter or lmfit.Parameter
        
        Returns
        -------
        lmfit.Parameter
        """
        if isinstance(parameter, lmfit.Parameter):
            return parameter
        lmfitParameter = lmfit.Parameter(
              name=parameter.name,
              value=parameter.value,
              min=parameter.lower,
              max=parameter.upper)
        return lmfitParameter

    @staticmethod
    def mkParameters(parameters):
        """
        Creates lmfit.Parameters

        Parameters
        ----------
        parameters: list-Parameter
        
        Returns
        -------
        lmfit.Parameters
        """
        lmfitParameters = lmfit.Parameters()
        _ = [lmfitParameters.add(Parameter.mkParameter(p)) for p in parameters]
        return lmfitParameters
        

class ParameterManager():
    """
    Manages overlapping parameters for models.
    - modelDct: key is model name; value is lmfit.Parameters
    - parameterDct: key is parameter name: value lmfit.Parameter
    """

    def __init__(self, modelNames, parameterCollection):
        """
        Parameters
        ----------
        parameterCollection: list-lmfit.Parameters
        modelNames: list-str
            name of models corresponding to parameters
        """
        self.parameterDct = self._mkParameterDct(parameterCollection)
        self.modelDct = self._mkModelDct(modelNames, parameterCollection)

    def _mkParameterDct(self, parameterCollection):
        """
        The dictionary that relates parameter names to models.
        Where there are multiple occurrences of the same parameter,
        the min, max, and value of the parameter are adjusted.

        Returns
        -------
        dict
            key: str (parameter name)
            value: list-modelName
        """
        parameterDct = {}
        countDct = {}
        for parameters in parameterCollection:
            for parameterName, parameter in parameters.items():
                if not parameterName in parameterDct.keys():
                    newParameter = lmfit.Parameter(
                          name=parameter.name,
                          min=parameter.min,
                          max=parameter.max,
                          value=parameter.value)
                    parameterDct[parameterName] = newParameter
                    countDct[parameterName] = 1
                else:
                    # Adjust parameter values
                    curParameter = parameterDct[parameterName]
                    curParameter.set(min=min(curParameter.min, parameter.min))
                    curParameter.set(max=max(curParameter.max, parameter.max))
                    curParameter.set(value=curParameter.value + parameter.value)
                    countDct[parameterName] += 1
        for parameterName, parameter in parameterDct.items():
            parameter.set(value=parameter.value/countDct[parameterName])
        return parameterDct

    def _mkModelDct(self, modelNames, parameterCollection):
        """
        Ensures that use the same lmfit.Parameter object for shared
        parameters.

        Parameters
        ----------
        modelNames: list-str
        parameterCollection: list-lmfit.Parameters
        
        Returns
        -------
        dict
            key: modelName
            value: lmfit.Parameters
        """
        modelDct = {}
        for modelName, parameters in zip(modelNames, parameterCollection):
            modelParameters = lmfit.Parameters()
            for parameterName, _ in parameters.items():
                modelParameters.add(self.parameterDct[parameterName])
            modelDct[modelName] = modelParameters
        # Consruct parameters for ALL model
        parameters = lmfit.Parameters()
        [parameters.add(p) for p in self.parameterDct.values()]
        modelDct[ALL] = parameters
        return modelDct

    def updateValues(self, parameters):
        """
        Updates parameter values.

        Parameters
        ----------
        parameters: lmfit.Parameters
        """
        for parameterName, parameter in parameters.items():
            self.parameterDct[parameterName].set(value=parameter.value)

    def getParameters(self, modelName=ALL):
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
        return self.modelDct[modelName]
