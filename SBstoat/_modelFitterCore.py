# -*- coding: utf-8 -*-
"""
 Created on August 18, 2020

@author: joseph-hellerstein

Core logic of model fitter. Does not include plots.
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME, mkNamedTimeseries
import SBstoat.timeseriesPlotter as tp
from SBstoat import namedTimeseries
from SBstoat import _helpers

import collections
import lmfit
import numpy as np
import pandas as pd
import random
import roadrunner
import tellurium as te
import typing

# Constants
PARAMETER_LOWER_BOUND = 0
PARAMETER_UPPER_BOUND = 10
#  Minimizer methods
METHOD_BOTH = "both"
METHOD_DIFFERENTIAL_EVOLUTION = "differential_evolution"
METHOD_LEASTSQ = "leastsqr"
MAX_CHISQ_MULT = 5
PERCENTILES = [2.5, 97.55]  # Percentile for confidence limits
INDENTATION = "  "
NULL_STR = ""
IS_REPORT = False


ParameterSpecification = collections.namedtuple("ParameterSpecification",
      "lower upper value")



##############################
class ModelFitterCore(object):

    def __init__(self, modelSpecification, observedData, parametersToFit,
                 selectedColumns=None, method=METHOD_BOTH,
                 parameterLowerBound=PARAMETER_LOWER_BOUND,
                 parameterUpperBound=PARAMETER_UPPER_BOUND,
                 parameterDct={},
                 fittedDataTransformDct={},
                 isPlot=True
                 ):
        """
        Parameters
        ---------
        modelSpecification: ExtendedRoadRunner/str
            roadrunner model or antimony model
        observedData: NamedTimeseries/str
            str: path to CSV file
        parametersToFit: list-str/None
            parameters in the model that you want to fit
            if None, no parameters are fit
        selectedColumns: list-str
            species names you wish use to fit the model
            default: all columns in observedData
        parameterLowerBound: float
            lower bound for the fitting parameters
        parameterUpperBound: float
            upper bound for the fitting parameters
        parameterDct: dict
            key: parameter name
            value: ParameterSpecification
        fittedDataTransformDct: dict
            key: column in selectedColumns
            value: function of the data in selectedColumns;
                   input: NamedTimeseries
                   output: array for the values of the column
        method: str
            method used for minimization

        Usage
        -----
        f = ModelFitter(roadrunnerModel, "observed.csv", ['k1', 'k2'])
        """
        self.modelSpecification = modelSpecification
        self.parametersToFit = parametersToFit
        self.lowerBound = parameterLowerBound
        self.upperBound = parameterUpperBound
        self.parameterDct = dict(parameterDct)
        self.observedTS = mkNamedTimeseries(observedData)
        self.fittedDataTransformDct = fittedDataTransformDct
        if selectedColumns is None:
            selectedColumns = self.observedTS.colnames
        self.selectedColumns = selectedColumns
        self._method = method
        self._isPlot = isPlot
        self._plotter = tp.TimeseriesPlotter(isPlot=self._isPlot)
        # The following are calculated during fitting
        self.roadrunnerModel = None
        self.minimizer = None  # lmfit.minimizer
        self.minimizerResult = None  # Results of minimization
        self.params = None  # params property in lmfit.minimizer
        self.fittedTS = self.observedTS.copy()  # Initialization of columns
        self.residualsTS = None  # Residuals for selectedColumns
        self.bootstrapResult = None  # Result from bootstrapping
        # Validation checks
        self._validateFittedDataTransformDct()

    def _validateFittedDataTransformDct(self):
        excess = set(self.fittedDataTransformDct.keys()).difference(
              self.selectedColumns)
        if len(excess) > 0:
            msg = "fittedDataTransformDct columns not in selectedColumns: %s"  \
                  % str(excess)
            raise ValueError(excess)

    def _updateFittedTS(self, data):
        """
        Updates the fittedTS taking into account required transformations.
 
        Parameters
        ----------
        data: np.ndarray
        """
        self.fittedTS[self.fittedTS.allColnames] = data
        for column, func in self.fittedDataTransformDct.items():
            self.fittedTS[column] = func(self.fittedTS)
        
    @staticmethod
    def addParameter(parameterDct: dict,
          name: str, lower: float, upper: float, value: float):
        """
        Adds a parameter to a list of parameters.

        Parameters
        ----------
        parameterDct: parameter dictionary to agument
        name: parameter name
        lower: lower range of parameter value
        upper: upper range of parameter value
        value: initial value
        
        Returns
        -------
        dict
        """
        parameterDct[name] = ParameterSpecification(
              lower=lower, upper=upper, value=value)

    def copy(self):
        """
        Creates a copy of the model fitter.
        """
        if not isinstance(self.modelSpecification, str):
            modelSpecification = self.modelSpecification.getAntimony()
        else:
            modelSpecification = self.modelSpecification
        newModelFitter = self.__class__(
              modelSpecification,
              self.observedTS,
              self.parametersToFit,
              selectedColumns=self.selectedColumns,
              method=self._method,
              parameterLowerBound=self.lowerBound,
              parameterUpperBound=self.upperBound,
              parameterDct=self.parameterDct,
              fittedDataTransformDct=self.fittedDataTransformDct,
              isPlot=self._isPlot)
        return newModelFitter

    def _initializeRoadrunnerModel(self):
        """
        Sets self.roadrunnerModel.
        """
        if isinstance(self.modelSpecification,
              te.roadrunner.extended_roadrunner.ExtendedRoadRunner):
            self.roadrunnerModel = self.modelSpecification
        elif isinstance(self.modelSpecification, str):
            self.roadrunnerModel = te.loada(self.modelSpecification)
        else:
            msg = 'Invalid model.'
            msg = msg + "\nA model must either be a Roadrunner model "
            msg = msg + "an Antimony model."
            raise ValueError(msg)

    def simulate(self, params=None, startTime=None, endTime=None, numPoint=None):
        """
        Runs a simulation. Defaults to parameter values in the simulation.

        Parameters
        ----------
        params: lmfit.Parameters
        startTime: float
        endTime: float
        numPoint: int

        Return
        ------
        NamedArray
        """
        def set(default, parameter):
            # Sets to default if parameter unspecified
            if parameter is None:
                return default
            else:
                return parameter
        #
        startTime = set(self.observedTS.start, startTime)
        endTime = set(self.observedTS.end, endTime)
        numPoint = set(len(self.observedTS), numPoint)
        if self.roadrunnerModel is None:
            self._initializeRoadrunnerModel()
        self.roadrunnerModel.reset()
        if params is not None:
          # Parameters have been specified
          self._setupModel(params)
        return self.roadrunnerModel.simulate(startTime, endTime, numPoint)

    def _simulate(self, **kwargs):
        """
        Runs a simulation.

        Parameters
        ----------
        kwargs: dict

        Instance Variables Updated
        --------------------------
        self.fittedTS
        """
        data = self.simulate(**kwargs)
        columnIndices = [i for i in range(len(data.colnames))
              if data.colnames[i][1:-1] in self.fittedTS.allColnames]
        columnIndices.insert(0, 0)
        self._updateFittedTS(data[:, columnIndices])

    def _residuals(self, params)->np.ndarray:
        """
        Compute the residuals between objective and experimental data

        Parameters
        ----------
        kwargs: dict
            arguments for simulation

        Instance Variables Updated
        --------------------------
        self.residualsTS

        Returns
        -------
        1-d ndarray of residuals
        """
        self._simulate(params=params)
        cols = self.selectedColumns
        if self.residualsTS is None:
            self.residualsTS = self.observedTS.subsetColumns(cols)
        self.residualsTS[cols] = self.observedTS[cols] - self.fittedTS[cols]
        residuals = self.residualsTS.flatten()
        return residuals

    def fitModel(self, params:lmfit.Parameters=None,
          max_nfev:int=100):
        """
        Fits the model by adjusting values of parameters based on
        differences between simulated and provided values of
        floating species.

        Parameters
        ----------
        params: starting values of parameters
        max_nfev: maximum number of function evaluations

        Example
        -------
        f.fitModel()
        """
        self._initializeRoadrunnerModel()
        if self.parametersToFit is None:
            # Compute fit and residuals for base model
            self.params = None
        else:
            if params is None:
                params = self._initializeParams()
            residuals_DE = self.observedTS.flatten()
            residuals_LS = residuals_DE
            # Fit the model to the data
            # Use two algorithms:
            #   Global differential evolution to get us close to minimum
            #   A local Levenberg-Marquardt to getsus to the minimum
            if self._method in [METHOD_BOTH, METHOD_DIFFERENTIAL_EVOLUTION]:
                minimizer = lmfit.Minimizer(self._residuals, params,
                      max_nfev=max_nfev)
                self.minimizerResult = minimizer.minimize(
                      method=METHOD_DIFFERENTIAL_EVOLUTION,
                      max_nfev=max_nfev)
                params_DE = self.minimizerResult.params
                residuals_DE = self._residuals(params=params_DE)
            if self._method in [METHOD_BOTH, METHOD_LEASTSQ]:
                minimizer = lmfit.Minimizer(self._residuals, params,
                      max_nfev=max_nfev)
                self.minimizerResult = minimizer.minimize(
                      method=METHOD_LEASTSQ,
                      max_nfev=max_nfev)
                params_LS = self.minimizerResult.params
                residuals_LS = self._residuals(params=params_LS)
            if np.std(residuals_DE) <= np.std(residuals_LS):
                self.params = params_DE
            else:
                self.params = params_LS
            self.minimizer = minimizer
            if not self.minimizer.success:
                msg = "*** Minimizer failed for this model and data."
                raise ValueError(msg)
        # Ensure that residualsTS and fittedTS match the parameters
        _ = self._residuals(params=self.params)

    def getFittedModel(self):
        """
        Provides the roadrunner model with fitted parameters

        Returns
        -------
        ExtendedRoadrunner
        """
        self._checkFit()
        self.roadrunnerModel.reset()
        self._setupModel(self.params)
        return self.roadrunnerModel

    def _setupModel(self, params):
        """
        Sets up the model for use based on the parameter parameters

        Parameters
        ----------
        params: lmfit.Parameters

        """
        pp = params.valuesdict()
        for parameter in self.parametersToFit:
            self.roadrunnerModel.model[parameter] = pp[parameter]

    def _initializeParams(self):
        params = lmfit.Parameters()
        value = np.mean([self.lowerBound, self.upperBound])
        for parameterName in self.parametersToFit:
            if parameterName in self.parameterDct.keys():
              specification = self.parameterDct[parameterName]
              params.add(parameterName,
                    value=specification.value,
                    min=specification.lower,
                    max=specification.upper,
                    )
            else:
              params.add(parameterName, value=value,
                    min=self.lowerBound, max=self.upperBound)
        return params

    def _checkFit(self):
        if self.params is None:
            raise ValueError("Must use fitModel before using this method.")