"""
Harness used to test codes using SBML models.

Usage
-----
harness = TestHarness(sbmlPath, selectedColumns, parametersToFit)
harness.evaluate(stdResiduals=0.5, maxRelativeError=0.1)
"""

from SBstoat.modelFitter import ModelFitter
from SBstoat.namedTimeseries import NamedTimeseries, TIME
from SBstoat import _helpers
from SBstoat import _logger
from SBstoat.observationSynthesizer import ObservationSynthesizerRandomErrors

import numpy as np
import os
import tellurium as te
import typing


HTTP200 = 200
HTTP = "http://"


class TestHarness(object):

    def __init__(self, sbmlPath:str,
          parametersToFit:typing.List[str]=None,
          selectedColumns:typing.List[str]=None,
          **kwargs):
        """
        Parameters
        ----------
        sbmlPath: path or URL to SBML files
        selectedColumns: names of floating species used in fitting
        parametersToFit: names of parameters to fit
        kwargs: passed to ModelFitter constructor
        """
        self.sbmlPath = sbmlPath
        self.selectedColumns = selectedColumns
        self.parametersToFit = parametersToFit
        self.kwargs = kwargs
        self.roadRunner = self._initializeRoadrunner()
        self._validate()
        if self.parametersToFit is None:
            self.parametersToFit = list(self.roadRunner.getGlobalParameterIds())
        self.parameterValueDct = {p: self.roadRunner[p]
              for p in self.parametersToFit}
        if "logger" in kwargs.keys():
            self._logger = kwargs["logger"]
        else:
            self._logger = _logger.Logger()

    def _checkNamesInModel(self, names:typing.List[str], errorMsgPattern:str):
        """
        Verifies that the list of names is in the model.

        Parameters
        ----------
        names: what to check
        errorMsgPattern: string pattern taking one argument (name) for error msg
        
        Raises ValueError
        """
        if self.selectedColumns is None:
            return
        for name in self.selectedColumns:
            if not name in self.roadRunner.model.keys():
                raise ValueError(errorMsgPatter % name)

    def _initializeRoadrunner(self):
        """
        Initializes the object and does error checking.

        Return
        ------
        Roadrunner
        """
        try:
            rr = te.roadrunner.ExtendedRoadRunner(self.sbmlPath)
        except Exception as err:
            msg = "sbmlPath does is not a valid file path or URL: %s" \
                  % self.sbmlPath
            raise ValueError(msg)
        return te.loads(self.sbmlPath)

    def _validate(self):
        """
        Validates names of species and parameters
        """
        # Validate the column names
        errorMsgPattern = "Variable name is not in model: %s"
        self._checkNamesInModel(self.selectedColumns, errorMsgPattern)
        # Validate the parameter names
        errorMsgPatter = "Parameter name is not in model: %s"
        self._checkNamesInModel(self.parametersToFit, errorMsgPattern)

    def _checkParameters(self, params, relError):
        valuesDct = params.valuesdict()
        for name in self.parameterValueDct.keys():
            estimatedValue = valuesDct[name]
            actualRelError = _helpers.calcRelError(self.parameterValueDct[name],
                  estimatedValue)
            if actualRelError > relError:
                self._logger.result("Parameter %s has high relError: %2.3f"
                      % (name, actualRelError))

    def evaluate(self, stdResiduals:float=0.1, relError:float=0.1,
          endTime:float=10.0, numPoint:int=30,
          fractionParameterDeviation:float=0.5):
        """
        Evaluates model fitting accuracy and bootstrapping for model
 
        Parameters
        ----------
        stdResiduals: Standard deviations of variable used in constructing reiduals
        relError: relative error of parameter used in evaluation
        endTime: ending time of the simulation
        numPoint: number of points in the simulatin
        fractionParameterDeviation: fractional amount that the parameter can vary
        """
        msg = "Model %s" % self.sbmlPath
        msg += "\n   parameters: %s" % str(self.selectedColumns)
        self._logger.activity(msg)
        # Construct synthetic observations
        if self.selectedColumns is None:
            data = self.roadRunner.simulate(0, endTime, numPoint)
        else:
            allColumns = list(self.selectedColumns)
            if not TIME in allColumns:
                allColumns.append(TIME)
            data = self.roadRunner.simulate(0, endTime, numPoint, allColumns)
        simTS = NamedTimeseries(namedArray=data)
        synthesizer = ObservationSynthesizerRandomErrors(
              fittedTS=simTS, std=stdResiduals)
        observedTS = synthesizer.calculate()
        # Construct the parameter ranges
        parameterDct = {}
        for name in self.parameterValueDct.keys():
            lower = self.parameterValueDct[name]*(1 - fractionParameterDeviation)
            upper = self.parameterValueDct[name]*(1 + fractionParameterDeviation)
            value = np.random.uniform(lower, upper)
            parameterDct[name] = (lower, upper, value)
        # Create the fitter
        fitter = ModelFitter(self.roadRunner, observedTS,
              selectedColumns=self.selectedColumns, parameterDct=parameterDct,
              **self.kwargs)
        # Evaluate the fit
        fitter.fitModel()
        self._checkParameters(fitter.params, relError)
        # Evaluate bootstrap
        fitter.bootstrap(numIteration=100)
        self._checkParameters(fitter.bootstrapResult.params, relError)
