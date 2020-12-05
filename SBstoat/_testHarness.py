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
from SBstoat import logs
from SBstoat.observationSynthesizer import ObservationSynthesizerRandomErrors

import numpy as np
import os
import tellurium as te
import typing


HTTP200 = 200
HTTP = "http://"
MAX_PARAMETER = 5  # Maximum number of parameters estimated
NUM_BOOTSTRAP_ITERATION = 100


class TestHarnessResult(object):

    def __init__(self):
        self.parameterRelErrorDct = {}

    def addParameter(self, name, relerror):
        self.parameterRelErrorDct[name] = relerror

    def __repr__(self):
        return str(self.parameterRelErrorDct)


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
        if "logger" in kwargs.keys():
            self.logger = kwargs["logger"]
        else:
            self.logger = logs.Logger()
        #
        self.sbmlPath = sbmlPath
        self.roadRunner = self._initializeRoadrunner()
        self.selectedColumns = selectedColumns
        self.parametersToFit = self._getSetableParameters(parametersToFit)
        self.kwargs = kwargs
        self.parametersToFit = self._getSetableParameters(parametersToFit)
        self.parameterValueDct = {p: self.roadRunner[p]
              for p in self.parametersToFit}
        self.fitModelResult = TestHarnessResult()
        self.bootstrapResult = TestHarnessResult()
        self._validate()

    def _getSetableParameters(self, initialParametersToFit,
          maxParameter=MAX_PARAMETER):
        if initialParametersToFit is None:
            parameters = list(self.roadRunner.getGlobalParameterIds())
        else:
            parameters = initialParametersToFit
        # See which parameters can be changed
        parametersToFit = []
        for parameter in parameters:
            try:
                self.roadRunner.model[parameter] = self.roadRunner.model[parameter]
                parametersToFit.append(parameter)
            except Exception:
                pass
        #
        if len(parametersToFit) > maxParameter:
             parametersToFit = parametersToFit[:maxParameter]
        return parametersToFit

    def _checkNamesInModel(self, names:typing.List[str], errorMsgPattern:str):
        """
        Verifies that the list of names is in the model.

        Parameters
        ----------
        names: what to check
        errorMsgPattern: string pattern taking one argument (name) for error msg
        
        Raises ValueError
        """
        diffs = set(names).difference(self.roadRunner.model.keys())
        if len(diffs) > 0:
            raise ValueError(errorMsgPatter % str(diffs))

    def _initializeRoadrunner(self):
        """
        Initializes the object and does error checking.

        Return
        ------
        Roadrunner
        """
        try:
            rr = te.loads(self.sbmlPath)
        except Exception as err:
            msg = "sbmlPath is not a valid file path or URL: %s" \
                  % self.sbmlPath
            self.logger.error("_initializeRoadrunner", err)
            raise ValueError(msg)
        return te.loads(self.sbmlPath)

    def _validate(self):
        """
        Validates names of species and parameters
        """
        # Validate the column names
        errorMsgPattern = "Variable name is not in model: %s"
        if self.selectedColumns is not None:
            self._checkNamesInModel(self.selectedColumns, errorMsgPattern)
        # Validate the parameter names
        errorMsgPatter = "Parameter name is not in model: %s"
        self._checkNamesInModel(self.parametersToFit, errorMsgPattern)

    def _recordResult(self, params, relError, testHarnessResult):
        valuesDct = params.valuesdict()
        for name in self.parameterValueDct.keys():
            estimatedValue = valuesDct[name]
            actualRelError = _helpers.calcRelError(self.parameterValueDct[name],
                  estimatedValue)
            testHarnessResult.addParameter(name, actualRelError)
            if actualRelError > relError:
                self.logger.result("Parameter %s has high relError: %2.3f"
                      % (name, actualRelError))

    def evaluate(self, stdResiduals:float=0.1, relError:float=0.1,
          endTime:float=10.0, numPoint:int=30,
          fractionParameterDeviation:float=0.5,
          numIteration=NUM_BOOTSTRAP_ITERATION):
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
        # Construct synthetic observations
        if self.selectedColumns is None:
            data = self.roadRunner.simulate(0, endTime, numPoint)
        else:
            allColumns = list(self.selectedColumns)
            if not TIME in allColumns:
                allColumns.append(TIME)
            data = self.roadRunner.simulate(0, endTime, numPoint, allColumns)
        bracketTime = "[%s]" % TIME
        if bracketTime in data.colnames:
            # Exclude any column named '[time]'
            idx = data.colnames.index(bracketTime)
            dataArr = np.delete(data, idx, axis=1)
            colnames = list(data.colnames)
            colnames.remove(bracketTime)
            colnames = [s[1:-1] if s != TIME else s for s in colnames]
            simTS = NamedTimeseries(array=dataArr, colnames=colnames)
        else:
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
        msg = "Fitting the parameters %s" % str(self.parameterValueDct.keys())
        self.logger.result(msg)
        # Evaluate the fit
        fitter.fitModel()
        self._recordResult(fitter.params, relError, self.fitModelResult)
        # Evaluate bootstrap
        fitter.bootstrap(numIteration=numIteration)
        if fitter.bootstrapResult is not None:
            if fitter.bootstrapResult.numSimulation > 0:
                self._recordResult(fitter.bootstrapResult.params,
                      relError, self.bootstrapResult)
