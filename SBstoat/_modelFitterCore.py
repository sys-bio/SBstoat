# -*- coding: utf-8 -*-
"""
 Created on August 18, 2020

@author: joseph-hellerstein

Core logic of model fitter. Does not include plots.
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME, mkNamedTimeseries
from SBstoat.logs import Logger
import SBstoat.timeseriesPlotter as tp
from SBstoat import rpickle

import collections
import copy
import lmfit
import numpy as np
import tellurium as te
import typing

# Constants
PARAMETER_LOWER_BOUND = 0
PARAMETER_UPPER_BOUND = 10
#  Minimizer methods
METHOD_DIFFERENTIAL_EVOLUTION = "differential_evolution"
METHOD_BOTH = "both"
METHOD_LEASTSQ = "leastsq"
METHOD_FITTER_DEFAULTS = [METHOD_DIFFERENTIAL_EVOLUTION, METHOD_LEASTSQ]
METHOD_BOOTSTRAP_DEFAULTS = [METHOD_LEASTSQ,
      METHOD_DIFFERENTIAL_EVOLUTION, METHOD_LEASTSQ]
MAX_CHISQ_MULT = 5
PERCENTILES = [2.5, 97.55]  # Percentile for confidence limits
INDENTATION = "  "
NULL_STR = ""
IS_REPORT = False
LOWER_PARAMETER_MULT = 0.95
UPPER_PARAMETER_MULT = 0.95
LARGE_RESIDUAL = 1000000


_BestParameters = collections.namedtuple("_BestParameters",
      "params rssq")  #  parameters, residuals sum of squares



##############################
class ParameterSpecification():

    def __init__(self, lower=None, value=None, upper=None):
        self.lower = lower
        self.value = value
        self.upper = upper


class ModelFitterCore(rpickle.RPickler):

    # Subclasses used in interface
    class OptimizerMethod():

        def __init__(self, method, kwargs):
            self.method = method
            self.kwargs = kwargs


    def __init__(self, modelSpecification, observedData,
          parametersToFit=None,
          selectedColumns=None,
          fitterMethods=None,
          numFitRepeat=1,
          bootstrapMethods=None,
          parameterLowerBound=PARAMETER_LOWER_BOUND,
          parameterUpperBound=PARAMETER_UPPER_BOUND,
          parameterDct=None,
          fittedDataTransformDct=None,
          logger=Logger(),
          isPlot=True,
          _loggerPrefix="",
          # The following must be kept in sync with ModelFitterBootstrap.bootstrap
          numIteration:int=10,
          reportInterval:int=1000,
          maxProcess:int=None,
          serializePath:str=None,
          ):
        """
        Constructs estimates of parameter values.

        Parameters
        ----------
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
            value: triple - (lowerVange, startingValue, upperRange)
        fittedDataTransformDct: dict
            key: column in selectedColumns
            value: function of the data in selectedColumns;
                   input: NamedTimeseries
                   output: array for the values of the column
        logger: Logger
        fitterMethods: str/list-str/list-OptimizerMethod
            method used for minimization in fitModel
        numFitRepeat: int
            number of times fitting is repeated for a method
        bootstrapMethods: str/list-str/list-OptimizerMethod
            method used for minimization in bootstrap
        numIteration: number of bootstrap iterations
        reportInterval: number of iterations between progress reports
        maxProcess: Maximum number of processes to use. Default: numCPU
        serializePath: Where to serialize the fitter after bootstrap

        Usage
        -----
        parameterDct = {
            "k1": (1, 5, 10),  # name of parameter: low value, initial, high
            "k2": (2, 3, 6)}
        ftter = ModelFitter(roadrunnerModel, "observed.csv",
            parameterDct=parameterDct)
        fitter.fitModel()  # Do the fit
        fitter.bootstrap()  # Estimate parameter variance with bootstrap
        """
        if modelSpecification is not None:
            # Not the default constructor
            self._loggerPrefix = _loggerPrefix
            self.modelSpecification = modelSpecification
            self.parametersToFit = parametersToFit
            self.lowerBound = parameterLowerBound
            self.upperBound = parameterUpperBound
            self.bootstrapKwargs = dict(
                  numIteration=numIteration,
                  reportInterval=reportInterval,
                  maxProcess=maxProcess,
                  serializePath=serializePath,
                  )
            self.parameterDct = ModelFitterCore._updateParameterDct(parameterDct)
            self._numFitRepeat = numFitRepeat
            if self.parametersToFit is None:
                self.parametersToFit = list(self.parameterDct.keys())
            self.observedTS = observedData
            if self.observedTS is not None:
                self.observedTS = mkNamedTimeseries(observedData)
            #
            self.fittedDataTransformDct = fittedDataTransformDct
            #
            if (selectedColumns is None) and (self.observedTS is not None):
                selectedColumns = self.observedTS.colnames
            self.selectedColumns = selectedColumns
            if self.observedTS is not None:
                self._observedArr = self.observedTS[self.selectedColumns].flatten()
            else:
                self._observedArr = None
            # Other internal state
            self._fitterMethods = self._makeMethods(fitterMethods,
                  METHOD_FITTER_DEFAULTS)
            self._bootstrapMethods = self._makeMethods(bootstrapMethods,
                  METHOD_BOOTSTRAP_DEFAULTS)
            if isinstance(self._bootstrapMethods, str):
                self._bootstrapMethods = [self._bootstrapMethods]
            self._isPlot = isPlot
            self._plotter = tp.TimeseriesPlotter(isPlot=self._isPlot)
            self._plotFittedTS = None  # Timeseries that is plotted
            self.logger = logger
            # The following are calculated during fitting
            self.roadrunnerModel = None
            self.minimizer = None  # lmfit.minimizer
            self.minimizerResult = None  # Results of minimization
            self.params = None  # params property in lmfit.minimizer
            self.fittedTS = self.observedTS.copy(isInitialize=True)  # Initialize
            self.residualsTS = None  # Residuals for selectedColumns
            self.bootstrapResult = None  # Result from bootstrapping
            # Validation checks
            self._validateFittedDataTransformDct()
            self._bestParameters = _BestParameters(rssq=None, params=None)
        else:
            pass

    def _makeMethods(self, methods, default):
        """
        Creates a method dictionary.

        Parameters
        ----------
        methods: str/list-str/dict
            method used for minimization in fitModel
            dict: key-method, value-optional parameters

        Returns
        -------
        list-OptimizerMethod
            key: method name
            value: dict of optional parameters
        """
        if methods is None:
            methods = default
        if isinstance(methods, str):
            if methods == METHOD_BOTH:
                methods = METHOD_FITTER_DEFAULTS
            else:
                methods = [methods]
        if isinstance(methods, list):
            if isinstance(methods[0], str):
                results = [ModelFitterCore.OptimizerMethod(method=m, kwargs={})
                      for m in methods]
            else:
                results = methods
        else:
            raise RuntimeError("Must be a list")
        trues = [isinstance(m, ModelFitterCore.OptimizerMethod) for m in results]
        if not all(trues):
            raise ValueError("Invalid methods: %s" % str(methods))
        return results


    @classmethod
    def mkParameters(cls, parameterDct:dict=None,
          parametersToFit:list=None,
          logger:Logger=Logger(),
          lowerBound:float=PARAMETER_LOWER_BOUND,
          upperBound:float=PARAMETER_UPPER_BOUND)->lmfit.Parameters:
        """
        Constructs lmfit parameters based on specifications.

        Parameters
        ----------
        parameterDct: key=name, value=ParameterSpecification
        parametersToFit: list of parameters to fit
        logger: error logger
        lowerBound: lower value of range for parameters
        upperBound: upper value of range for parameters

        Returns
        -------
        lmfit.Parameters
        """
        def get(value, base_value, multiplier):
            if value is not None:
                return value
            return base_value*multiplier
        #
        if (parametersToFit is None) and (parameterDct is None):
            raise RuntimeError("Must specify one of these parameters.")
        if parameterDct is None:
            parameterDct = {}
        if parametersToFit is None:
            parametersToFit = parameterDct.keys()
        if logger is None:
            logger = logger()
        params = lmfit.Parameters()
        for parameterName in parametersToFit:
            if parameterName in parameterDct.keys():
                specification = parameterDct[parameterName]
                value = get(specification.value, specification.value, 1.0)
                if value > 0:
                    lower_factor = LOWER_PARAMETER_MULT
                    upper_factor = UPPER_PARAMETER_MULT
                else:
                    upper_factor = UPPER_PARAMETER_MULT
                    lower_factor = LOWER_PARAMETER_MULT
                lower = get(specification.lower, specification.value,
                      lower_factor)
                upper = get(specification.upper, specification.value,
                      upper_factor)
                if np.isclose(lower - upper, 0):
                    upper = 0.0001
                try:
                    params.add(parameterName, value=value, min=lower, max=upper)
                except Exception as err:
                    msg = "modelFitterCore/mkParameters parameterName %s" \
                          % parameterName
                    logger.error(msg, err)
            else:
                value = np.mean([lowerBound, upperBound])
                params.add(parameterName, value=value,
                      min=lowerBound, max=upperBound)
        return params

    @classmethod
    def initializeRoadrunnerModel(cls, modelSpecification):
        """
        Sets self.roadrunnerModel.

        Parameters
        ----------
        modelSpecification: ExtendedRoadRunner/str

        Returns
        -------
        ExtendedRoadRunner
        """
        if isinstance(modelSpecification,
              te.roadrunner.extended_roadrunner.ExtendedRoadRunner):
            roadrunnerModel = modelSpecification
        elif isinstance(modelSpecification, str):
            roadrunnerModel = te.loada(modelSpecification)
        else:
            msg = 'Invalid model.'
            msg = msg + "\nA model must either be a Roadrunner model "
            msg = msg + "an Antimony model."
            raise ValueError(msg)
        return roadrunnerModel

    @classmethod
    def setupModel(cls, roadrunner, parameters, logger=Logger()):
        """
        Sets up the model for use based on the parameter parameters

        Parameters
        ----------
        roadrunner: ExtendedRoadRunner
        parameters: lmfit.Parameters
        logger Logger
        """
        pp = parameters.valuesdict()
        for parameter in pp.keys():
            try:
                roadrunner.model[parameter] = pp[parameter]
            except Exception as err:
                msg = "_modelFitterCore.setupModel: Could not set value for %s"  \
                      % parameter
                logger.error(msg, err)

    @classmethod
    def runSimulation(cls, parameters=None,
          roadrunner=None,
          startTime=0,
          endTime=5,
          numPoint=30,
          selectedColumns=None,
          returnDataFrame=True,
          _logger=Logger(),
          _loggerPrefix="",
          ):
        """
        Runs a simulation. Defaults to parameter values in the simulation.

        Parameters
       ----------
        roadrunner: ExtendedRoadRunner/str
            Roadrunner model
        parameters: lmfit.Parameters
            lmfit parameters
        startTime: float
            start time for the simulation
        endTime: float
            end time for the simulation
        numPoint: int
            number of points in the simulation
        selectedColumns: list-str
            output columns in simulation
        returnDataFrame: bool
            return a DataFrame
        _logger: Logger
        _loggerPrefix: str


        Return
        ------
        NamedTimeseries (or None if fail to converge)
        """
        if isinstance(roadrunner, str):
            roadrunner = cls.initializeRoadrunnerModel(roadrunner)
        else:
            roadrunner.reset()
        if parameters is not None:
            # Parameters have been specified
            cls.setupModel(roadrunner, parameters, logger=_logger)
        # Do the simulation
        if selectedColumns is not None:
            newSelectedColumns = list(selectedColumns)
            if TIME not in newSelectedColumns:
                newSelectedColumns.insert(0, TIME)
            try:
                data = roadrunner.simulate(startTime, endTime, numPoint,
                      newSelectedColumns)
            except Exception as err:
                _logger.error("Roadrunner exception: ", err)
                data = None
        else:
            try:
                data = roadrunner.simulate(startTime, endTime, numPoint)
            except Exception as err:
                _logger.exception("Roadrunner exception: %s", err)
                data = None
        if data is None:
            return data
        fittedTS = NamedTimeseries(namedArray=data)
        if returnDataFrame:
            result = fittedTS.to_dataframe()
        else:
            result = fittedTS
        return result

    @classmethod
    def rpConstruct(cls):
        """
        Overrides rpickler.rpConstruct to create a method that
        constructs an instance without arguments.

        Returns
        -------
        Instance of cls
        """
        return cls(None, None, None)

    def rpRevise(self):
        """
        Overrides rpickler.
        """
        if "logger" not in self.__dict__.keys():
            self.logger = Logger()

    def _validateFittedDataTransformDct(self):
        if self.fittedDataTransformDct is not None:
            keySet = set(self.fittedDataTransformDct.keys())
            selectedColumnsSet = self.selectedColumns
            if (keySet is not None) and (selectedColumnsSet is not None):
                excess = set(keySet).difference(selectedColumnsSet)
                if len(excess) > 0:
                    msg = "Columns not in selectedColumns: %s"  % str(excess)
                    raise ValueError(msg)

    def _transformFittedTS(self, data):
        """
        Updates the fittedTS taking into account required transformations.

        Parameters
        ----------
        data: np.ndarray

        Results
        ----------
        NamedTimeseries
        """
        colnames = list(self.selectedColumns)
        colnames.insert(0, TIME)
        fittedTS = NamedTimeseries(array=data[:, :], colnames=colnames)
        if self.fittedDataTransformDct is not None:
            for column, func in self.fittedDataTransformDct.items():
                if func is not None:
                    fittedTS[column] = func(fittedTS)
        return fittedTS

    @staticmethod
    def _updateParameterDct(parameterDct):
        """
        Handles values that are tuples instead of ParameterSpecification.
        """
        if parameterDct is None:
            parameterDct = {}
        dct = dict(parameterDct)
        for name, value in parameterDct.items():
            if isinstance(value, tuple):
                dct[name] = ParameterSpecification(lower=value[0],
                      upper=value[1], value=value[2])
        return dct

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

    def _adjustNames(self, antimonyModel:str, observedTS:NamedTimeseries)  \
          ->typing.Tuple[NamedTimeseries, list]:
        """
        Antimony exports can change the names of floating species
        by adding a "_" at the end. Check for this and adjust
        the names in observedTS.

        Return
        ------
        NamedTimeseries: newObservedTS
        list: newSelectedColumns
        """
        rr = te.loada(antimonyModel)
        dataNames = rr.simulate().colnames
        names = ["[%s]" % n for n in observedTS.colnames]
        missingNames = [n[1:-1] for n in set(names).difference(dataNames)]
        newSelectedColumns = list(self.selectedColumns)
        if len(missingNames) > 0:
            newObservedTS = observedTS.copy()
            self.logger.exception("Missing names in antimony export: %s"
                  % str(missingNames))
            for name in observedTS.colnames:
                missingName = "%s_" % name
                if name in missingNames:
                    newObservedTS = newObservedTS.rename(name, missingName)
                    newSelectedColumns.remove(name)
                    newSelectedColumns.append(missingName)
        else:
            newObservedTS = observedTS
        return newObservedTS, newSelectedColumns

    def copy(self, isKeepLogger=False):
        """
        Creates a copy of the model fitter.
        Preserves the user-specified settings and the results
        of bootstrapping.
        """
        if not isinstance(self.modelSpecification, str):
            try:
                modelSpecification = self.modelSpecification.getAntimony()
            except Exception as err:
                self.logger.error("Problem wth conversion to Antimony. Details:",
                      err)
                raise ValueError("Cannot proceed.")
            observedTS, selectedColumns = self._adjustNames(
                  modelSpecification, self.observedTS)
        else:
            modelSpecification = self.modelSpecification
            observedTS = self.observedTS.copy()
            selectedColumns = self.selectedColumns
        #
        if isKeepLogger:
            logger = self.logger
        elif self.logger is not None:
            logger = self.logger.copy()
        else:
            logger = None
        newModelFitter = self.__class__(
              copy.deepcopy(modelSpecification),
              observedTS,
              copy.deepcopy(self.parametersToFit),
              selectedColumns=selectedColumns,
              fitterMethods=self._fitterMethods,
              bootstrapMethods=self._bootstrapMethods,
              parameterLowerBound=self.lowerBound,
              parameterUpperBound=self.upperBound,
              parameterDct=copy.deepcopy(self.parameterDct),
              fittedDataTransformDct=copy.deepcopy(self.fittedDataTransformDct),
              logger=logger,
              isPlot=self._isPlot)
        if self.bootstrapResult is not None:
            newModelFitter.bootstrapResult = self.bootstrapResult.copy()
            newModelFitter.params = newModelFitter.bootstrapResult.params
        else:
            newModelFitter.bootstrapResult = None
            newModelFitter.params = self.params
        return newModelFitter

    def initializeRoadRunnerModel(self):
        """
        Sets self.roadrunnerModel.
        """
        self.roadrunnerModel = ModelFitterCore.initializeRoadrunnerModel(
              self.modelSpecification)

    def getDefaultParameterValues(self):
        """
        Obtain the original values of parameters.

        Returns
        -------
        dict:
            key: parameter name
            value: value of parameter
        """
        dct = {}
        self.initializeRoadRunnerModel()
        self.roadrunnerModel.reset()
        for parameterName in self.parametersToFit:
            dct[parameterName] = self.roadrunnerModel.model[parameterName]
        return dct

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
        NamedTimeseries
        """
        def setValue(default, parameter):
            # Sets to default if parameter unspecified
            if parameter is None:
                return default
            return parameter
        #
        startTime = setValue(self.observedTS.start, startTime)
        endTime = setValue(self.observedTS.end, endTime)
        numPoint = setValue(len(self.observedTS), numPoint)
        #
        if self.roadrunnerModel is None:
            self.initializeRoadRunnerModel()
        #
        return ModelFitterCore.runSimulation(parameters=params,
              roadrunner=self.roadrunnerModel,
              startTime=startTime,
              endTime=endTime,
              numPoint=numPoint,
              selectedColumns=self.selectedColumns,
              _logger=self.logger,
              _loggerPrefix=self._loggerPrefix,
              returnDataFrame=False)

    def updateFittedAndResiduals(self, **kwargs)->np.ndarray:
        """
        Updates values of self.fittedTS and self.residualsTS
        based on self.params.

        Parameters
        ----------
        kwargs: dict
            arguments for simulation

        Instance Variables Updated
        --------------------------
        self.fittedTS
        self.residualsTS

        Returns
        -------
        1-d ndarray of residuals
        """
        self.fittedTS = self.simulate(**kwargs)  # Updates self.fittedTS
        residualsArr = self._residuals(self.params)
        numRow = len(self.fittedTS)
        numCol = len(residualsArr)//numRow
        residualsArr = np.reshape(residualsArr, (numRow, numCol))
        cols = self.selectedColumns
        if self.residualsTS is None:
            self.residualsTS = self.observedTS.subsetColumns(cols)
        self.residualsTS[cols] = residualsArr

    def _residuals(self, params)->np.ndarray:
        """
        Compute the residuals between objective and experimental data
        Handle nan values in observedTS. This internal-only method
        is implemented to maximize efficieency.

        Parameters
        ----------
        kwargs: dict
            arguments for simulation

        Returns
        -------
        1-d ndarray of residuals
        """
        data = ModelFitterCore.runSimulation(parameters=params,
              roadrunner=self.roadrunnerModel,
              startTime=self.observedTS.start,
              endTime=self.observedTS.end,
              numPoint=len(self.observedTS),
              selectedColumns=self.selectedColumns,
              _logger=self.logger,
              _loggerPrefix=self._loggerPrefix,
              returnDataFrame=False)
        if data is None:
            residualsArr = np.repeat(LARGE_RESIDUAL, len(self._observedArr))
        else:
            residualsArr = self._observedArr - data.flatten()
            residualsArr = np.nan_to_num(residualsArr)
        rssq = sum(residualsArr**2)
        if (self._bestParameters.rssq is None)  \
              or (rssq < self._bestParameters.rssq):
            self._bestParameters = _BestParameters(
                  params=params.copy(), rssq=rssq)
        return residualsArr

    def fitModel(self, params:lmfit.Parameters=None, max_nfev=100):
        """
        Fits the model by adjusting values of parameters based on
        differences between simulated and provided values of
        floating species.

        Parameters
        ----------
        params: starting values of parameters

        Example
        -------
        f.fitModel()
        """
        ParameterDescriptor = collections.namedtuple("ParameterDescriptor",
              "params method rssq kwargs minimizer minimizerResult")
        MAX_NFEV = "max_nfev"
        block = Logger.join(self._loggerPrefix, "fitModel")
        guid = self.logger.startBlock(block)
        self.initializeRoadRunnerModel()
        self.params = None
        if self.parametersToFit is not None:
            if params is None:
                params = self.mkParams()
            # Fit the model to the data using one or more methods.
            # Choose the result with the lowest residual standard deviation
            paramResults = []
            lastExcp = None
            for idx, optimizerMethod in enumerate(self._fitterMethods):
                method = optimizerMethod.method
                kwargs = optimizerMethod.kwargs
                if MAX_NFEV not in kwargs:
                    kwargs[MAX_NFEV] = max_nfev
                for _ in range(self._numFitRepeat):
                    self._bestParameters = _BestParameters(params=None, rssq=None)
                    minimizer = lmfit.Minimizer(self._residuals, params)
                    try:
                        minimizerResult = minimizer.minimize(
                              method=method, **kwargs)
                    except Exception as excp:
                        lastExcp = excp
                        msg = "Error minimizing for method: %s" % method
                        self.logger.error(msg, excp)
                        continue
                    params = self._bestParameters.params.copy()
                    rssq = np.sum(self._residuals(params)**2)
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
                msg = "*** Minimizer failed for this model and data."
                self.logger.error(msg, lastExcp)
            else:
                # Select the result that has the smallest residuals
                sortedMethods = sorted(paramResults, key=lambda r: r.rssq)
                bestMethod = sortedMethods[0]
                self.params = bestMethod.params
                self.minimizer= bestMethod.minimizer
                self.minimizerResult = bestMethod.minimizerResult
        # Ensure that residualsTS and fittedTS match the parameters
        self.updateFittedAndResiduals(params=self.params)
        self.logger.endBlock(guid)

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

    def _setupModel(self, parameters):
        """
        Sets up the model for use based on the parameter parameters

        Parameters
        ----------
        parameters: lmfit.Parameters

        """
        ModelFitterCore.setupModel(self.roadrunnerModel, parameters, logger=self.logger)

    def mkParams(self, parameterDct:dict=None)->lmfit.Parameters:
        """
        Constructs lmfit parameters based on specifications.

        Parameters
        ----------
        parameterDct: key=name, value=ParameterSpecification

        Returns
        -------
        lmfit.Parameters
        """
        if parameterDct is None:
            parameterDct = self.parameterDct
        return ModelFitterCore.mkParameters(parameterDct,
              parametersToFit=self.parametersToFit,
              logger=self.logger,
              lowerBound=self.lowerBound,
              upperBound=self.upperBound)

    def _checkFit(self):
        if self.params is None:
            raise ValueError("Must use fitModel before using this method.")

    def serialize(self, path):
        """
        Serialize the model to a path.

        Parameters
        ----------
        path: str
            File path
        """
        newModelFitter = self.copy()
        with open(path, "wb") as fd:
            rpickle.dump(newModelFitter, fd)

    @classmethod
    def deserialize(cls, path):
        """
        Deserialize the model from a path.

        Parameters
        ----------
        path: str
            File path

        Return
        ------
        ModelFitter
            Model is initialized.
        """
        with open(path, "rb") as fd:
            fitter = rpickle.load(fd)
        fitter.initializeRoadRunnerModel()
        return fitter
