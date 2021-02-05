# -*- coding: utf-8 -*-
"""
 Created on Tue Jul  7 14:24:09 2020

@author: joseph-hellerstein

A model study is an application of a model to multiple sets of
observed data (fitting the same model parameters). We refer to each
collection of observed data as a data source.
ModelStudy provides a way to create one ModelFitter for each
data source, and provides access to these fitters.


data_sets should be a list of data inputs as required by ModelFitter.

Usage
-----
    modelStudy = ModelStudy(model, dataSources, parameters)
    modelStudy.bootstrap()  # Create and save the bootstrap results
    modelStudy.plotFitAll()  # Plot of observed and fitted values for the study
    modelstudy.plotParameterEstimates()
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME
import SBstoat._plotOptions as po
from SBstoat.logs import Logger
from SBstoat.modelFitter import ModelFitter

from docstring_expander.expander import Expander
import inspect
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import warnings


DIR_NAME = "ModelStudyFitters"
MIN_COUNT_BOOTSTRAP = 10


############## FUNCTIONS ###################
def mkDataSourceDct(filePath, colName, dataSourceNames=None, isTimeColumns=True):
    """
    Creates a dataSource dictionary as required by ModelStudy from
    a file whose columns are observed values of the same variable.

    Parameters
    ----------
    filepath: str
        path to the file containing columns of observed values
    colName: str
        Name of the simulation variable to fit to observed values
    dataSourceNames: list-str
        Names for the instances of observedValues
        if None, then column headers are used (or column number)
    isTimeColumns: boolean
        Columns are time. If False, rows are time.

    Returns
    -------
    dict: key is instance name; value is NamedTimeseries
    """
    dataDF = pd.read_csv(filePath, header=None)
    if isTimeColumns:
        dataDF = dataDF.transpose()
    if dataSourceNames is not None:
        # Use instance names
        if len(dataSourceNames) != len(dataDF.columns):
            msg = "Number of instances is not equal to the number of columns"
            raise ValueError(msg)
        dct = {k: v for k, v in zip(dataDF.columns, dataSourceNames)}
        dataDF = dataDF.rename(columns=dct)
    # Ensure that column names are strings
    dct = {k: str(k) for k in dataDF.columns}
    dataDF = dataDF.rename(columns=dct)
    dataDF.index.name = "time"
    # Construct the data source dictionary
    dataDF.index = range(len(dataDF))
    dataDF.index.name = TIME
    dataTS = NamedTimeseries(dataframe=dataDF)
    dataSourceDct = {d: dataTS.subsetColumns([d]) for d in dataDF.columns}
    dataSourceDct = {d: dataSourceDct[d].rename(d, colName)
          for d in dataSourceDct.keys()}
    #
    return dataSourceDct


############## CLASSES ###################
class ModelStudy():

    def __init__(self, modelSpecification, dataSources,
          dirStudyPath=None,
          instanceNames=None,
          logger=Logger(),
          useSerialized=True, doSerialize=True,
          isPlot=True,  **kwargs):
        """
        Parameters
        ---------
        modelSpecification: ExtendedRoadRunner/str
            roadrunner model or antimony model
        dataSources: list-NamedTimeseries/list-str or dict of either
            str: path to CSV file
        dirStudyPath: str
            Path to the output directory containing the serialized fitters
            for the study.
        instanceNames: list-str
            Names of study instances corresponds to the list of dataSources
        useSerialized: bool
            Use the serialization of each ModelFitter, if it exists
        doSerialized: bool
            Serialize each ModelFitter
        isPlot: bool
            Do plots
        kwargs: dict
            arguments passed to ModelFitter
        """
        self.dirStudyPath = dirStudyPath  # Path to the directory serialized ModelFitter
        if self.dirStudyPath is None:
            length = len(inspect.stack())
            absPath = os.path.abspath((inspect.stack()[length-1])[1])
            dirCaller = os.path.dirname(absPath)
            self.dirStudyPath = os.path.join(dirCaller, DIR_NAME)
        self.isPlot = isPlot
        self.doSerialize = doSerialize
        self.useSerialized = useSerialized
        self.instanceNames, self.dataSourceDct = self._mkInstanceData(
              instanceNames, dataSources)
        self.fitterPathDct = {}  # Path to serialized fitters; key is instanceName
        self.fitterDct = {}  # Fitters: key is instanceName
        self.logger = logger
        # Ensure that the directory exists
        if not os.path.isdir(self.dirStudyPath):
            os.makedirs(self.dirStudyPath)
        # Construct the fitters
        for name, dataSource in self.dataSourceDct.items():
            filePath = self._getSerializePath(name)
            self.fitterPathDct[name] = filePath
            if os.path.isfile(filePath) and useSerialized:
                self.fitterDct[name] = ModelFitter.deserialize(filePath)
            else:
                self.fitterDct[name] = ModelFitter(modelSpecification,
                       dataSource, logger=self.logger,
                       isPlot=self.isPlot, **kwargs)
                self._serializeFitter(name)

    def _mkInstanceData(self, instanceNames, dataSources):
        # Determine value for instances of observed data
        if isinstance(instanceNames, list):
            newInstanceNames = list(instanceNames)
        elif instanceNames is None:
            newInstanceNames = ["src_%d" %d
                  for d in range(1, len(dataSources)+1)]
        else:
            msg = "Invalid type for instanceNames: %s" % str(type(instanceNames))
            msg += "  \nMust be list or None."
            raise ValueError(msg)
        # Value of newDataSourceDct
        if isinstance(dataSources, dict):
            newDataSourceDct = dict(dataSources)
            newInstanceNames = list(newDataSourceDct.keys())
        elif isinstance(dataSources, list):
            newDataSourceDct = {n: dataSources[i]
                  for i, n in enumerate(newInstanceNames)}
        else:
            msg = "Invalid type for dataSources: %s" % str(type(dataSources))
            msg += "  \nMust be dict or list."
            raise ValueError(msg)
        #
        return newInstanceNames, newDataSourceDct

    def _getSerializePath(self, name):
        """
        Constructs to serialization file for fitter.

        Parameters
        ----------
        name: str
            Name of the fitter instance/data source

        Returns
        -------
        str
        """
        return os.path.join(self.dirStudyPath, "%s.pcl" % name)

    def _serializeFitter(self, name):
        filePath = self._getSerializePath(name)
        if self.doSerialize:
            self.fitterDct[name].serialize(filePath)

    def fitModel(self):
        """
        Does fits for all models and serializes the results.
        """
        for name in self.instanceNames:
            self.logger.activity("Fit for data %s" % name)
            fitter = self.fitterDct[name]
            fitter.fitModel()
            self._serializeFitter(name)
            self.logger.result(fitter.reportFit())

    def _hasBootstrapResult(self, fitter):
        result = False
        if fitter.bootstrapResult is not None:
            if fitter.bootstrapResult.numSimulation > 0:
                result = True
        return result

    def bootstrap(self, **kwargs):
        """
        Does bootstrap for models that have not bootstrap. Serializes the result.

        Parameters
        ----------
        kwargs: dict
            arguments passed to ModelFitter.bootstrap
        """
        for name in self.instanceNames:
            fitter = self.fitterDct[name]
            if (not self.useSerialized) and (not self._hasBootstrapResult(fitter)):
                msg = "Doing bootstrapp for instance %s" % name
                self.logger.activity(msg)
                if fitter.params is None:
                    fitter.fitModel()
                fitter.bootstrap(**kwargs)
                if not self._hasBootstrapResult(fitter):
                    # Not a valid bootstrap result
                    fitter.bootstrapResult = None
                self._serializeFitter(name)
            else:
                if self._hasBootstrapResult(fitter):
                    msg = "Using existing bootstrap results (%d) for %s"  \
                          % (fitter.bootstrapResult.numSimulation, name)
                    self.logger.result(msg)
                else:
                    msg = "No bootstrap results for data source %s"  % name
                    self.logger.result(msg)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8, header=po.HEADER)
    def plotFitAll(self, **kwargs):
        """
        Constructs plots using parameters from bootstrap if available.

        Parameters
        ----------
        #@expand
        """
        for name in self.instanceNames:
            newKwargs = dict(kwargs)
            if po.SUPTITLE in newKwargs.keys():
                title = "%s: %s" % (name, newKwargs[po.SUPTITLE])
            else:
                title = "%s: Fitted vs. Observed (with 95th percentile)"  \
                      % name
            newKwargs[po.SUPTITLE] = title
            fitter = self.fitterDct[name]
            if fitter.params is None:
                self.logger.result("Must do fitModel or bootstrap before plotting.")
            else:
                if fitter.bootstrapResult is not None:
                    if fitter.bootstrapResult.numSimulation > 0:
                        fitter.plotFitAll(**newKwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8, header=po.HEADER)
    def plotParameterEstimates(self):
        """
        Parameters
        ----------
        #@expand
        """
        SCALE = 1.1  # Amount by which to scale an upper boundary
        fitterDct = {}
        for dataSource, fitter in self.fitterDct.items():
            if self._hasBootstrapResult(fitter):
                for parameterName, values in  \
                      fitter.bootstrapResult.parameterDct.items():
                    length = len(values)
                    if length < MIN_COUNT_BOOTSTRAP:
                        msg = "Only %d samples from bootstrap of %s."  \
                              % (length, dataSource)
                        msg += "Unable to do plot for parameter %s."  \
                               % parameterName
                        self.logger.result(msg)
                    else:
                        fitterDct[dataSource] = fitter
        if len(fitterDct) == 0:
            self.logger.result("No data to plot.")
        else:
            instanceNames = fitterDct.keys()
            trues = [f.bootstrapResult is None for f in fitterDct.values()]
            if any(trues):
                raise ValueError("\n***Must do bootstrap before getting report.")
            fitter = list(fitterDct.values())[0]
            parameters = fitter.parametersToFit
            parameterDct = fitter.getDefaultParameterValues()
            # Construct plot
            _, axes = plt.subplots(len(parameters),1, figsize=(12,10))
            for pos, pName in enumerate(parameters):
                ax = axes[pos]
                means = [f.bootstrapResult.parameterMeanDct[pName]
                      for f in fitterDct.values()]
                stds = [f.bootstrapResult.parameterStdDct[pName]
                      for f in fitterDct.values()]
                y_upper = max([s + m for s, m in zip(stds, means)])*SCALE
                y_upper = max(y_upper, parameterDct[pName]*SCALE)
                ax.errorbar(instanceNames, means, yerr=stds, linestyle="")
                ax.scatter(instanceNames, means, s=18.0)
                values = np.repeat(parameterDct[pName], len(instanceNames))
                ax.plot(instanceNames, values, linestyle="dashed", color="grey")
                ax.set_title(pName)
                ax.set_ylim([0, y_upper])
                if pos == len(parameters) - 1:
                # Ignore bogus warning from matplotlib
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ax.set_xticklabels(instanceNames)
                    ax.set_xlabel("Observed Data")
                else:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
            suptitle = "Bootstrap Parameters With 1-Standard"
            suptitle += "\n(Dashed line is parameter value in model.)"
            _ = plt.suptitle(suptitle)
            if self.isPlot:
                plt.show()
