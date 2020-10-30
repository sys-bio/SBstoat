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

from SBstoat.namedTimeseries import NamedTimeseries, TIME, mkNamedTimeseries
import SBstoat._plotOptions as po
from SBstoat.modelFitter import ModelFitter

from docstring_expander.expander import Expander
import inspect
import lmfit
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import shutil
import typing
import warnings


DIR_NAME = "ModelStudyFitters"
MIN_COUNT_BOOTSTRAP = 10


class ModelStudy(object):

    def __init__(self, modelSpecification, dataSources, parametersToFit,
          dirPath=None, instanceNames=None, isSerialized=True,
          isPlot=True,  **kwargs):
        """
        Parameters
        ---------
        modelSpecification: ExtendedRoadRunner/str
            roadrunner model or antimony model
        dataSources: list-NamedTimeseries/list-str
            str: path to CSV file
        parametersToFit: list-str/None
            parameters in the model that you want to fit
            if None, no parameters are fit
        dirPath: str
            Path to the output directory containing the serialized fitters
            for the study.
        instanceNames: list-str
            Names of study instances corresponds to the list of dataSources
        isSerialized: bool
            Use the serialization of each ModelFitter, if it exists
        isPlot: bool
            Do plots
        kwargs: dict
            arguments passed to ModelFitter
        """
        self.dirPath = dirPath  # Path to the directory serialized ModelFitter
        if self.dirPath is None:
            length = len(inspect.stack())
            absPath = os.path.abspath((inspect.stack()[length-1])[1])
            dirCaller = os.path.dirname(absPath)
            self.dirPath = os.path.join(dirCaller, DIR_NAME)
        self.isPlot = isPlot
        if not os.path.isdir(self.dirPath):
            os.mkdir(self.dirPath)
        if instanceNames is None:
            self.instanceNames = ["src_%d" %d
                  for d in range(1, len(dataSources)+1)]
        else:
            self.instanceNames = [str(i) for i in instanceNames]
        self.fitterPathDct = {}  # Path to serialized fitters
        self.fitterDct = {}  # Fitters
        for idx, name in enumerate(self.instanceNames):
            dataSource = dataSources[idx]
            filePath = self._getSerializePath(name)
            self.fitterPathDct[name] = filePath
            if os.path.isfile(filePath) and isSerialized:
                self.fitterDct[name] = ModelFitter.deserialize(filePath)
            else:
                self.fitterDct[name] = ModelFitter(modelSpecification, dataSource,
                       parametersToFit, isPlot=self.isPlot, **kwargs)
                self.fitterDct[name].serialize(filePath)

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
        return os.path.join(self.dirPath, "%s.pcl" % name)

    def _serializeFitter(self, name):
        filePath = self._getSerializePath(name)
        self.fitterDct[name].serialize(filePath)

    def fitModel(self):
        """
        Does fits for all models and serializes the results.
        """
        for name in self.instanceNames:
            print("\n***Fit for data %s" % name)
            fitter = self.fitterDct[name]
            fitter.fitModel()
            self._serializeFitter(name)
            print(fitter.reportFit())

    def _isBootstrapResult(self, fitter):
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
            print("Bootstrapping for instance %s" % name)
            fitter = self.fitterDct[name]
            if not self._isBootstrapResult(fitter):
                if fitter.params is None:
                    fitter.fitModel()
                fitter.bootstrap(**kwargs)
                if not self._isBootstrapResult(fitter):
                    fitter.bootstrapResult = None
                self._serializeFitter(name)

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
            print("Plots for instance %s" % name)
            fitter = self.fitterDct[name]
            if fitter.params is None:
                print("***Must do fitModel or bootstrap before plotting.")
            else:
                if fitter.bootstrapResult is not None:
                    if fitter.bootstrapResult.numSimulation > 0:
                        fitter.plotFitAll(**newKwargs)

    @Expander(po.KWARGS, po.BASE_OPTIONS, indent=8, header=po.HEADER)
    def plotParameterEstimates(self, **kwargs):
        """
        Parameters
        ----------
        #@expand
        """
        SCALE = 1.1  # Amount by which to scale an upper boundary
        fitterDct = {}
        for dataSource, fitter in self.fitterDct.items():
            for parameterName, values in  \
                  fitter.bootstrapResult.parameterDct.items():
                length = len(values)
                if length < MIN_COUNT_BOOTSTRAP:
                    print("***Warning. Only %d samples from bootstrap of %s." %
                          (length, dataSource))
                    print("            Unable to do parameer plot.")
                else:
                    fitterDct[dataSource] = fitter
        if len(fitterDct) == 0:
            print("***Nothing to plot.")
        else:
            instanceNames = fitterDct.keys()
            trues = [f.bootstrapResult is None for f in fitterDct.values()]
            if any(trues):
                raise ValueError("\n***Must do bootstrap before getting report.")
            fitter = [v for v in fitterDct.values()][0]
            parameters = fitter.parametersToFit
            parameterDct = fitter.getDefaultParameterValues()
            # Construct plot
            fig, axes = plt.subplots(len(parameters),1, figsize=(12,10))
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
            _ = plt.suptitle("Bootstrap Parameters With 1-Standard")
            if self.isPlot:
                plt.show()
