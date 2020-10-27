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
    modelStudy.report()  # Generates a standard report of the results
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME, mkNamedTimeseries
import SBstoat._plotOptions as po
from SBstoat.modelFitter import ModelFitter

import lmfit
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import shutil
import typing


FILE_NAME = os.path.splitext(__file__)[0]  # Extract the file name
DIR_PATH = "%sFiters" % FILE_NAME


class ModelStudy(object):

    def __init__(self, modelSpecification, dataSources, parametersToFit,
          dirPath=DIR_PATH, instanceNames=None, isSerialized=True,
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
        self.isPlot = isPlot
        if not os.path.isdir(dirPath):
            os.mkdir(dirPath)
        if instanceNames is None:
            self.instanceNames = ["src_%d" %d
                  for d in range(1, len(dataSources)+1)]
        else:
            self.instanceNames = [str(i) for i in instanceNames]
        self.fitterPathDct = {}  # Path to serialized fitters
        self.fitterDct = {}  # Fitters
        for idx, name in enumerate(self.instanceNames):
            source = dataSources[idx]
            filePath = self._getSerializePath(name)
            self.fitterPathDct[name] = filePath
            if os.path.isfile(filePath) and isSerialized:
                self.fitterDct[name] = ModelFitter.deserialize(filePath)
            else:
                self.fitterDct[name] = ModelFitter(modelSpecification, source,
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

    def bootstrap(self, **kwargs):
        """
        Does bootstrap the models and serializes the results.
        
        Parameters
        ----------
        kwargs: dict
            arguments passed to ModelFitter.bootstrap
        """
        for name in self.instanceNames:
            print("Bootstrapping for instance %s" % name)
            fitter = self.fitterDct[name]
            if fitter.params is None:
                fitter.fitModel()
            fitter.bootstrap(**kwargs)
            self._serializeFitter(name)

    def plotFitAll(self, **kwargs):
        """
        Constructs plots using parameters from bootstrap if available.

        Parameters
        ----------
        kwargs: dict
            arguments passed to ModelFitter plots
        """
        for name in self.instanceNames:
            print("Plots for instance %s" % name)
            fitter = self.fitterDct[name]
            if fitter.params is None:
                print("***Must do fitModel or bootstrap before plotting.")
            else:
                fitter.plotFitAll()

    def plotParameterEstimates(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            arguments passed to ModelFitter plots
        """
        trues = [f.bootstrapResult is None for f in self.fitterDct.values()]
        if any(trues):
            raise ValueError("\n***Must do bootstrap before getting report.")
        fitter = [v for v in self.fitterDct.values()][0]
        parameters = fitter.parametersToFit
        # Construct plot
        fig, axes = plt.subplots(len(parameters),1, figsize=(12,10))
        for pos, param in enumerate(parameters):
            ax = axes[pos]
            means = [f.bootstrapResult.parameterMeanDct[param]
                  for f in self.fitterDct.values()]
            stds = [f.bootstrapResult.parameterStdDct[param]
                  for f in self.fitterDct.values()]
            y_upper = max([s + m for s, m in zip(stds, means)])*1.1
            ax.errorbar(self.instanceNames, means, yerr=stds, linestyle="")
            ax.scatter(self.instanceNames, means, s=18.0)
            # FIXME: need to get the intial values of parameters
            if False:
                values = np.repeat(parameterDct[param].value, len(self.instanceNames))
                ax.plot(self.instanceNames, values, linestyle="dashed", color="grey")
            ax.set_title(param)
            ax.set_ylim([0, y_upper])
            if pos == len(parameters) - 1:
                ax.set_xticklabels(self.instanceNames)
                ax.set_xlabel("Observed Data")
            else:
                ax.set_xticklabels([])
                ax.set_xlabel("")
        _ = plt.suptitle("Bootstrap Parameters With 1-Standard")
        if self.isPlot:
            plt.show()
