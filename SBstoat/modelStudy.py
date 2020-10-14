# -*- coding: utf-8 -*-
"""
 Created on Tue Jul  7 14:24:09 2020

@author: joseph-hellerstein

A model study is an application of a model to multiple sets of
observed data with fitting the same model parameters.


data_sets should be a list of data inputs as required by ModelFitter.

Usage
-----
    modelStudy = ModelStudy(model, data_sets, parameters)
    modelStudy.bootstrap()  # Create and save the bootstrap results
    modelStudy.report()  # Generates a standard report of the results
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME, mkNamedTimeseries
import SBstoat._plotOptions as po
from SBstoat.modelFitter import ModelFitter

import lmfit
import numpy as np
import os
import pandas as pd
import shutil
import typing


DIR_PATH = "study_results"


class ModelStudy(object):

    def __init__(self, modelSpecification, dataSources, parametersToFit,
          dirPath=DIR_PATH, instanceNames=None, **kwargs):
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
            Names of study instances
        kwargs: dict
            arguments passed to ModelFitter
        """
        self.dirPath = dirPath  # Path to the directory for serialization
        if not os.path.isdir(dirPath):
            os.mkdir(dirPath)
        if instanceNames is None:
            self.instanceNames = ["study_%d" %d for d in range(1, len(dataSources)+1)]
        else:
            self.instanceNames = instanceNames
        self.fitterPathDct = {}  # Path to serialized fitters
        self.fitters = {}  # Fitters
        for idx, name in enumerate(self.instanceNames):
            source = dataSources[idx]
            ffile = "%s.pcl" % os.path.split(source)[0]
            filePath = os.path.join(dirPath, ffile)
            self.fitterPathDct[name] = filePath
            if os.path.isfile(filePath):
                self.fitters[name] = modelFitter.deserialize(filePath)
            else:
                self.fitters[name] = ModelFitter(modelSpecification, source,
                       parametersToFit, **kwargs)

    def fitModel(self):
        """
        Does fits for the models and serializes the results.
        """
        for name in self.instanceNames:
            print("\n***Fit for data %s" % name)
            self.fitters[source].fitModel()
            filePath = self.fitterPathDct[name]
            self.fitters[name].serialize(filePath)
            self.fitters[name].reportFit()

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
            fitter = self.fitters[name]
            fitter.fitModel()
            fitter.bootstrapResult(**kwargs)
            fitter.serialize(filePath)

    def plotFitAll(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            arguments passed to ModelFitter plots
        """
        for name in self.instanceNames:
            print("Plots for instance %s" % name)
            fitter = self.fitters[name]
            fitter.plotFitAll()

    def plotParametersEstimates(self):
        """
        Parameters
        ----------
        kwargs: dict
            arguments passed to ModelFitter plots
        """
        trues = [f.bootstrapResult is None for f in self.fitters.values()]
        if not all(trues):
            raise ValueError("\n***Must do bootstrap before getting report.")
        fitter = self.fitters.values()[0]
        parameters = fitter.parametersToFit
        fig, axes = plt.subplots(len(parameters),1, figsize=(12,10))
        for pos, param in enumerate(PARAMS):
            ax = axes[pos]
            param = PARAMS[pos]
            means = [f.bootstrapResult.meanDct[param] for f in fitters]
            stds = [f.bootstrapResult.stdDct[param] for f in fitters]
            y_upper = max([s + m for s, m in zip(stds, means)])*1.1
            ax.errorbar(parameters, means, yerr=stds, linestyle="")
            ax.scatter(parameters, means, s=18.0)
            values = np.repeat(parameterDct[param].value, len(parameters))
            ax.plot(parameters, values, linestyle="dashed", color="grey")
            ax.set_title(param)
            ax.set_ylim([0, y_upper])
            if pos == len(PARAMS) - 1:
                ax.set_xticklabels(self.instanceNames)
        _ = plt.suptitle("Bootstrap Parameters With 1-Standard")
