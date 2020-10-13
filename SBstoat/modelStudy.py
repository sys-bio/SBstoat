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


OUT_PATH = "study_results"


class ModelStudy(object):

    def __init__(self, modelSpecification, dataSources, parametersToFit,
          **kwargs):
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
        kwargs: dict
            arguments passed to ModelFitter
        """
        self.fitters = {d: ModelFitter(modelSpecification, d, parametersToFit,
               **kwargs) for d in dataSources}

    def bootstrap(self, outPath=OUT_PATH, **kwargs):
        """
        Does bootstrap fits for the models.
        
        Parameters
        ----------
        outPath: str
            Path to the output directory containing the serialized fitters
            for the study.
        kwargs: dict
            arguments passed to ModelFitter.bootstrap
        """
        if os.path.isdir(outPath):
            shutil.rmtree(outPath)

    def rerport(self):
        """
        Plots from the study.
            1. Plot of observed and fitted values
            2. Plot of parameter values
        """
