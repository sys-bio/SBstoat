"""Class that does fitting for a suite of related models."""

import SBstoat._constants as cn
from SBstoat.modelFitter import modelFitter

import numpy as np
import lmfit


class ModelSuite(object):

    def __init__(self, models, datasets, parametersets,
          modelNames=None, modelWeights=None, fitterMethods=None,
          **kwargs):
        """
        Parameters
        ----------
        models: list-modelSPecification argument for ModelFitter
        datasets: list-observedData argument for ModelFitter
        parametersets: list-iparametersToFit argument for modelFitter
        modelWeights: list-float
            how models are weighted in least squares
        modelNames: list-str
        kwargs: dict
            constructor arguments passed to fitter for a model

        Raises
        ------
        ValueError: len(models) == len(parametersets) == len(datasets)
                     == len(modelNames)
        """
        if modelWeights is None:
            modelWeights = np.repeat(1, len(models))
        if modelNames is None:
            modelNames = [str(v) for v in np.repeat(1, len(models)]
        #
        if len(models) != len(datasets):
            raise ValueError("Number of datasets must equal number of models")
        if len(models) != len(parametersets):
            raise ValueError("Number of parametersets must equal number of models")
        if len(models) != len(modelWeights):
            raise ValueError("Number of modelWeights must equal number of models")
        if len(models) != len(modelNames):
            raise ValueError("Number of modelNames must equal number of models")
        #
        self._modelWeights = modelWeights
        self.fitterDct = {}  # key: model name, value: ModelFitter
        for model, dataset, parametersToFit, modelName in   \
              zip(models, datatsets, parametersets, modelNames):
            self.fitterDct[modelName] = ModelFitter(model, dataset,
                  parametersToFit=parametersToFit,
                  **kwargs)

    def fitSuite(self, params:lmfit.Parameters=None, max_nfev:int=100):
        """
        Fits the model by adjusting values of parameters based on
        differences between simulated and provided values of
        floating species.

        Parameters
        ----------
        params: lmfit.parameters
            starting values of parameters
        max_nfev: int
            Maximum number of iterations for an evaluation

        Example
        -------
        f.fitSuite()
        """
        
