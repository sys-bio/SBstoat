"""
Class that does fitting for a suite of related models.

A parameter has a lower bound, upper bound, and value.
Parameters are an lmfit collection of parameter.
A parameter collection is a collection of parameters.
"""

from SBstoat import _constants as cn
from SBstoat.logs import Logger
from SBstoat.modelFitter import ModelFitter
from SBstoat._suiteFitterCrossValidator import SuiteFitterCrossValidator

class SuiteFitter(SuiteFitterCrossValidator):
    pass

def mkSuiteFitter(modelSpecifications, datasets, parametersCol,
      modelNames=None, modelWeights=None, fitterMethods=None,
      numRestart=0, isParallel=False, logger=Logger(), **kwargs):
    """
    Constructs a SuiteFitterCore with fitters that have similar
    structure.

    Parameters
    ----------
    modelSpecifications: list-modelSpecification as in ModelFitter
    datasets: list-observedData as in ModelFitter
    paramersCol: list-parametersToFit as in ModelFitter
    modelNames: list-str
    modelWeights: list-float
        how models are weighted in least squares
    fitterMethods: list-optimization methods
    numRestart: int
        number of times the minimization is restarted with random
        initial values for parameters to fit.
    isParallel: bool
        run fits in parallel for each fitter
    logger: Logger
    kwargs: dict
        keyword arguments for ModelFitter
    
    Returns
    -------
    SuiteFitter
    """
    modelFitters = []
    for modelSpecification, dataset, parametersToFit in   \
          zip(modelSpecifications, datasets, parametersCol):
        modelFitter = ModelFitter(modelSpecification, dataset,
              parametersToFit=parametersToFit, logger=logger, **kwargs)
        modelFitters.append(modelFitter)
    return SuiteFitter(modelFitters, modelNames=modelNames,
          modelWeights=modelWeights, fitterMethods=fitterMethods,
          numRestart=numRestart, isParallel=isParallel, logger=logger)
