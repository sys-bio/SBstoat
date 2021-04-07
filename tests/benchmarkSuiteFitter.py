# -*- coding: utf-8 -*-
"""
Created on Thurs March 25, 2021

@author: joseph-hellerstein

Timing history

date         Version  no. Fold        max_nfev  Time (sec)
3/26/2021     1.16       1              10,000   173.17
4/05/2021     1.16       3              10,000   120.19  (parallel)
"""
import SBstoat
from SBstoat.namedTimeseries import NamedTimeseries
from SBstoat.suiteFitter import SuiteFitter
from SBstoat import logs
from tests.benchmarkSuiteFitterModel import MODEL, PARAMETERS

import matplotlib
import numpy as np
import os
import tellurium as te
import time


IS_TEST = False
IS_PLOT = False
IS_PARALLEL = False
NUM_FOLD = 3
DIR = os.path.dirname(os.path.abspath(__file__))
MAX_NFEV = 10000
NUM_MODEL = 4
DATA_PAT = "benchmarkSuiteFitterData_%d.csv"
OBSERVED_FILES = [os.path.join(DIR, DATA_PAT % d) for d in range(1, NUM_MODEL+1)]
MODEL_NAMES = ["Ras%d" % n for n in range(1, NUM_MODEL+1)]
        

def main(maxNfev=MAX_NFEV):
    """
    Calculates the time to run the benchmark.

    Parameters
    ----------
    numPopulation: int
    
    Returns
    -------
    float: time in seconds
    """
    logger = logs.Logger(logLevel=logs.LEVEL_ACTIVITY, logPerformance=IS_TEST)
    models = [MODEL for _ in range(NUM_MODEL)]
    parametersList = [PARAMETERS for _ in range(NUM_MODEL)]
    optimizerMethod = SBstoat.OptimizerMethod(method="differential_evolution",
          kwargs={"popsize": 10, 'max_nfev': maxNfev})
    startTime = time.time()
    suiteFitter = SuiteFitter(models, OBSERVED_FILES, parametersList,
                              MODEL_NAMES, isParallel=IS_PARALLEL,
                              logger=logger,
                              fitterMethods=[optimizerMethod])
    if NUM_FOLD == 1:
        suiteFitter.fitSuite()
    else:
        suiteFitter.crossValidate(NUM_FOLD)
    elapsedTime = time.time() - startTime
    if IS_TEST:
        print(suiteFitter.reportFit())
    if IS_PLOT:
        suiteFitter.plotFitAll()
        suiteFitter.plotResidualsSSQ()
    return elapsedTime
        

if __name__ == '__main__':
    if IS_PLOT:
        matplotlib.use('TkAgg')
    print("Elapsed time: %4.2f" % main())
