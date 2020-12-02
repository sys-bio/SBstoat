# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:24:09 2020

@author: hsauro
@author: joseph-hellerstein

Timing history

date         Version         numIteration    numProcess  Time (sec)
11/30/2020   1.0             10,000          5           11.29
11/30/2020   1.1             10,000          5           110.0
"""

from SBstoat.modelFitter import ModelFitter
from SBstoat import _logger

import numpy as np
import os
import time


BENCHMARK1_TIME = 30 # Actual is 20 sec
DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(DIR, "groundtruth_2_step_0_1.txt")
MODEL = """
    J1: S1 -> S2; k1*S1
    J2: S2 -> S3; k2*S2
   
    S1 = 1; S2 = 0; S3 = 0;
    k1 = 0; k2 = 0; 
"""
        

def main(numIteration):
    """
    Calculates the time to run iterations of the benchmark.

    Parameters
    ----------
    numIteration: int
    
    Returns
    -------
    float: time in seconds
    """
    logger = _logger.Logger(logLevel=_logger.LEVEL_MAX)
    fitter = ModelFitter(MODEL, BENCHMARK_PATH,
          ["k1", "k2"], selectedColumns=['S1', 'S3'], isPlot=False,
          logger=logger)
    fitter.fitModel()
    startTime = time.time()
    fitter.bootstrap(numIteration=numIteration, reportInterval=numIteration)
    elapsedTime = time.time() - startTime
    print(logger.performanceDF)
    return elapsedTime
        

if __name__ == '__main__':
    print("Elapsed time: %4.2f" % main(300))
