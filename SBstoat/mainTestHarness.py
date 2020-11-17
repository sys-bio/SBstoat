"""
Runs TestHarness for BioModels

@author: joseph-hellerstein
"""

from SBstoat._testHarness import TestHarness
from SBstoat._logger import Logger

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

matplotlib.use('TkAgg')


IGNORE_TEST = True
IS_PLOT = True
DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(DIR), "biomodels")
PATH_PAT = os.path.join(DATA_DIR, "BIOMD0000000%d.xml")
FIRST_MODEL = 210
NUM_MODEL = 2
PCL_FILE = "mainTestHarness.pcl"
FIT_MODEL = "fitModel"
BOOTSTRAP = "bootstrap"
NUM_NOERROR = "num_noerror"
NUM_MODEL = "num_model"
FIG_FILE = "mainTestHarness.png"


class Runner(object):

    """Runs tests on biomodels."""

    def __init__(self, firstModel:int=210, numModel:int=2,
          pclPath=PCL_FILE, figPath=FIG_FILE,
          useExisting:bool=False, reportInterval:int=10):
        """
        Parameters
        ----------
        firstModel: first model to use
        numModel: number of models to use
        pclPath: file to which results are saved
        reportInterval: how frequently report progress
        """
        self.pclPath = pclPath
        self.figPath = figPath
        self.reportInterval = reportInterval
        self.useExisting = useExisting and os.path.isfile(PCL_FILE)
        if self.useExisting:
            self.restore()
        else:
            self.firstModel = firstModel
            self.numModel = numModel
            self.numNoError = 0
            self.fitModelRelerrors = []
            self.bootstrapRelerrors = []
            self.processedModels = []

    def run(self):
        """
        Runs the tests. Saves state after each tests.
        """
        modelNums = self.firstModel + np.array(range(self.numModel))
        if not self.useExisting:
            nonErroredModels = []
            erroredModels = []
            for modelNum in modelNums:
                if not modelNum in self.processedModels:
                    logger = Logger(isReport=False)
                    input_path = PATH_PAT % modelNum
                    try:
                        harness = TestHarness(input_path, logger=logger)
                        harness.evaluate(stdResiduals=1.0, fractionParameterDeviation=1.0,
                              relError=2.0)
                        nonErroredModels.append(modelNum)
                        values = [v for v in 
                              harness.fitModelResult.parameterRelErrorDct.values()]
                        self.fitModelRelerrors.extend(values)
                        values = [v for v in 
                              harness.bootstrapResult.parameterRelErrorDct.values()]
                        self.bootstrapRelerrors.extend(values)
                    except:
                        erroredModels.append(modelNum)
                    self.numNoError = self.numModel - len(erroredModels)
                    self.processedModels.append(modelNum)
                    self.save()
                    if modelNum % self.reportInterval == 0:
                        print("*** Processed model %d" % modelNum)
        self.plot()

    def save(self):
        """
        Saves state. Maintain in sync with self.restore().
        """
        if self.pclPath is not None:
            data = [self.fitModelRelerrors, self.bootstrapRelerrors,
                  self.numModel, self.numNoError, self.firstModel,
                  self.processedModels],
            with (open(self.pclPath, "wb")) as fd:
                pickle.dump(data, fd)

    def restore(self):
        """
        Restores state. Maintain in sync with self.save().
        """
        if os.path.isfile(self.pclPath):
            with (open(self.pclPath, "rb")) as fd:
                data = pickle.load(fd)[0]
            self.fitModelRelerrors, self.bootstrapRelerrors,  \
                  self.numModel, self.numNoError, self.firstModel,  \
                  self.processedModels = data
        else:
            raise ValueError("***Restart file %s does not exist"
                  % self.pclPath)

    def plot(self):
        """
        Does all plots.
        """
        fig, axes = plt.subplots(1, 2)
        self.plotRelativeErrors(axes[0], self.fitModelRelerrors, FIT_MODEL)
        self.plotRelativeErrors(axes[1], self.bootstrapRelerrors, BOOTSTRAP)
        frac = 1.0*self.numNoError/len(self.processedModels)
        suptitle = "Models %d-%d. Fraction non-errored: %2.3f"
        lastModel = self.firstModel + len(self.processedModels)
        suptitle = suptitle % (self.firstModel, lastModel, frac)
        plt.suptitle(suptitle)
        plt.savefig(self.figPath)
    
    def plotRelativeErrors(self, ax, relErrors, title):
        ax.hist(relErrors)
        ax.set_title(title)
        ax.set_xlabel("relative error")
    

if __name__ == '__main__':
    runner = Runner(firstModel=400, numModel=200, useExisting=True)
    runner.run()
