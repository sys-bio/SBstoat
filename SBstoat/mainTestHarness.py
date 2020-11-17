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

    def __init__(self, firstModel=210, numModel=2, useExisting=False, reportInterval=10):
        self.useExisting = useExisting and os.path.isfile(PCL_FILE)
        self.reportInterval = reportInterval
        if self.useExisting:
            self.restore()
        else:
            self.firstModel = firstModel
            self.numModel = numModel
            self.numNoError = 0
            self.fitModelRelerrors = []
            self.bootstrapRelerrors = []

    def run(self):
        modelNums = self.firstModel + np.array(range(self.numModel))
        if not self.useExisting:
            nonErroredModels = []
            erroredModels = []
            for modelNum in modelNums:
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
                self.save()
                if modelNum % self.reportInterval == 0:
                    print("*** Processed model %d" % modelNum)
        self.plot()

    def save(self):
        data = [self.fitModelRelerrors, self.bootstrapRelerrors,
              self.numModel, self.numNoError, self.firstModel],
        pickle.dump(data, open(PCL_FILE, "wb"))

    def restore(self):
        data = pickle.load(open(PCL_FILE, "rb"))[0]
        self.fitModelRelerrors, self.bootstrapRelerrors,  \
              self.numModel, self.numNoError, self.firstModel = data

    def plot(self):
        fig, axes = plt.subplots(1, 2)
        self.plotRelativeErrors(axes[0], self.fitModelRelerrors, FIT_MODEL)
        self.plotRelativeErrors(axes[1], self.bootstrapRelerrors, BOOTSTRAP)
        frac = 1.0*self.numNoError/self.numModel
        suptitle = "Fraction non-errored: %2.3f" % frac
        plt.suptitle(suptitle)
        plt.savefig(FIG_FILE)
    
    def plotRelativeErrors(self, ax, relErrors, title):
        ax.hist(relErrors)
        ax.set_title(title)
        ax.set_xlabel("relative error")
    

if __name__ == '__main__':
    runner = Runner(numModel=50, useExisting=True)
    runner.run()
