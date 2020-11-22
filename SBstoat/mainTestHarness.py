"""
Runs TestHarness for BioModels

@author: joseph-hellerstein
"""

from SBstoat._testHarness import TestHarness
from SBstoat._logger import Logger

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

# Handle problem with module load
try:
    matplotlib.use('TkAgg')
except ImportError:
    pass

IGNORE_TEST = True
IS_PLOT = True
DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(DIR), "biomodels")
PATH_PAT = os.path.join(DATA_DIR, "BIOMD0000000%03d.xml")
LOG_PATH = os.path.join(DIR, "mainTestHarness.log")
FIG_PATH = os.path.join(DIR, "mainTestHarness.png")
FIRST_MODEL = 210
NUM_MODEL = 2
PCL_FILE = "mainTestHarness.pcl"
FIT_MODEL = "fitModel"
BOOTSTRAP = "bootstrap"
NUM_NOERROR = "num_noerror"
NUM_MODEL = "num_model"
LOGGER = "logger"
CONTEXT =  [ "firstModel", "numModel", "numNoError", "fitModelRelerrors",
      "bootstrapRelerrors", "processedModels", "nonErroredModels", "erroredModels",
      "modelParameterDct",
      ]


############### FUNCTIONS ##################
def str2Bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def remove(ffile):
    if os.path.isfile(ffile):
        os.remove(ffile)


############### CLASSES ##################
class Runner(object):

    """Runs tests on biomodels."""

    def __init__(self, firstModel:int=210, numModel:int=2,
          pclPath=PCL_FILE, figPath=FIG_PATH,
          useExisting:bool=False, reportInterval:int=10,
          isPlot=IS_PLOT, **kwargs):
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
        self.isPlot = isPlot
        self.reportInterval = reportInterval
        self.kwargs = kwargs
        if LOGGER in kwargs.keys():
            self.logger = kwargs[LOGGER]
        else:
            self.logger = Logger()
            kwargs[LOGGER] = self.logger
        self.useExisting = useExisting and os.path.isfile(PCL_FILE)
        if self.useExisting:
            self.restore()
        else:
            # Must be consistent with the variable CONTEXT
            self.firstModel = firstModel
            self.numModel = numModel
            self.numNoError = 0
            self.fitModelRelerrors = []
            self.bootstrapRelerrors = []
            self.processedModels = []
            self.nonErroredModels = []
            self.erroredModels = []
            self.modelParameterDct = {}

    def _isListSame(self, list1, list2):
        diff = set(list1).symmetric_difference(list2)
        return len(diff) == 0

    def equals(self, other):
        selfKeys = list(self.__dict__.keys())
        otherKeys = list(other.__dict__.keys())
        if not self._isListSame(selfKeys, otherKeys):
            return False
        #
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                isEqual = self._isListSame(value, other.__getattribute__(key))
                if not isEqual:
                    return True
            elif any([isinstance(value, t) for t in [int, str, float, bool]]):
                if self.__getattribute__(key) != other.__getattribute__(key):
                    return False
            else:
                pass
        #
        return True

    def run(self):
        """
        Runs the tests. Saves state after each tests.
        """
        modelNums = self.firstModel + np.array(range(self.numModel))
        if not self.useExisting:
            for modelNum in modelNums:
                if not modelNum in self.processedModels:
                    input_path = PATH_PAT % modelNum
                    harness = TestHarness(input_path, **self.kwargs)
                    try:
                        harness.evaluate(stdResiduals=1.0,
                              fractionParameterDeviation=1.0, relError=2.0)
                    except Exception as err:
                        self.logger.error("TestHarness failed", err)
                        continue
                    # Parameters for model
                    self.modelParameterDct[modelNum] =  \
                          list(harness.fitModelResult.parameterRelErrorDct.keys())
                    # Relative error in initial fit
                    values = [v for v in 
                          harness.fitModelResult.parameterRelErrorDct.values()]
                    self.fitModelRelerrors.extend(values)
                    # Relative error in bootstrap
                    values = [v for v in 
                          harness.bootstrapResult.parameterRelErrorDct.values()]
                    self.bootstrapRelerrors.extend(values)
                    # Count models without exceptions
                    self.nonErroredModels.append(modelNum)
                    self.erroredModels.append(modelNum)
                    self.numNoError =  len(self.nonErroredModels)
                    self.processedModels.append(modelNum)
                    self.save()
                    if modelNum % self.reportInterval == 0:
                        self.logger.result("Processed model %d" % modelNum)
        self.plot()

    def save(self):
        """
        Saves state. Maintain in sync with self.restore().
        """
        if self.pclPath is not None:
            data = [self.__getattribute__(n) for n in CONTEXT]
            with (open(self.pclPath, "wb")) as fd:
                pickle.dump(data, fd)

    def restore(self):
        """
        Restores state. Maintain in sync with self.save().
        """
        if os.path.isfile(self.pclPath):
            with (open(self.pclPath, "rb")) as fd:
                data = pickle.load(fd)
            [self.__setattr__(n, v) for n, v in zip(CONTEXT, data)]
        else:
            raise ValueError("***Restart file %s does not exist"
                  % self.pclPath)

    def plot(self):
        """
        Does all plots.
        """
        fig, axes = plt.subplots(1, 2)
        maxBin1 = self._plotRelativeErrors(axes[0], self.fitModelRelerrors, FIT_MODEL)
        maxBin2 = self._plotRelativeErrors(axes[1], self.bootstrapRelerrors, BOOTSTRAP)
        maxBin = max(maxBin1, maxBin2)
        if maxBin > 0:
            axes[0].set_ylim([0, maxBin])
            axes[1].set_ylim([0, maxBin])
        #
        if len(self.processedModels) == 0:
            frac = 0.0
        else:
            frac = 1.0*self.numNoError/len(self.processedModels)
        suptitle = "Models %d-%d. Fraction non-errored: %2.3f"
        lastModel = self.firstModel + len(self.processedModels) - 1
        suptitle = suptitle % (self.firstModel, lastModel, frac)
        plt.suptitle(suptitle)
        if self.isPlot:
            plt.show()
        else:
            plt.savefig(self.figPath)
    
    def _plotRelativeErrors(self, ax, relErrors, title):
        """
        Plots histogram of relative errors.

        Parameters
        ----------
        ax: Matplotlib.axes
        relErrors: list-float
        title: str
        
        Returns
        -------
        float: maximum number in a bin
        """
        rr = ax.hist(relErrors)
        ax.set_title(title)
        ax.set_xlabel("relative error")
        ax.set_xlim([0, 1])
        return max(rr[0])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SBstoat tests for BioModels.')
    parser.add_argument('--firstModel', type=int,
        help='First BioModel to process.', default=100)
    parser.add_argument('--numModel', type=int,
        help='Number of models to process.', default=1)
    parser.add_argument('--useExisting', nargs=1, type=str2Bool,
        help="Use saved data from an previous run.",
        default = [True])
    parser.add_argument('--logPath', type=str, help='Path for log file.',
        default=LOG_PATH)
    parser.add_argument('--figPath', type=str, help='Path for figure.',
        default=FIG_PATH)
    args = parser.parse_args()
    #
    remove(args.logPath)  # Start fresh each run
    runner = Runner(firstModel=args.firstModel,
                    numModel=args.numModel,
                    useExisting=args.useExisting[0],
                    figPath=args.figPath,
                    logger=Logger(toFile=args.logPath))
    runner.run()
