"""
Runs TestHarness for BioModels. Creates:
    *.png figure that plots relative errors
    *.pcl file with data collected from run
    *.log file with information about run

 Common usage:

  # Access information about command arguments
  python SBstoat/mainTestHarness.py --help

  # Process the BioModels 1-800, creating a new log file and data file
  python SBstoat/mainTestHarness.py --firstModel 1 --numModel 800

  # Process the BioModels 1-800, using the existing log and data files.
  python SBstoat/mainTestHarness.py --firstModel 1 --numModel 800 --useExistingData --useExistingLog

  # Create a plot from the existing data file
  python SBstoat/mainTestHarness.py --plot

  # Run analysis

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
# Context variables that are saved. Uses the following naming convention:
#  ends in "s" is a list: initialized to []
#  ends in "Dct" is a dict: initialized to {}
#  ends in "Path" is file path: initialized to None
#  begins with "is" is a bool: initialized to False
#  otherwise: int: initialized to 0
CONTEXT =  [ "firstModel", "numModel", "numNoError", "fitModelRelerrors",
      "bootstrapRelerrors", "processedModels", "nonErroredModels", "erroredModels",
      "modelParameterDct", "pclPath", "figPath", "isPlot", "reportInterval",
      "kwargDct"
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
          useExistingData:bool=False, reportInterval:int=10,
          isPlot=IS_PLOT, **kwargDct):
        """
        Parameters
        ----------
        firstModel: first model to use
        numModel: number of models to use
        pclPath: file to which results are saved
        reportInterval: how frequently report progress
        useExistingData: use data in existing PCL file
        """
        self.useExistingData = useExistingData and os.path.isfile(pclPath)
        # Recover previously saved results if desired
        if self.useExistingData:
            self.restore(pclPath=pclPath)
        else:
            # Initialize based on type of context variable
            for name in CONTEXT:
                if name[-1:] == "s":
                    self.__setattr__(name, [])
                elif name[-3:]  == "Dct":
                    self.__setattr__(name, {})
                elif name[-4:]  == "Path":
                    self.__setattr__(name, None)
                elif name[0:2]  == "is":
                    self.__setattr__(name, False)
                else:
                    self.__setattr__(name, 0)
        # Initialize to parameters for this instantiation
        self.firstModel = firstModel
        self.numModel = numModel
        self.pclPath = pclPath
        self.figPath = figPath
        self.reportInterval = reportInterval
        self.kwargDct = kwargDct
        self.isPlot = isPlot
        self.useExistingData = useExistingData
        #
        if LOGGER in kwargDct.keys():
            self.logger = kwargDct[LOGGER]
        else:
            self.logger = Logger()
            kwargDct[LOGGER] = self.logger
        self.save()

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
                    return False
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
        # Processing models
        modelNums = self.firstModel + np.array(range(self.numModel))
        for modelNum in modelNums:
            if (modelNum in self.processedModels) and self.useExistingData:
                continue
            else:
                self.processedModels.append(modelNum)
                input_path = PATH_PAT % modelNum
                msg = "Model %s" % input_path
                self.logger.activity(msg)
                try:
                    harness = TestHarness(input_path, **self.kwargDct)
                    if len(harness.parametersToFit) == 0:
                        self.logger.result("No fitable parameters in model.")
                        self.save()
                        continue
                    harness.evaluate(stdResiduals=1.0,
                          fractionParameterDeviation=1.0, relError=2.0)
                except Exception as err:
                    self.erroredModels.append(modelNum)
                    self.logger.error("TestHarness failed", err)
                    self.save()
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
                self.numNoError =  len(self.nonErroredModels)
                if modelNum % self.reportInterval == 0:
                    self.logger.result("Processed model %d" % modelNum)
                self.save()
        # Check for plot
        if self.isPlot:
            self.plot()

    def save(self):
        """
        Saves state. Maintain in sync with self.restore().
        """
        if self.pclPath is not None:
            data = [self.__getattribute__(n) for n in CONTEXT]
            with (open(self.pclPath, "wb")) as fd:
                pickle.dump(data, fd)

    def restore(self, pclPath=None):
        """
        Restores state. Maintain in sync with self.save().
        """
        if pclPath is None:
            pclPath = self.pclPath
        if os.path.isfile(pclPath):
            with (open(pclPath, "rb")) as fd:
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
        maxBin1 = self._plotRelativeErrors(axes[0], self.fitModelRelerrors,
              FIT_MODEL)
        maxBin2 = self._plotRelativeErrors(axes[1], self.bootstrapRelerrors,
              BOOTSTRAP, isYLabel=False)
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
        plt.show()
        plt.savefig(self.figPath)
    
    def _plotRelativeErrors(self, ax, relErrors, title, isYLabel=True):
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
        if isYLabel:
            ax.set_ylabel("number parameters")
        ax.set_xlim([0, 1])
        return max(rr[0])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SBstoat tests for BioModels.')
    default = 1
    parser.add_argument('--firstModel', type=int,
        help='First BioModel to process (int); default: %d' % default,
        default=default)
    default = 0
    parser.add_argument('--numModel', type=int,
        help='Number of models to process (int); default = %d' % default,
        default=default)
    parser.add_argument('--logPath', type=str,
        help='Path for log file (str); default: %s' % LOG_PATH,
        default=LOG_PATH)
    parser.add_argument('--figPath', type=str,
        help='Path for figure (str); Default: %s' % FIG_PATH,
        default=FIG_PATH)
    parser.add_argument('--useExistingData', action='store_true',
        help="Use saved data from an previous run (flag).")
    parser.add_argument('--plot', action='store_true',
        help="Plot existing data (flag).")
    parser.add_argument('--useExistingLog', action='store_true',
        help="Append to the existing log file, if it exists (flag).")
    args = parser.parse_args()
    useExistingLog = args.plot or args.useExistingLog
    useExistingData = args.plot or args.useExistingData
    #
    if not useExistingLog:
        remove(args.logPath)
    runner = Runner(firstModel=args.firstModel,
                    numModel=args.numModel,
                    useExistingData=useExistingData,
                    figPath=args.figPath,
                    isPlot=args.plot,
                    logger=Logger(toFile=args.logPath))
    runner.run()
