"""
Manages logging.
Messages are structured as follows:
    activity: some kind of significant processing
    result: outcome of the major processing
    status: status of an activity: start, finish
    exception: an exception occurred
    error: a processing error occurred

Time logging is provided as well.
    blockGuid = logger.startBlock("My block")
    ... lots of code ...
    logger.endBlock(blockGuid)
    ... more code ...
    logger.report(csvFile) # writes the results to a csv file
"""

import collections
import pandas as pd
import numpy as np
import sys
import time
from multiprocessing import Process, Manager

LEVEL_ACTIVITY = 1
LEVEL_RESULT = 2
LEVEL_STATUS = 3
LEVEL_EXCEPTION = 4
LEVEL_ERROR = 5
LEVEL_MAX = LEVEL_ERROR



class BlockSpecification(object):
    # Describes an entry for timing a block of code
    guid = 0
    
    def __init__(self, block):
        self.guid = BlockSpecification.guid
        BlockSpecification.guid += 1
        self.startTime = time.time()
        self.block = block
        self.duration = None

    def setDuration(self):
        self.duration = time.time() - self.startTime

    def __repr__(self):
        repr = "guid: %d, block: %s, startTime: %f"  \
              % (self.guid, self.block, self.startTime)
        if self.duration is not None:
            repr += ", duration: %f" % self.duration
        return repr


class Logger(object):


    def __init__(self, isReport=True, toFile=None, logLevel=LEVEL_STATUS):
        self.isReport = isReport
        self.toFile = toFile
        self.startTime = time.time()
        self.level = logLevel
        manager = Manager()
        # Make multiprocesor safe
        self.blockDct = manager.dict()  # key: guid, value: BlockSpecification

    def getFileDescriptor(self):
        if self.toFile is not None:
            return open(self.toFile, "a")
        else:
            return None

    def _write(self, msg, numNL):
        relTime = time.time() - self.startTime
        newLineStr = ('').join(["\n" for _ in range(numNL)])
        newMsg = "\n%s%f: %s" % (newLineStr, relTime, msg)
        if self.toFile is None:
            print(newMsg)
        else:
            with open(self.toFile, "a") as fd:
                fd.write(newMsg)

    def activity(self, msg, preString=""):
       # Major processing activity
       if self.isReport and (self.level >= LEVEL_ACTIVITY):
           self._write("***%s***" %msg, 2)
    
    def result(self, msg, preString=""):
       # Result of an activity
       if self.isReport and (self.level >= LEVEL_RESULT):
           self._write("\n **%s" %msg, 1)
    
    def status(self, msg, preString=""):
       # Progress message
       if self.isReport and (self.level >= LEVEL_STATUS):
           self._write("    (%s)" %msg, 0)
    
    def exception(self, msg, preString=""):
       # Progress message
       if self.isReport and (self.level >= LEVEL_EXCEPTION):
           self._write("    (%s)" %msg, 0)
    
    def error(self, msg, excp):
       # Progress message
       if self.isReport and (self.level >= LEVEL_ERROR):
           fullMsg = "%s: %s" % (msg, str(excp))
           self._write("    (%s)" % fullMsg, 0)

    ###### BLOCK TIMINGS ######
    def startBlock(self, block:str)->float:
        """
        Records the beginning of a block.
        This is multiprocessing safe.

        Parameters
        ----------
        block: name of the block
        
        Returns
        -------
        int: identifier for the BlockSpecification
        """
        spec = BlockSpecification(block)
        self.blockDct[spec.guid] = spec
        return spec.guid

    def endBlock(self, guid:int):
        """
        Records the end of a block.
        This is multiprocessing safe.

        Parameters
        ----------
        guid: identifies the block instance
        """
        spec = self.blockDct[guid]
        spec.setDuration()
        self.blockDct[guid] = spec

    def blockReport(self, csvPath:str)->pd.Series:
        """
        Writes the results of collected logs.
        This is NOT multiprocessing safe.

        Parameters
        ----------
        csvPath: path to CSV file to write

        Returns
        -------
        pd.Series: index: block, value: mean times
        """
        # Accumulate the durations
        dataDct = {}
        for spec in self.blockDct.values():
            if not spec.block in dataDct.keys():
                dataDct[spec.block] = []
            dataDct[spec.block].append(spec.duration)
        #
        meanDct = {b: np.mean(v) for b, v in dataDct.items()}
        ser = pd.Series(meanDct)
        ser.to_csv(csvPath)
        return ser
        
