"""
Manages logging. There are two parts: messages and performance statistics

Messages are structured as follows:
    activity: some kind of significant processing
    result: outcome of the major processing
    status: status of an activity: start, finish
    exception: an exception occurred
    error: a processing error occurred

Perofrmance logging is provided as well. Usage is:
    blockGuid = logger.startBlock("My block")
    ... lots of code ...
    logger.endBlock(blockGuid)
    ... more code ...
    print(logger.performanceDF)  # performance data
"""
from SBstoat import _helpers

import pandas as pd
import time

SEPARATOR = "/"  # Separates levels in performance logs
LEVEL_SUPPRESS = 0
LEVEL_ACTIVITY = 1
LEVEL_RESULT = 2
LEVEL_STATUS = 3
LEVEL_EXCEPTION = 4
LEVEL_ERROR = 5
LEVEL_DETAILS = 6
LEVEL_MAX = LEVEL_DETAILS
# Dataframe columns
COUNT = "count"
MEAN = "mean"
TOTAL = "total"
# Attributes used in equals comparisons
STATISTIC_ATTR_MERGE = ["count", "total"]
STATISTIC_ATTR_EQUALS = list(STATISTIC_ATTR_MERGE)
STATISTIC_ATTR_EQUALS.append("mean")
LOGGER_ATTR = ["isReport", "toFile", "startTime", "logLevel", "unpairedBlocks",
      "blockDct", "performanceDF", "statisticDct"]


class BlockSpecification():
    # Describes an entry for timing a block of code
    guid = 0

    def __init__(self, block):
        self.guid = BlockSpecification.guid
        BlockSpecification.guid += 1
        self.startTime = time.time()
        self.block = block
        self.duration = None  # Time duration of the block

    def setDuration(self):
        self.duration = time.time() - self.startTime

    def __repr__(self):
        result = "guid: %d, block: %s, startTime: %f"  \
              % (self.guid, self.block, self.startTime)
        if self.duration is not None:
            result += ", duration: %f" % self.duration
        return result


class Statistic():
    # Statistics for a block
    def __init__(self, block=None):
        self.block = block
        self.count = 0
        self.total = 0.0
        self.mean = None

    def __repr__(self):
        result = "Statistic[block: %s, count: %d, total: %f"  \
              % (self.block, self.count, self.total)
        if self.mean is not None:
            result += "mean: %f"  % self.mean
        result += "]"
        return result

    def copy(self):
        return _helpers.copyObject(self)

    def update(self, value):
        self.count += 1
        self.total += value

    def equals(self, other):
        true = True
        for attr in STATISTIC_ATTR_EQUALS:
            result = self.__getattribute__(attr) == other.__getattribute__(attr)
            if not isinstance(result, bool):
                result = all(result)
            true = true and result
        return true

    def merge(self, other):
        newStatistic = self.copy()
        for attr in STATISTIC_ATTR_MERGE:
            value = self.__getattribute__(attr) + other.__getattribute__(attr)
            newStatistic.__setattr__(attr, value)
        return newStatistic

    def summarize(self):
        if self.count == 0:
            self.mean = 0.0
        else:
            self.mean = self.total/self.count


class Logger():

    def __init__(self, isReport=True, toFile=None, logLevel=LEVEL_ERROR,
          logPerformance=False):
        self.isReport = isReport
        self.toFile = toFile
        self.logPerformance = logPerformance
        self.startTime = time.time()
        self.logLevel = logLevel
        self.unpairedBlocks = 0  # Count of blocks begun without an end
        self.blockDct = {}  # key: guid, value: BlockSpecification. Must be lock protected.
        self._performanceDF = None  # Summarizes performance report
        self.statisticDct = {}

    def copy(self):
        return _helpers.copyObject(self)

    def equals(self, other):
        true = True
        for attr in LOGGER_ATTR:
            result = self.__getattribute__(attr) == other.__getattribute__(attr)
            if not isinstance(result, bool):
                result = all(result)
            true = true and result
        return true

    @property
    def performanceDF(self):
        """
        Summarizes the performance data collected.

        Returns
        -------
        pd.Series
            index: block name
            Columns: COUNT, MEAN
        """
        if self._performanceDF is None:
            # Accumulate the durations
            self.unpairedBlocks = len(self.blockDct)
            #
            indices = list(self.statisticDct.keys())
            _ = [s.summarize() for s in self.statisticDct.values()]
            means = [self.statisticDct[b].mean for b in indices]
            counts = [self.statisticDct[b].count for b in indices]
            totals = [self.statisticDct[b].total for b in indices]
            self._performanceDF = pd.DataFrame({
                  COUNT: counts,
                  MEAN: means,
                  TOTAL: totals,
                  })
            self._performanceDF.index = indices
            self._performanceDF = self.performanceDF.sort_index()
        return self._performanceDF

    def formatPerformanceDF(self, numLevel=2):
        """
        Constructs a string that trims the index of performanceDF

        Parameters
        ----------
        numLevel: int
            Number of identifier separators included.

        Returns
        -------
        str
        """
        performanceDF = self.performanceDF.copy()
        indices = self.performanceDF.index
        newIndices = list(indices)
        for idx, item in enumerate(indices):
            splits = [s for s in item.split(SEPARATOR) if len(s) > 0]
            if len(splits) > numLevel - 1:
                newIndices[idx] = ".." + SEPARATOR +  \
                       SEPARATOR.join(splits[-numLevel:])
        performanceDF.index = newIndices
        return str(performanceDF)

    def getFileDescriptor(self):
        if self.toFile is not None:
            return open(self.toFile, "a")
        return None

    @staticmethod
    def join(*args):
        """
        Joins together a list of block names.

        Parameters
        ----------
        *args: list-str

        Returns
        -------
        str
        """
        if len(args) == 1:
            return args[0]
        if len(args) == 2:
            if len(args[0]) == 0:
                return args[1]
        return SEPARATOR.join(args)

    def _write(self, msg, numNL):
        relTime = time.time() - self.startTime
        newLineStr = ('').join(["\n" for _ in range(numNL)])
        newMsg = "\n%s%f: %s" % (newLineStr, relTime, msg)
        if self.toFile is None:
            print(newMsg)
        else:
            with open(self.toFile, "a") as fd:
                fd.write(newMsg)

    def activity(self, msg):
        # Major processing activity
        if self.isReport and (self.logLevel >= LEVEL_ACTIVITY):
            self._write("***%s***" %msg, 2)

    def result(self, msg):
        # Result of an activity
        if self.isReport and (self.logLevel >= LEVEL_RESULT):
            self._write("\n **%s" %msg, 1)

    def status(self, msg):
        # Progress message
        if self.isReport and (self.logLevel >= LEVEL_STATUS):
            self._write("    (%s)" %msg, 0)

    def exception(self, msg):
        # Progress message
        if self.isReport and (self.logLevel >= LEVEL_EXCEPTION):
            self._write("    (%s)" %msg, 0)

    def error(self, msg, excp):
        # Progress message
        if self.isReport and (self.logLevel >= LEVEL_ERROR):
            fullMsg = "%s: %s" % (msg, str(excp))
            self._write("    (%s)" % fullMsg, 0)

    def details(self, msg):
        # Progress message
        if self.isReport and (self.logLevel >= LEVEL_DETAILS):
            self._write("    (%s)" %msg, 0)

    ###### BLOCK TIMINGS ######
    def startBlock(self, block:str)->float:
        """
        Records the beginning of a block.

        Parameters
        ----------
        block: name of the block

        Returns
        -------
        int: identifier for the BlockSpecification
        """
        if self.logPerformance:
            spec = BlockSpecification(block)
            self.blockDct[spec.guid] = spec
            return spec.guid
        return None

    def _merge(self, other):
        """
        Merges the results of another logger.
        """
        newLogger = self.copy()
        for block, statistic in self.statisticDct.items():
            otherStatistic = other.statisticDct[block]
            newLogger.statisticDct[block] = statistic.merge(otherStatistic)
        return newLogger

    @staticmethod
    def merge(others):
        if len(others) == 0:
            raise ValueError("Cannot provide an empty list.")
        newLogger = others[0]
        curLogger = newLogger
        for other in others[1:]:
            newLogger = curLogger._merge(other)
            curLogger = newLogger
        return newLogger


    def endBlock(self, guid:int):
        """
        Records the end of a block. Items are removed as
        statistics are accumulated.

        Parameters
        ----------
        guid: identifies the block instance
        """
        if self.logPerformance:
            if not guid in self.blockDct.keys():
                self.exception("missing guid: %d" % guid)
            else:
                spec = self.blockDct[guid]
                spec.setDuration()
                if not spec.block in self.statisticDct.keys():
                    self.statisticDct[spec.block] = Statistic(block=spec.block)
                self.statisticDct[spec.block].update(spec.duration)
                del self.blockDct[spec.guid]

    def update(self, other):
        """
        Copies the content of other to this object.

        Parameters
        ----------
        other: Logger
        """
        _ = _helpers.copyObject(other, self)
