"""
Manages logging.
Messages are structured as follows:
    activity: some kind of significant processing
    result: outcome of the major processing
    status: status of an activity: start, finish
"""

import sys
import time

LEVEL_ACTIVITY = 1
LEVEL_RESULT = 2
LEVEL_STATUS = 3
LEVEL_EXCEPTION = 4
LEVEL_ERROR = 5
LEVEL_MAX = LEVEL_ERROR


class Logger(object):

    # Logging levels: 1

    def __init__(self, isReport=True, toFile=None, logLevel=LEVEL_STATUS):
        self.isReport = isReport
        self.toFile = toFile
        self.startTime = time.time()
        self.level = logLevel

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
