"""
Manages logging.
Messages are structured as follows:
    activity: some kind of significant processing
    result: outcome of the major processing
    status: status of an activity: start, finish
"""

import sys
import time


class Logger(object):

    def __init__(self, isReport=True, toFile=None):
        self.isReport = isReport
        self.toFile = toFile
        self.startTime = time.time()

    def getFileDescriptor(self):
        if self.toFile is not None:
            return open(self.toFile, "a")
        else:
            return None

    def _write(self, msg):
        if self.toFile is None:
            print(msg)
        else:
            with open(self.toFile, "a") as fd:
                relTime = time.time() - self.startTime
                fd.write("\n%f: %s" % (relTime, msg))

    def activity(self, msg, preString=""):
       # Major processing activity
       if self.isReport:
           self._write("\n\n***%s***" %msg)
    
    def result(self, msg, preString=""):
       # Result of an activity
       if self.isReport:
           self._write("\n **%s" %msg)
    
    def status(self, msg, preString=""):
       # Progress message
       if self.isReport:
           self._write("    (%s)" %msg)
