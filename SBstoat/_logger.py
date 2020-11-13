"""
Manages logging.
Messages are structured as follows:
    activity: some kind of significant processing
    result: outcome of the major processing
    status: status of an activity: start, finish
"""

class Logger(object):

    def __init__(self, isReport=True):
        self.isReport = isReport

    def activity(self, msg, preString=""):
       # Major processing activity
       if self.isReport:
           print("\n\n***%s***" %msg)
    
    def result(self, msg, preString=""):
       # Result of an activity
       if self.isReport:
           print("\n **%s" %msg)
    
    def status(self, msg, preString=""):
       # Progress message
       if self.isReport:
           print("    (%s)" %msg)
