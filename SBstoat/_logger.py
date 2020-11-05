"""
Manages logging.
Messages are structured as follows:
    activity: some kind of significant processing
    result: outcome of the major processing
    status: status of an activity: start, finish
"""

class Logger(object):

    def activity(self, msg, preString=""):
       # Major processing activity
       print("\n\n***%s***" %msg)
    
    def result(self, msg, preString=""):
       # Result of an activity
       print("\n **%s" %msg)
    
    def status(self, msg, preString=""):
       # Progress message
       print("    (%s)" %msg)
