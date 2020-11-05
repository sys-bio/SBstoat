"""
Writes messages.
Messages are structured as follows:
    activity: some kind of significant processing
    result: outcome of the major processing
    status: status of an activity: start, finish
"""

def activity(msg, preString=""):
   # Major processing activity
   print("\n\n***%s***" %msg)

def result(msg, preString=""):
   # Result of an activity
   print("\n **%s" %msg)

def status(msg, preString=""):
   # Progress message
   print("    (%s)" %msg)
