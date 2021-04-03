"""
Class that does fitting for a suite of related models.

A parameter has a lower bound, upper bound, and value.
Parameters are an lmfit collection of parameter.
A parameter collection is a collection of parameters.
"""

from SBstoat import _constants as cn
from SBstoat._suiteFitterCrossValidator import SuiteFitterCrossValidator

class SuiteFitter(SuiteFitterCrossValidator):
    pass
