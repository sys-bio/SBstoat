"""
Harness used to test codes using SBML models.

Usage
-----
harness = TestHarness(sbmlPath, variableNames, parameterNames)
harness.initialize()  # Validates the parameters provided
harness.evaluate(stdResiduals=0.5, maxRelativeError=0.1)
"""

import tellurium as tellurium
from SBstoat.modelFitter import ModelFitter

import httplib2
import os
import typing


HTTP200 = 200


class TestHarness(object):

    def __init__(self, sbmlPath:str,
          variableNames:typing.List[str],
          parameterNames:typing.List[str]):
        """
        Parameters
        ----------
        sbmlPath: path or URL to SBML files
        variableNames: names of floating species used in fitting
        parameterNames: names of parameters to fit
        """
        self.sbmlPath = sbmlPath
        self.variableNames = variableNames
        self.parameterNames = parameterNames
        self.roadRunner = None

    def _checkNamesInModel(names:typing.List[str], errorMsgPattern:str):
        """
        Verifies that the list of names is in the model.

        Parameters
        ----------
        names: what to check
        errorMsgPattern: string pattern taking one argument (name) for error msg
        
        Raises ValueError
        """
        for name in self.variableNames:
            if not name in self.roadRunner.model.keys():
                raise ValueError(errorMsgPatter % name)

    def initialize(self):
        """
        Initializes the object and does error checking.
        """
        # Validate the SBML path
        if HTTP in self.sbmlPath:
            connection = httplib2.HTTPConnection('www.example.com')
            connection.request("HEAD", '')
            status = connection.getresponse().status
            if status == HTTP200:
                msg = "sbmlPath appears to be a URL, "
                msg += "but URL returns status: %s" % str(status)
                raise ValueError(msg)
        else:
            if not os.path.isfile(self.sbmlPath):
                msg = "sbmlPath appears to be a file, "
                msg += "but file does not exist: %s" % self.sbmlPath
                raise ValueError(msg)
        self.roadRunner = te.load(self.sbmlPath)
        # Validate the column names
        errorMsgPatter = "Variable name is not in model: %s"
        self._checkNamesInModel(self.variableNames, errorMsgPattern)
        # Validate the parameter names
        errorMsgPatter = "Parameter name is not in model: %s"
        self._checkNamesInModel(self.parameterNames, errorMsgPattern)
        
      


