# -*- coding: utf-8 -*-
"""
 Created on August 18, 2020

@author: joseph-hellerstein

Reports for model fitter
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME, mkNamedTimeseries
from SBstoat._modelFitterBootstrap import ModelFitterBootstrap

import lmfit
import numpy as np
import pandas as pd
import typing


##############################
class ModelFitterReport(ModelFitterBootstrap):

    def reportFit(self)->str:
        """
        Provides details of the parameter fit.
        Notes:
            1. Deletes lines about the optimization performance since the

        
        Example
        -------
        f.reportFit()
        """
        self._checkFit()
        if self.minimizerResult is None:
            raise ValueError("Must do fitModel before reportFit.")
        reportSplit = str(lmfit.fit_report(self.minimizerResult)).split("\n")
        newReportSplit = [r for r in reportSplit if not " # " in r]
        return "\n".join(newReportSplit)

    def reportBootstrap(self):
        """
        Prints a report of the bootstrap results.
        ----------
        
        Example
        -------
        f.reportBootstrap()
        """
        self._checkBootstrap()
        self.logger.activity(self.bootstrapResult)
