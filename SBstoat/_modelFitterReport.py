# -*- coding: utf-8 -*-
"""
 Created on August 18, 2020

@author: joseph-hellerstein

Reports for model fitter
"""

from SBstoat.namedTimeseries import NamedTimeseries, TIME, mkNamedTimeseries
from SBstoat._modelFitterBootstrap import ModelFitterBootstrap
from SBstoat import _helpers

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
        VARIABLE_STG = "[[Variables]]"
        CORRELATION_STG = "[[Correlations]]"
        self._checkFit()
        if self.minimizerResult is None:
            raise ValueError("Must do fitModel before reportFit.")
        valuesDct = self.params.valuesdict()
        valuesStg = _helpers.ppDict(dict(valuesDct), indent=4)
        reportSplit = str(lmfit.fit_report(self.minimizerResult)).split("\n")
        # Eliminate Variables section
        inVariableSection = False
        trimmedReportSplit = []
        for line in reportSplit:
            if VARIABLE_STG in line:
                inVariableSection = True
            if CORRELATION_STG in line:
                inVariableSection = False
            if inVariableSection:
                continue
            else:
                trimmedReportSplit.append(line)
        # Construct the report
        newReportSplit = [VARIABLE_STG]
        newReportSplit.extend(valuesStg.split("\n"))
        newReportSplit.extend(trimmedReportSplit)
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
