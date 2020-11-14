"""Helper functions used in SBstoat."""


from SBstoat import _plotOptions as po

import numpy as np
import scipy

INDENTATION = "  "
NULL_STR = ""


def updatePlotDocstring(target, keyphrase=None):
    """
    Changes the docstring of plot function to include all
    plot options.

    Parameters
    ----------
    target: class or function
    keyprhase: string searched for in docstring
    """
    # Place import here to avoid circular dependencies
    plot_options = str(po.PlotOptions())
    def updateFunctionDocstring(func):
        docstring = func.__doc__
        if not po.EXPAND_KEYPHRASE in docstring:
            msg = "Keyword not found in method: %s"  \
                  % func.__name__
            raise RuntimeError(msg)
        new_docstring =  \
              docstring.replace(
                    po.EXPAND_KEYPHRASE, plot_options)
        func.__doc__ = new_docstring
    #
    if "__call__" in dir(target):
        # Handle a function
        updateFunctionDocstring(target)
    else:
        # Update a class
        cls = target
        for name in dir(cls):
            if name[0:4] == po.PLOT:
                method = eval("cls.%s" % name)
                updateFunctionDocstring(method)


class Report():
    """Class used to generate reports."""

    def __init__(self):
        self.reportStr= NULL_STR
        self.numIndent = 0

    def indent(self, num: int):
        self.numIndent += num

    def _getIndentStr(self):
        return NULL_STR.join(np.repeat(
              INDENTATION, self.numIndent))
    
    def addHeader(self, title:str):
        indentStr = self._getIndentStr()
        self.reportStr+= "\n%s%s" % (indentStr, title)

    def addTerm(self, name:str, value:object):
        indentStr = self._getIndentStr()
        self.reportStr+= "\n%s%s: %s" %  \
              (indentStr, name, str(value))

    def get(self)->str:
        return self.reportStr

def calcRelError(actual:float, estimated:float, isAbsolute:bool=True):
    """
    Calculates the relative error of the estimate.

    Parameters
    ----------
    actual: actual value
    estimated: estimated values
    isAbsolute: return absolute value
    
    Returns
    -------
    float
    """
    if np.isclose(actual, 0):
        return np.nan
    relError = (estimated - actual) / actual
    if isAbsolute:
        relError = np.abs(relError)
    return relError

def filterOutliersFromZero(data, maxSL):
    """
    Removes values that are distant from 0 using a F-statistic criteria.
    Extreme values are iteratively removed until the F-statistic exceeds
    the significance level.

    Parameters
    ----------
    data: iterable-float
    maxSL: float
        Maximum significance level to accept a difference in variance
    
    Returns
    -------
    np.array
    """
    def calcSL(arr1, arr2):
        """
        Calculates the significance level that the variance of the first array
        is larger than the variance of the second array.
        Parameters
        ----------
        
        Returns
        -------
        """
        def calc(arr):
            return np.var(arr), len(arr) - 1
        #
        var1, df1 = calc(arr1)
        var2, df2 = calc(arr2)
        if var2 > 0:
            fstat = var1/var2
        else:
            fstat = 1000*var1
        sl = 1 - scipy.stats.f.cdf(fstat, df1, df2)
        return sl
    sortedData = sorted(data, key=lambda v: np.abs(v), reverse=True)
    sortedData = np.array(sortedData)
    # Construct the array without outliers
    for _ in range(len(data)):
        shortSortedData = sortedData[1:]
        sl = calcSL(sortedData, shortSortedData)
        if sl < maxSL:
            sortedData = shortSortedData
        else:
            break
    #
    return sortedData
