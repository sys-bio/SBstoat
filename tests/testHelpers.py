"""
Created on Nov 11, 2020

@author: joseph-hellerstein
"""

from SBstoat import _helpers

import copy
import numpy as np
import os
import unittest


IGNORE_TEST = False
IS_PLOT = False


class TestFunctions(unittest.TestCase):

    def testCalcRelError(self):
        if IGNORE_TEST:
            return
        result = _helpers.calcRelError(1.5, 1)
        self.assertEquals(result, 1.0/3)
        result = _helpers.calcRelError(1.5, 1, isAbsolute=False)
        self.assertEquals(result, -1.0/3)
        #
        result = _helpers.calcRelError(0, 1)
        self.assertTrue(np.isnan(result))

    def testFilterOutliersFromZero(self):
        if IGNORE_TEST:
            return
        MAX_SL = 0.1
        SIZE = 1000
        def test(data, baseData):
            newData = _helpers.filterOutliersFromZero(data, MAX_SL)
            diff = set(baseData).symmetric_difference(newData)
            self.assertEqual(len(diff), 0)
        #
        baseData = [1, 1.1, 1.2, 1.3, 1.4]
        data = list(baseData)
        data.insert(1, 10)
        data.insert(3, -2)
        test(data, baseData)
        #
        data = np.random.random(SIZE)
        test(data, data)
        #
        baseData = list(data)
        data = list(baseData)
        data.insert(SIZE-10, 10)
        data.insert(SIZE-10, 20)
        test(data, baseData)

if __name__ == '__main__':
    unittest.main()
