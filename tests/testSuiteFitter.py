# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

import SBstoat._constants as cn

import collections
import matplotlib
import numpy as np
import lmfit
import unittest

try:
    matplotlib.use('TkAgg')
except ImportError:
    pass


IGNORE_TEST = False
IS_PLOT = False



################ TEST CLASSES #############
class TestSuiteFitter(unittest.TestCase):

    def setUp(self):

    def testConstructor(self):
        if IGNORE_TEST:
            return

        

if __name__ == '__main__':
    unittest.main()
