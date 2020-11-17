# -*- coding: utf-8 -*-
"""
Created on Nov 16, 2020

@author: joseph-hellerstein
"""

from SBstoat.mainTestHarness import Runner

import os
import unittest
import matplotlib
import matplotlib.pyplot as plt


IGNORE_TEST = False
IS_PLOT = False
DIR = os.path.dirname(os.path.abspath(__file__))
PCL_PATH = os.path.join(DIR, "testMainTestHarness.pcl")
FIG_PATH = os.path.join(DIR, "testMainTestHarness.png")
FILES = [PCL_PATH, FIG_PATH]
       
 
class TestRunner(unittest.TestCase):

    def setUp(self):
        self._remove()
        self.runner = Runner(firstModel=202, numModel=2, useExisting=False,
              figPath=FIG_PATH, pclPath=PCL_PATH)
    
    def tearDown(self):
        self._remove()

    def _remove(self):
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertFalse(os.path.isfile(PCL_PATH))
        runner = Runner(firstModel=202, numModel=2, useExisting=False,
              figPath=None, pclPath=PCL_PATH)
        self.assertFalse(os.path.isfile(PCL_PATH))


if __name__ == '__main__':
    #matplotlib.use('TkAgg')
    unittest.main()
