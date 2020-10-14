# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19, 2020

@author: joseph-hellerstein
"""

from SBstoat.modelStudy import ModelStudy
import tests._testHelpers as th

import numpy as np
import os
import shutil
import unittest


IGNORE_TEST = False
TIMESERIES = th.getTimeseries()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SERIALIZE_DIR = os.path.join(THIS_DIR, "modelStudy")
DATA_FILE = os.path.join(THIS_DIR, "tst_data.txt")
DATA_FILE2 = os.path.join(THIS_DIR, "tst_data2.txt")
FILES = []
DIRS = [SERIALIZE_DIR]
        

class TestModelFitterCore(unittest.TestCase):

    def setUp(self):
        self._remove()
        self.study = ModelStudy(th.ANTIMONY_MODEL,
              [DATA_FILE, DATA_FILE2], th.PARAMETER_DCT.keys())
    
    def tearDown(self):
        self._remove()

    def _remove(self):
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)
        for ddir in DIRS:
            if os.path.isdir(ddir):
                shutil.rmtree(ddir)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(len(self.study.fitters), 0)
        

if __name__ == '__main__':
    unittest.main()
