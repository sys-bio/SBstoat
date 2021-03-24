# -*- coding: utf-8 -*-
"""
Created on March 23, 2021

@author: joseph-hellerstein
"""

import SBstoat._serverManager as sm

import multiprocessing
import numpy as np
import unittest


IGNORE_TEST = True
IS_PLOT = True
SIZE = 10


class PrimeFinder(sm.AbstractConsumer):
    """A work unit is number of primes to calculate."""

    @staticmethod
    def _isPrime(number, primes):
        if number < 2:
            return False
        maxNumber = np.sqrt(number)
        for prime in primes:
            if prime > maxNumber:
                return True
            if np.mod(number, prime) == 0:
                return False
        return True

    def _function(self, numPrime):
        """
        Calculates the specified number of prime numbers.

        Parameters
        ----------
        numPrime: int
        
        Returns
        -------
        np.array
        """
        # Find primes until enough are accumulated
        primes = []
        num = 2
        while len(primes) < numPrime:
            if self._isPrime(num, primes):
                primes.append(num)
            num += 1
        return np.array(primes)


################## CLASSES BEING TESTED ##############
class TestAbstractConsumer(unittest.TestCase):

    def setUp(self):
        self.inputQ = multiprocessing.Queue()
        self.outputQ = multiprocessing.Queue()
        self.finder = PrimeFinder(None, self.inputQ, self.outputQ)

    def testPrimeFinder(self):
        if IGNORE_TEST:
            return
        primes = self.finder._function(SIZE)
        self.assertEqual(len(primes), SIZE)

    def testRun(self):
        # TESTING
        consumer = PrimeFinder(None, self.inputQ, self.outputQ)
        consumer.start()
        self.inputQ.put(SIZE)
        result = self.outputQ.get()
        self.inputQ.put(None)
        self.assertEqual(len(result), SIZE)
        
        

class TestConsumerlRunner(unittest.TestCase):

    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()
