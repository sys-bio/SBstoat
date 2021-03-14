# -*- coding: utf-8 -*-
"""
Created on March 14, 2021

@author: joseph-hellerstein
"""

import SBstoat._parallelRunner as pr

import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
COUNT = 10000


def findPrimes(count=COUNT):
    # finds the specified number of primes

    def isPrime(number, primes):
        upper = int(np.sqrt(number))
        maxNumber = np.sqrt(number)
        for prime in primes:
            if prime > maxNumber:
                return True
            if np.mod(number, prime) == 0:
                return False
        return True
    #
    currentInt = 2
    primes = []
    done = False
    while len(primes) < count:
        if isPrime(currentInt, primes):
            primes.append(currentInt)
        currentInt += 1
    #
    return primes
    

class TestParallelRunner(unittest.TestCase):

    def setUp(self):
        self.runner = pr.ParallelRunner(findPrimes)

    def testFindPrimes(self):
        if IGNORE_TEST:
            return
        primes = findPrimes()
        self.assertEqual(len(primes), COUNT)

    def runPrimes(self, **kwargs):
        SIZE = 4
        arguments = np.repeat(COUNT, SIZE)
        results = self.runner.runSync(arguments, **kwargs)
        self.assertEqual(len(results), SIZE)
        for idx in range(1, SIZE):
            diff = set(results[0]).symmetric_difference(results[idx])
            self.assertEqual(len(diff), 0)

    def testRunSync(self):
        if IGNORE_TEST:
            return
        self.runPrimes()

    def testRunSyncSequential(self):
        if IGNORE_TEST:
            return
        self.runPrimes(isParallel=False)
        

if __name__ == '__main__':
    unittest.main()
