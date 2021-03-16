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
COUNT = 5000


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
        SIZE = 15
        arguments = np.repeat(COUNT, SIZE)
        results = self.runner.runSync(arguments, **kwargs)
        self.assertEqual(len(results), SIZE)
        for idx in range(1, SIZE):
            diff = set(results[0]).symmetric_difference(results[idx])
            self.assertEqual(len(diff), 0)

    def testRunner(self):
        if IGNORE_TEST:
            return
        NUM_WORK = 15
        NUM_PROCESS = 5
        ARGUMENTS = [5, 10, 15]
        result1 = pr._runner(findPrimes, ARGUMENTS, 0, NUM_WORK, NUM_PROCESS,
              "task", None)
        result2 = pr._runner(findPrimes, ARGUMENTS, 1, NUM_WORK, NUM_PROCESS,
              "task", None)
        self.assertEqual(len(result1), NUM_WORK // NUM_PROCESS)
        self.assertEqual(len(result1), len(result2))

    def testRunSync(self):
        if IGNORE_TEST:
            return
        self.runPrimes()

    def testRunSyncSequential(self):
        if IGNORE_TEST:
            return
        self.runPrimes(isParallel=False)

    def testMkArgumentCollections(self):
        if IGNORE_TEST:
            return
        MAX_SIZE = 20
        MAX_PROCESS = 3
        runner = pr.ParallelRunner(findPrimes, maxProcess=MAX_PROCESS)
        for size in range(1, MAX_SIZE+1):
            arguments = list(range(size))
            argumentsCollection = runner._mkArgumentsCollections(arguments)
            newArguments = []
            _ = [newArguments.extend(c) for c in argumentsCollection]
            diff = set(arguments).symmetric_difference(newArguments)
            self.assertEqual(len(diff), 0)
            trues = [len(argumentsCollection[0]) - len(c) in [0, 1]
                  for c in argumentsCollection]
            self.assertTrue(all(trues))
        

if __name__ == '__main__':
    unittest.main()
