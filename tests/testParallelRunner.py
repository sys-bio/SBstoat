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


class PrimeFinder(pr.AbstractRunner):
    """A work unit is the calculation of a prime."""

    def __init__(self, count):
        self.count = count  # each count is a work unit
        self._primes = []
        self._isDone = False

    @property
    def numWorkUnit(self):
        return self.count

    @property
    def isDone(self):
        return self._isDone

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

    def run(self):
        # Find primes until enough are accumulated
        if len(self._primes) == 0:
            num = 2
        else:
            num = self._primes[-1] + 1  # Start with number after the last prime
        if len(self._primes) < self.count:
            while not self._isPrime(num, self._primes):
                num += 1
            self._primes.append(num)
        if len(self._primes) == self.count:
            self._isDone = True
        return num


class TestRunnerManager(unittest.TestCase):

    def setUp(self):
        self.arguments = [10, 20, 30]
        self.numWork = sum(self.arguments)
        self.manager = pr.RunnerManager(PrimeFinder, self.arguments, "primes")

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(len(self.arguments), len(self.manager.runners))

    def generatorTest(self, generator):
        count = 0
        for _ in generator:
            count += 1
        self.assertEqual(count, self.numWork)

    def testProgressGenerator(self):
        if IGNORE_TEST:
            return
        MULTIPLIER = 2
        generator = self.manager._progressGenerator(MULTIPLIER)
        self.generatorTest(generator)

    def testDummyGenerator(self):
        if IGNORE_TEST:
            return
        MULTIPLIER = 2
        generator = self.manager._dummyGenerator(MULTIPLIER)
        self.generatorTest(generator)

    def testRunAll(self):
        if IGNORE_TEST:
            return
        for numProcess in [1, 2]:
            for isReport in [False, True]:
                manager = pr.RunnerManager(PrimeFinder, self.arguments, "primes")
                results = manager.runAll(isReport, numProcess)
                self.assertEqual(len(results), self.numWork)


class TestParallelRunner(unittest.TestCase):

    def setUp(self):
        self.runner = pr.ParallelRunner(PrimeFinder)

    def testPrimeFinder(self):
        if IGNORE_TEST:
            return
        finder = PrimeFinder(COUNT)
        primes = []
        while not finder.isDone:
            primes.append(finder.run())
        self.assertEqual(len(primes), COUNT)

    def runPrimes(self, **kwargs):
        SIZE = 15
        arguments = np.repeat(COUNT, SIZE)
        numWork = COUNT*SIZE
        results = self.runner.runSync(arguments, **kwargs)
        self.assertEqual(len(results), numWork)
        trues = [r is not None for r in results]
        self.assertTrue(all(trues))


    def testToplevelRunner(self):
        if IGNORE_TEST:
            return
        NUM_PROCESS = 1
        ARGUMENTS = [5, 10, 15]
        numWork = sum(ARGUMENTS)
        result2 = pr._toplevelRunner(PrimeFinder, ARGUMENTS, False, NUM_PROCESS,
            "iterations", None)
        result1 = pr._toplevelRunner(PrimeFinder, ARGUMENTS, True, NUM_PROCESS,
            "iterations", None)
        self.assertEqual(len(result1), numWork)
        self.assertEqual(len(result1), len(result2))

    def testRunner2(self):
        if IGNORE_TEST:
            return
        NUM_PROCESS = 5
        ARGUMENTS = [5, 10, 15]
        numWork = sum(ARGUMENTS)
        result1 = pr._toplevelRunner(PrimeFinder, ARGUMENTS, True,
            NUM_PROCESS, "tasks", None)
        result2 = pr._toplevelRunner(PrimeFinder, ARGUMENTS, False,
             NUM_PROCESS, "tasks", None)
        self.assertEqual(len(result1), numWork)
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
        runner = pr.ParallelRunner(PrimeFinder, maxProcess=MAX_PROCESS)
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
