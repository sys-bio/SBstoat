# -*- coding: utf-8 -*-
"""
Created on March 23, 2021

@author: joseph-hellerstein
"""

import SBstoat._serverManager as sm
from SBstoat.logs import Logger

import multiprocessing
import numpy as np
import unittest


IGNORE_TEST = False
IS_PLOT = False
SIZE = 10
PRIME_SIZES = [5, 10, 15]


class PrimeFinder(sm.AbstractServer):
    """A work unit is number of primes to calculate."""

    def __init__(self, initialArgument, inputQ, outputQ, isException=False,
              logger=Logger()):
        super().__init__(initialArgument, inputQ, outputQ, logger=Logger())
        self.isException = isException

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

    def runFunction(self, numPrime):
        """
        Calculates the specified number of prime numbers.

        Parameters
        ----------
        numPrime: int

        Returns
        -------
        np.array
        """
        if self.isException:
            raise RuntimeError("Generated RuntimeError.")
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
        primes = self.finder.runFunction(SIZE)
        self.assertEqual(len(primes), SIZE)

    def testRunNoException(self):
        if IGNORE_TEST:
            return
        server = PrimeFinder(None, self.inputQ, self.outputQ)
        server.start()
        self.inputQ.put(SIZE)
        result = self.outputQ.get()
        self.inputQ.put(None)
        self.assertEqual(len(result), SIZE)

    def testRunWithException(self):
        if IGNORE_TEST:
            return
        server = PrimeFinder(None, self.inputQ, self.outputQ,
              isException=True)
        server.start()
        self.inputQ.put(SIZE)
        result = self.outputQ.get()
        self.inputQ.put(None)
        self.assertIsNone(result)


class TestConsumerlRunner(unittest.TestCase):

    def _init(self):
        self.manager = sm.ServerManager(PrimeFinder, PRIME_SIZES)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self._init()
        pids = [s.pid for s in self.manager.servers]
        self.assertEqual(len(pids), len(PRIME_SIZES))
        self.manager.stop()

    def testRunServers(self):
        if IGNORE_TEST:
            return
        self._init()
        results = self.manager.submit(PRIME_SIZES)
        self.manager.stop()
        for result, size in zip(results, PRIME_SIZES):
            self.assertEqual(len(result), size)


if __name__ == '__main__':
    unittest.main()
