# -*- coding: utf-8 -*-
"""
 Created on March 14, 2021.

@author: joseph-hellerstein

Provides a simple interface for running a function in parallel
with multiple instances of arguments.
"""

import multiprocessing

TIMEOUT = 120  # 2 minute timeout for a task


def _runner(function, argument, queue):
    queue.put(function(argument))   


class ParallelRunner():

    def __init__(self, function, maxProcess=None, timeout=TIMEOUT):
        """
        Parameters
        ----------
        function: Calable
            single argument
            one return value
        """
        self.function = function
        self.timeout = timeout
        numCPU = multiprocessing.cpu_count()
        if maxProcess is None:
            maxProcess = numCPU
        self.maxProcess = min(maxProcess, numCPU)

    def runSync(self, arguments, isParallel=True):
        """
        Runs the function for each of the arguments.
        The caller waits for completion.

        Parameters
        ----------
        args: List
            each element is an argument for self.function
        isParallel: True
            runs the function in parallel

        Returns
        -------
        list
            list of results from self.function
        """
        results = []
        if isParallel:
            processes = []
            queue = multiprocessing.Queue()
            # Start the processes
            for argument in arguments:
                process = multiprocessing.Process(target=_runner,
                      args=(self.function, argument, queue,))
                process.start()
                processes.append(process)
            # Wait for the results
            try:
                for _ in range(len(processes)):
                    result = queue.get(timeout=self.timeout)
                    if result is None:
                        raise ValueError("Got None result")
                    results.append(result)
            except Exception as err:
                print(err)
            # Get rid of possible zombies
            for process in processes:
                process.terminate()
        else:
            results = [self.function(a) for a in arguments]
        return results
