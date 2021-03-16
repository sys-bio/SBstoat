# -*- coding: utf-8 -*-
"""
 Created on March 14, 2021.

@author: joseph-hellerstein

Provides a simple interface for running a function in parallel
with multiple instances of arguments.
"""

import multiprocessing
import numpy as np
from tqdm import tqdm

TASK_TIMEOUT = 120  # 2 minute timeout for a task


def _runner(function, arguments, processIdx, numWork, numProcess, desc, queue):
    """
    Wrapper for running a function.

    Parameters
    ----------
    function: 1 argument, 1 return value function
    arguements: list of arguments to function
    processIdx: int
        index of the running process
    numWork: int
        total number of work units to process
    numProcess: int
        total number of processes
    desc: str
        description of the work unit
    queue: multiprocessing queue
    
    Returns
    -------
    list
    """
    def nullIterFun(x, **kwargs):
        # Returns the argument
        return x
    #
    results = []
    # Process 0 runs the progress bar
    if processIdx == 0:
        iterFun = tqdm
    else:
        iterFun = nullIterFun
    # To provide the progress bar, iterate across all work units.
    # Select the current work unit if it is a multiple of the processIdx
    workIdxs = list(range(numWork))
    curIdx = 0
    for workIdx in iterFun(workIdxs, desc=desc, total=numWork):
        if np.mod(workIdx, numProcess) == processIdx:
            argument = arguments[curIdx]
            curIdx += 1
            results.append(function(argument))   
    #
    if queue is None:
        return results
    queue.put(results)


class ParallelRunner():

    """
    Interface to running in parallel multiple instances of the same function.
    
    Usage
    -----
    runner = ParallelRunner(function)
    arguments = list of arguments for the instances run in parallel
    listOfResults = runner.runSync(arguments)
    """

    def __init__(self, function, maxProcess=None,
          taskTimeout=TASK_TIMEOUT, desc="task"):
        """
        Parameters
        ----------
        function: Calable
            single argument
            one return value
        maxProcess: int
            maximum number of concurrent tasks
        taskTimeout: float
            maximum runtime for a task
        desc: str
            description of the work unit
        """
        self.function = function
        self.taskTimeout = taskTimeout
        self.desc = desc
        numCPU = multiprocessing.cpu_count()
        if maxProcess is None:
            maxProcess = numCPU
        self.numProcess = min(maxProcess, numCPU)
        self.processes = []

    def _mkArgumentsCollections(self, arguments):
        """
        Makes a collection of arguments lists.

        Parameters
        ----------
        arguments: list
        
        Returns
        -------
        list-list
        """
        count = len(arguments)
        collection = [[] for _ in range(self.numProcess)]
        idx = 0
        for argument in arguments:
            collection[idx].append(argument)
            if idx >= self.numProcess-1:
                idx = 0
            else:
                idx += 1
        return collection
   

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
            self.processes = []
            queue = multiprocessing.Queue()
            # Start the processes
            argumentsCollection = self._mkArgumentsCollections(arguments)
            numWork = len(arguments)
            for idx, arguments in enumerate(argumentsCollection):
                process = multiprocessing.Process(target=_runner,
                      args=(self.function, arguments, idx, numWork,
                      self.numProcess, self.desc, queue,))
                process.start()
                self.processes.append(process)
            # Wait for the results
            try:
                for idx in range(len(self.processes)):
                    timeout = len(argumentsCollection[idx])*self.taskTimeout
                    result = queue.get(timeout=timeout)
                    if result is None:
                        raise ValueError("Got None result")
                    results.extend(result)
            except Exception as err:
                print(err)
            # Get rid of possible zombies
            for process in self.processes:
                process.terminate()
        else:
            results = [self.function(a) for a in arguments]
        return results
