# -*- coding: utf-8 -*-
"""
 Created on March 14, 2021.

@author: joseph-hellerstein

Provides a simple interface for running a function in parallel
with multiple instances of arguments. Provides progress bar.

The user wraps their function in a class that inherits from AbstractRunner.
This class if provided to ParallelRunner, which runs the codes in parallel.

    Usage
    -----
    runner = ParallelRunner(cls)  # cls is an AbstractRunner
    arguments = list of arguments for instances of cls
    listOfResults = runner.runSync(arguments)

Classes:

AbstractRunner. Wrapper for codes executed in parallel.
User must subclass and override:
    property: numWorkUnit
        Indicates the number of work units that will be performed by an instance.
    property: isDone
        Indicates that all work units have been completed.
    function: run()
        Do one work unit

ParallelRunner. Runs AbstractRunners in parallel.
"""

import multiprocessing
import numpy as np
from tqdm import tqdm

TASK_TIMEOUT = 120  # 2 minute timeout for a task
WORK_UNIT_DESC = "task"


##################### FUNCTIONS #########################
def _toplevelRunner(cls, arguments, isProgressBar, numProcess, desc, queue):
    """
    Top level function that runs each process. Handles progress reporting.

    Parameters
    ----------
    cls: inherits from AbstractRunner
    arguements: each element is an argument to construct
        an instance of cls
    isProgressBar: bool
        This process reports progress
    numProcess: int
    desc: str
        description of the work unit
    queue: multiprocessing queue

    Returns
    -------
    list
    """
    manager = RunnerManager(cls, arguments, desc)
    results = manager.runAll(isProgressBar, numProcess)
    # Post the results
    if queue is None:
        return results
    queue.put(results)


##################### CLASSES #########################
class AbstractRunner(object):
    """
    Wrapper for user-provided code that is run in parallel.
    An AbstractRunner has a run method that returns a list of
    work unit results.
    """

    @property
    def numWorkUnit(self):
        """
        Returns
        -------
        int: number of work units to be processed by runner
        """
        raise RuntimeError("Must override.")

    @property
    def isDone(self):
        """
        Returns
        -------
        bool: all work has been processed
        """
        raise RuntimeError("Must override.")

    def run(self):
        """
        Interface for repeated running of work units.

        Returns
        -------
        Object
            list of work unit results
        """
        raise RuntimeError("Must override.")


class RunnerManager():
    """Manages runners for a process. There is a runner for each argument."""

    def __init__(self, cls, arguments, desc):
        """
        Parameters
        ----------
        cls: inherits from AbstractWorkUnit
        arguements: each element is an argument to construct
            an instance of cls
        desc: str
            description of the work unit
        """
        self.cls = cls
        self.runners = [self.cls(a) for a in arguments]
        self.desc = desc
        try:
            self.totalWork = sum([r.numWorkUnit for r in self.runners])
        except Exception:
            msg = "Must implement attribute 'numWorkUnit' in AbstractRunner"
            raise ValueError(msg)

    def _progressGenerator(self, unitMultiplier):
        """
        Updates the progress bar.
        Should be called exactly self.totalWork times.

        Parameters
        ----------
        unitMultiplier: int
            How many units are updated for a work unit.

        Returns
        -------
        generator
        """
        allWork = unitMultiplier*self.totalWork
        indices = range(allWork)
        for idx in tqdm(indices, desc=self.desc, total=len(indices)):
            if np.mod(idx, unitMultiplier) == 0:
                yield None

    def _dummyGenerator(self, _):
        """Interface identical to _progressGenerator."""
        for _ in range(self.totalWork):
            yield None

    def runAll(self, isProgressBar, numProcess):
        """
        Runs all of the work units. Handles progress bar.

        Parameters
        ----------
        isProgressBar: bool
            display progress bar
        numProcess: int
            number of prceses

        Returns
        -------
        list-object
            results for each work unit
        """
        if isProgressBar:
            generator = self._progressGenerator(numProcess)
        else:
            # This doesn't generate a progress bar
            generator = self._dummyGenerator(numProcess)
        #
        results = []
        for runner in self.runners:
            for _ in range(runner.numWorkUnit):
                _ = generator.__next__()
                if not runner.isDone:
                    results.append(runner.run())
        # Ensure generator is emptied
        for _ in generator:
            pass
        #
        return results


class ParallelRunner():

    """
    Interface to running in parallel multiple instances of the same function.
    The user implements a class that inherits from AbstractRunner and provides
    this class to the constructor of ParallelRunner.
    """

    def __init__(self, cls, maxProcess=None,
           taskTimeout=TASK_TIMEOUT, desc=WORK_UNIT_DESC, isProgressBar=True):
        """
        Parameters
        ----------
        cls: Inherits from AbstractRunner
        maxProcess: int
            maximum number of concurrent tasks
        taskTimeout: float
            maximum runtime for a task
        desc: str
            description of the work unit
        isProgressBar: bool
            display the progress bar
        """
        self.cls = cls
        self.taskTimeout = taskTimeout
        self.desc = desc
        if maxProcess is None:
            maxProcess = multiprocessing.cpu_count()
        self.maxProcess = min(maxProcess, multiprocessing.cpu_count())
        self.processes = []
        self._isProgressBar = isProgressBar

    def _mkArgumentsCollections(self, arguments):
        """
        Allocates arguments among a set of processes.

        Parameters
        ----------
        arguments: list

        Returns
        -------
        list-list
        """
        size = min(len(arguments), self.maxProcess)
        collection = [[] for _ in range(size)]
        for idx, argument in enumerate(arguments):
            pos = np.mod(idx, size)
            collection[pos].append(argument)
        return collection


    def runSync(self, argumentsList, isParallel=True, isProgressBar=True):
        """
        Runs the function for each of the arguments.
        The caller waits for completion.

        Parameters
        ----------
        argumentsList: List
            each element results in running a separate process
        isParallel: True
            runs the function in parallel
        isProgressBar: bool
            display the progress bar

        Returns
        -------
        list
            list of results
        """
        results = []
        if isParallel:
            self.processes = []
            queue = multiprocessing.Queue()
            # Start the processes
            argumentsCollection = self._mkArgumentsCollections(argumentsList)
            numProcess = min(len(argumentsCollection), self.maxProcess)
            for idx, arguments in enumerate(argumentsCollection):
                isThisProgresBar = isProgressBar and (idx == 0)
                process = multiprocessing.Process(target=_toplevelRunner,
                      args=(self.cls, arguments, isThisProgresBar, numProcess,
                      self.desc, queue,))
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
            numProcess = 1
            results = _toplevelRunner(self.cls, argumentsList,
                isProgressBar, numProcess, self.desc, None)
        return results
