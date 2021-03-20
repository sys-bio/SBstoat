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
"""

import multiprocessing
import numpy as np
from tqdm import tqdm

TASK_TIMEOUT = 120  # 2 minute timeout for a task
WORK_UNIT_DESC = "task"


##################### FUNCTIONS #########################
def _toplevelRunner(cls, arguments, isReport, numProcess, desc, queue):
    """
    Top level function that runs each process. Handles progress reporting.

    Parameters
    ----------
    cls: inherits from AbstractWorkUnit
    arguements: each element is an argument to construct
        an instance of cls
    isReport: bool
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
    results = manager.runAll(isReport, numProcess)
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
        count = 0
        for _ in tqdm(indices, desc=self.desc, total=len(indices)):
            if count == 0:
                count = unitMultiplier
                yield None
            count -= 1

    def _dummyGenerator(self, _):
        """Interface identical to _progressGenerator."""
        for _ in range(self.totalWork):
            yield None

    def runAll(self, isReport, numProcess):
        """
        Runs all of the work units. Handles progress bar.

        Parameters
        ----------
        isReport: bool
            display progress bar
        numProcess: int
            number of prceses
        
        Returns
        -------
        list-object
            results for each work unit
        """
        if isReport:
            generator = self._progressGenerator(numProcess)
        else:
            # This doesn't generate a progress bar
            generator = self._dummyGenerator(numProcess)
        #
        results = []
        for runner in self.runners:
            for _ in range(runner.numWorkUnit):
                if not runner.isDone:
                    results.append(runner.run())
                try:
                    _ = generator.__next__()
                except StopIteration:
                    import pdb; pdb.set_trace()
                    break
        #
        return results


class ParallelRunner():

    """
    Interface to running in parallel multiple instances of the same function.
    The user implements a class that inherits from AbstractRunner and provides
    this class to the constructor of ParallelRunner.
    """

    def __init__(self, cls, maxProcess=None,
           taskTimeout=TASK_TIMEOUT, desc=WORK_UNIT_DESC):
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
        """
        self.cls = cls
        self.taskTimeout = taskTimeout
        self.desc = desc
        if maxProcess is None:
            maxProcess = multiprocessing.cpu_count()
        self.maxProcess = min(maxProcess, multiprocessing.cpu_count())
        self.processes = []

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
        collection = [[] for _ in range(self.maxProcess)]
        for idx, argument in enumerate(arguments):
            pos = np.mod(idx, self.maxProcess)
            collection[pos].append(argument)
        return collection


    def runSync(self, argumentsList, isParallel=True):
        """
        Runs the function for each of the arguments.
        The caller waits for completion.

        Parameters
        ----------
        argumentsList: List
            each element results in running a separate process
        isParallel: True
            runs the function in parallel

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
            for idx, arguments in enumerate(argumentsCollection):
                isReporter = idx == 0
                process = multiprocessing.Process(target=_toplevelRunner,
                      args=(self.cls, arguments, isReporter, self.maxProcess,
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
            isReport = True
            results = _toplevelRunner(self.cls, argumentsList,
                isReport, numProcess, self.desc, None)
        return results
