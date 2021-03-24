# -*- coding: utf-8 -*-
"""
 Created on March 23, 2021.

@author: joseph-hellerstein

Provides a model of parallelism where there are N heterogeneous servers
of work units that provide results.
The usual interaction is that the client prepares
N work units, one for each server, the servers process their work unit,
and a collection of results is returned.

The user constructs a ServerManager that starts an instance of 
N servers, and repeatedly interacts with the ServerManager. A server must be
an instance of an AbstractServer.

    Usage
    -----
    manager = ServerManager(abstractServer, initializationArguments)
    do until done:
        arguments = list of arguments for server
        listOfResults = manager.runAll(arguments)
        user processes results
    manager.stop()


Issues
1. multiple producer or multiple consumer?
"""

from SBstoat.logs import Logger

import multiprocessing
import numpy as np


class AbstractServer(multiprocessing.Process):

    """Wrapper for worker code. Must override _function."""
    
    def __init__(self, initialArgument, inputQ, outputQ, logger=Logger()):
        self.initialArgument = initialArgument
        multiprocessing.Process.__init__(self)
        self.inputQ = inputQ
        self.outputQ = outputQ
        self.logger = logger

    def run(self):
        """
        Repeatedly execute the function with inputs
        until a None is received.
        """
        proc_name = self.name
        print("Got here 1")
        while True:
            print("Got here 2")
            next_argument = self.inputQ.get()
            if next_argument is None:
                # None argument causes process to terminate
                self.logger.activity("Process %s terminated" % self.name)
                break
            result = self._function(next_argument)
            self.outputQ.put(result)
        return

    def _function(self, argument):
        raise RuntimeError("Must override.")


class ServerManager():

    """
    Creates the consumers, interacts with them, and kills them.
    """

    def __init__(self, cls, initialArguments, **kwargs):
        """
        Parameters
        ----------
        cls: AbstractServer
        initialArguments: list
        kwargs: dict
            optional arguments for cls constructor
        """
        self.cls = cls
        self.taskTimeout = taskTimeout
        numProcess = len(initialArguments)
        # Create the consumers
        self.inputQs = [multiprocessing.Queue() for _ in range(numProcess)]
        self.outputQs = [multiprocessing.Queue() for _ in range(numProcess)]
        self.consumers = [self.cls(initialArguments[i], self.inputQs[i],
              self.outputQs[i], **kwargs) for i in range(numProcess)]
        _ = [c.start() for c in self.consumers]

    def runServers(self, arguments):
        """
        Runs each consumer once.

        Parameters
        ----------
        arguments: list
            each element results corresponds to a separate consumer

        Returns
        -------
        list
            list of results
        """
        results = []
        for idx, argument in enumerate(arguments):
            self.inputQs[idx].put(argument)
        for queue in self.outputQs:
            results.append(queue.get())
        return results

    def stopAll(self):
        """
        Terminates the processes.
        """
        for queue in self.inputQs:
            queue.put(None)
