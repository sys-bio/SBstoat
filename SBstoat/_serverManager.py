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
        arguments = list of arguments for servers
        listOfResults = manager.submit(arguments)
        user processes results
    manager.stop()
"""

from SBstoat.logs import Logger

import multiprocessing


TIMEOUT = 120


class AbstractServer(multiprocessing.Process):

    """Wrapper for worker code. Must override runFunction."""

    def __init__(self, initialArgument, inputQ, outputQ,
              logger=Logger()):
        """
        Parameters
        ----------
        initialArgument: Object
            used by RunFunction
        inputQ: multiprocessing.queue
        outputQ: multiprocessing.queue
        logger: Logger
        """
        self.initialArgument = initialArgument
        multiprocessing.Process.__init__(self)
        self.inputQ = inputQ
        self.outputQ = outputQ
        self.logger = logger

    def run(self):
        """
        Repeatedly execute the function with inputs
        """
        done = False
        while not done:
            try:
                next_argument = self.inputQ.get(timeout=TIMEOUT)
                # Try executing the user code
                try:
                    result = self.runFunction(next_argument)
                    self.outputQ.put(result)
                except Exception as err:
                    self.logger.error("Process %s exception:" % self.name, err)
                    result = None
                    self.outputQ.put(result)
            except Exception:
                self.logger.activity("Process %s queue empty." % self.name)
                break

    def runFunction(self, argument):
        raise RuntimeError("Must override.")


class ServerManager():

    """
    Creates the servers, interacts with them, and kills them.
    """

    def __init__(self, cls, initialArguments, logger=Logger(), **kwargs):
        """
        Parameters
        ----------
        cls: AbstractServer
        initialArguments: list
        kwargs: dict
            optional arguments for cls constructor
        """
        self.cls = cls
        self.logger = logger
        self.numProcess = len(initialArguments)
        # Create the servers
        self.inputQs = [multiprocessing.Queue() for _ in range(self.numProcess)]
        self.outputQs = [multiprocessing.Queue() for _ in range(self.numProcess)]
        self.servers = [self.cls(initialArguments[i], self.inputQs[i],
              self.outputQs[i], logger=self.logger, **kwargs)
              for i in range(self.numProcess)]
        _ = [s.start() for s in self.servers]

    def submit(self, arguments):
        """
        Submits a work unit to each server. If a server does not respond,
        a None is inserted.

        Parameters
        ----------
        arguments: list-workRequest
            each element results corresponds to a separate server

        Returns
        -------
        list
            list of results
        """
        results = []
        for idx, argument in enumerate(arguments):
            self.inputQs[idx].put(argument)
        for queue in self.outputQs:
            try:
                result = queue.get(timeout=TIMEOUT)
            except Exception as err:
                self.logger.error("Timeout in %s" % self.name, err)
                result = None
            results.append(result)
        return results

    def stop(self):
        """
        Cleans up the queues and terminates all additional processes.
        """
        _ = [s.terminate() for s in self.servers]
        #_ = [s.join() for s in self.servers]
        _ = [q.close for q in self.inputQs]
        _ = [q.close for q in self.outputQs]
