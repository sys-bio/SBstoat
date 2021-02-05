"""
Adapations of pickle to make it more robust to changes in pickled objects.

Provides robustness by not serializing the methods of the target object, only
its __dict__. Deserialization involves: (a) create a default instantiation
of the object and (b) re-assigning its __dict__.

Objects for which this behavior is desired must inherit RPickler and
override the class method RPicklerConstructor(cls) as appropriate to
so that the method returns a default instance of the class.

"""

import copy
import os
import pickle


class Serialization():
    """RPickler serialization of an object and its sub-objects."""

    def __init__(self, obj):
        """
        Parameters
        ----------
        obj: Object being serialized
        """
        self.cls = obj.__class__  # Class being serialized
        self.dct = dict(obj.__dict__)  # __dict__ for the instance

    def serialize(self):
        """
        Recursively constructs the serialization of the object.
        """
        for key, value in self.dct.items():
            if issubclass(value.__class__, RPickler):
                self.dct[key] = Serialization(value)
                self.dct[key].serialize()

    def deserialize(self):
        """
        Recursively deserializes objects.

        Returns
        -------
        self.cls
        """
        obj = self.cls.rpConstruct()
        # Recursively instantiate serialized objects.
        # Save as instances of the constructed object.
        for key, value in self.dct.items():
            if isinstance(value, Serialization):
                obj.__dict__[key] = value.deserialize()
            else:
                obj.__dict__[key] = copy.deepcopy(value)
        # Revise the obj as required
        obj.rpRevise()
        #
        return obj

    def __repr__(self):
        return "Serialization of %s" % str(self.cls)


class RPickler():
    # Used by classes that implement robust pickling.

    @classmethod
    def rpConstruct(cls):
        """
        Provides a default construction of an object.

        Returns
        -------
        Instance of cls
        """
        return cls()

    def rpRevise(self):
        """
        Provides a hook to modify instance variables after they have
        been initialized by RPickle.
        """


def dump(obj, fd):
    """
    Dumps the objects to a file.

    Parameters
    ----------
    obj: object to be serialized
    fd: file descriptor
    """
    # Construct the serialization
    serialization = Serialization(obj)
    serialization.serialize()
    # Serialize
    pickle.dump(serialization, fd)

def load(fd):
    """
    Restores a serialized object.

    Parameters
    ----------
    fd: file descriptor

    Returns
    -------
    object
    """
    serialization = pickle.load(fd)
    return serialization.deserialize()
