"""
Adapations of pickle to make it more robust to changes in pickled objects.

Provides robustness by not serializing the methods of the target object, only
its __dict__. Deserialization involves: (a) create a default instantiation
of the object and (b) re-assigning its __dict__.

Objects for which this behavior is desired must inherit RPickler and
override the class method RPicklerConstructor(cls) as appropriate to
so that the method returns a default instance of the class.

"""

import os
import pickle


class Serialization(object):
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
                obj.__dict__[key] = value
        # Revise the obj as required
        obj.rpRevise()
        #
        return obj

    def __repr__(self):
        return "Serialization of %s" % str(self.cls)


class RPickler(object):
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
        pass


def dump(obj, fd):
    """
    Dumps the objects to a file.

    Parameters
    ----------
    obj: object to be serialized
    fd: file descriptor
    """
    # Construct the module name
    fsplits = os.path.split(__file__)
    module = os.path.splitext(fsplits[1])[0]
    project = os.path.split(fsplits[0])[1]
    full_module_name = "%s.%s" % (project, module)
    # Construct the serialization
    serialization = Serialization(obj)
    serialization.serialize()
    # Save and clear globals so not serialized
    global_dct = globals()
    save_global_dct = dict(global_dct)
    keys = list(global_dct.keys())
    # FIXME: Do we need to clean globals before serializing?
    if False:
        for key in keys:
            if (not "__" in key) and (key != full_module_name):
                value = global_dct[key]
                del global_dct[key]
    # Serialize
    pickle.dump(serialization, fd)
    # Restore globals
    # FIXME: Do we need to clean globals before serializing?
    if False:
        for key, value in global_dict:
            global_dct[key] = save_global_dct[key]

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

