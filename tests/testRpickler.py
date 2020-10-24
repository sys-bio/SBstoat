"""
Created on Oct 22, 2020

@author: joseph-hellerstein
"""

from SBstoat import rpickle

import copy
import os
import pickle
import unittest


IGNORE_TEST = False
IS_PLOT = False
DIR = os.path.dirname(os.path.abspath(__file__))
FILE_SERIALIZE = os.path.join(DIR, "testRpickler.pcl")
FILE_SERIALIZE2 = os.path.join(DIR, "testRpickler2.pcl")
FILES = [FILE_SERIALIZE, FILE_SERIALIZE2]
LIST = list(range(10))
A_VALUE = 10
B_VALUE = 100
C_VALUE = 500
# Crude way to test that can evolve a class and rpickle works
# Run the tests first with IS_SERIALIZE = True and then with IS_SERIALIZE = False
IS_TEST_PICKLE = False # Set to True if doing these tests
IS_SERIALIZE = False


############ SUPPORT CLASSES ################
def equals(obj1, obj2):
   # Works if objects only have basic types
   diff_keys = set(obj1.__dict__.keys()).symmetric_difference(
          obj2.__dict__.keys())
   if len(diff_keys) > 0:
       return False
   result = True
   for key in obj1.__dict__.keys():
       result = result and  \
             obj1.__getattribute__(key) == obj2.__getattribute__(key)
   return result


class DClassNoarg(rpickle.RPickler):

    def __init__(self):
        self.a = A_VALUE
        self.b = B_VALUE
        self.list = LIST

    def equals(self, other):
       return(self, other)


class DClassOnearg(rpickle.RPickler):

    def __init__(self, c):
        self.c = c
        self.d = DClassNoarg()

    @classmethod
    def rpConstruct(cls):
        return cls(C_VALUE)

    def equals(self, other):
       return(self, other)


class DClassRevise(rpickle.RPickler):

    def __init__(self, c):
        self.c = c
        self.d = DClassNoarg()

    @classmethod
    def rpConstruct(cls):
        return cls(None)

    def rpRevise(self):
        self.a = A_VALUE

    def equals(self, other):
       return(self, other)


if IS_SERIALIZE:
    class _DClass(rpickle.RPickler):
    
        def __init__(self):
            self.a = A_VALUE
    
        def equals(self, other):
           return(self, other)
else:
    class _DClass(rpickle.RPickler):
    
        def __init__(self, a):
            self.a = a
            self.b = B_VALUE
            self.list = LIST
    
        @classmethod
        def rpConstruct(cls):
            return cls(None)

        def get(self):
            return self.b
    
        def equals(self, other):
           return(self, other)
        

#####################################
class TestRPickler(unittest.TestCase):

    def setUp(self):
        self.cls_noarg = DClassNoarg
        self.cls_onearg = DClassOnearg
        self.cls_revise = DClassRevise
        self.rpickler = rpickle.RPickler()

    def testRpConstruct1(self):
        if IGNORE_TEST:
            return
        self.assertTrue(isinstance(self.rpickler.rpConstruct(), rpickle.RPickler))

    def testRpConstruct2(self):
        if IGNORE_TEST:
            return
        for cls in [self.cls_noarg, self.cls_onearg]:
            obj = cls.rpConstruct()
            true  = isinstance(obj, cls)
            self.assertTrue(true)

    def testRpRevise(self):
        if IGNORE_TEST:
            return
        obj = self.cls_revise(C_VALUE)
        self.assertFalse("a" in obj.__dict__.keys())
        obj.rpRevise()
        self.assertTrue("a" in obj.__dict__.keys())
        self.assertEqual(obj.a, A_VALUE)
    

#####################################
class TestSerialization(unittest.TestCase):

    def setUp(self):
        self.noarg = DClassNoarg()
        self.onearg = DClassOnearg(C_VALUE)
        self.revise = DClassRevise(C_VALUE)
        self.objs = [self.noarg, self.onearg, self.revise]

    def testConstructor(self):
        if IGNORE_TEST:
            return
        for obj in self.objs:
            serialization = rpickle.Serialization(obj)
            self.assertEqual(obj.__class__, serialization.cls)
            trues = [obj.__dict__[k] == serialization.dct[k]
                  for k in serialization.dct.keys()]
            self.assertTrue(all(trues))

    def testSerialize(self):
        if IGNORE_TEST:
            return
        def test(obj, hasSerialization):
            serialization = rpickle.Serialization(obj)
            serialization.serialize()
            values = [v for v in serialization.dct.values()]
            isPresent = any([isinstance(v, rpickle.Serialization) for v in values])
            if hasSerialization:
                self.assertTrue(isPresent)
            else:
                self.assertFalse(isPresent)
        #
        test(self.noarg, False)
        test(self.onearg, True)

    def testDeserialize(self):
        if IGNORE_TEST:
            return
        for obj in self.objs:
            serialization = rpickle.Serialization(obj)
            serialization.serialize()
            new_obj = serialization.deserialize()
            self.assertTrue(obj.equals(new_obj))


#####################################
class TestFunctions(unittest.TestCase):

    def setUp(self):
        self._remove()
        self.cls_noarg = copy.deepcopy(DClassNoarg)
        self.cls_onearg = copy.deepcopy(DClassOnearg)
        self.cls_revise = copy.deepcopy(DClassRevise)
  
    def tearDown(self):
        self._remove()

    def _remove(self):
        if IS_TEST_PICKLE:
            return
        for ffile in FILES:
            if os.path.isfile(ffile):
                os.remove(ffile)

    def dump(self, obj):
        with (open(FILE_SERIALIZE, "wb")) as fd:
            rpickle.dump(obj, fd)

    def testDump(self):
        if IGNORE_TEST:
            return
        def test(obj):
            self.dump(obj)
            self.assertTrue(os.path.isfile(FILE_SERIALIZE))
        #
        test(self.cls_noarg())
        test(self.cls_onearg(C_VALUE))
        test(self.cls_revise(C_VALUE))

    def testLoad(self):
        if IGNORE_TEST:
            return
        def test(obj):
            self.dump(obj)
            with (open(FILE_SERIALIZE, "rb")) as fd:
                new_obj = rpickle.load(fd)
            self.assertTrue(new_obj.equals(obj))
        #
        test(self.cls_noarg())
        test(self.cls_onearg(C_VALUE))
        test(self.cls_revise(C_VALUE))

    def testPickleFails(self):
        if IGNORE_TEST:
            return
        if not IS_TEST_PICKLE:
            return
        if IS_SERIALIZE:
            obj = _DClass()
            with (open(FILE_SERIALIZE, "wb")) as fd:
                pickle.dump(obj, fd)
        else:
            with (open(FILE_SERIALIZE, "rb")) as fd:
                new_obj = pickle.load(fd)
            try:
                _ = new_obj.get()
                self.assertTrue(False)  # Should not get here fail
            except:
                self.assertTrue(True)  # Should get here

    def testRPickleSucceeds(self):
        if IGNORE_TEST:
            return
        if not IS_TEST_PICKLE:
            return
        if IS_SERIALIZE:
            obj = _DClass()
            with (open(FILE_SERIALIZE2, "wb")) as fd:
                rpickle.dump(obj, fd)
        else:
            with (open(FILE_SERIALIZE2, "rb")) as fd:
                new_obj = rpickle.load(fd)
            try:
                _ = new_obj.get()
                self.assertTrue(True)  # Should get here
            except:
                self.assertTrue(False)  # Should not fail


if __name__ == '__main__':
    unittest.main()
