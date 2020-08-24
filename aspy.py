#!/usr/bin/env python3
#####
# 
# aspy.py
#
# Andy Hammerlindl 2011/09/03
#
# Uses ctypes to interface with the shared library version of Python.
# Asymptote can run and its datatypes inspected via Python.
#
#
# To use the module:
# 1. make asymptote.so
# 2. Ensure that asymptote.so is visable to python, say adding its directory
#    to LD_LIBRARY_PATH
# 3. Run this module.  (See runExample for an example.)
#
#####

from ctypes import *

asyInt = c_longlong
handle_typ = c_void_p
arguments_typ = c_void_p
state_typ = c_void_p

function_typ = CFUNCTYPE(None, state_typ, c_void_p)

class string_typ(Structure):
    _fields_ = [
            ("buf", c_char_p), # Should be NUL-terminated? Maybe replace with
                               # POINTER(c_char).
            ("length", asyInt)
            ]

ErrorCallbackFUNC = CFUNCTYPE(None, string_typ)

NORMAL_ARG = 45000
REST_ARG = 45001

class Policy(Structure):
    _fields_ = [
            ("version",            asyInt),
            ("copyHandle",         CFUNCTYPE(handle_typ, handle_typ)),
            ("releaseHandle",      CFUNCTYPE(None, handle_typ)),
            ("handleFromInt",      CFUNCTYPE(handle_typ, asyInt)),
            ("handleFromBool",      CFUNCTYPE(handle_typ, asyInt)),
            ("handleFromDouble",      CFUNCTYPE(handle_typ, c_double)),
            ("handleFromString",   CFUNCTYPE(handle_typ, string_typ)),
            ("handleFromFunction", CFUNCTYPE(handle_typ,
                                             c_char_p,
                                             function_typ,
                                             c_void_p)),
            ("IntFromHandle",      CFUNCTYPE(asyInt, handle_typ)),
            ("boolFromHandle",     CFUNCTYPE(asyInt, handle_typ)),
            ("doubleFromHandle",   CFUNCTYPE(c_double, handle_typ)),
            ("stringFromHandle",   CFUNCTYPE(string_typ, handle_typ)),
            ("getField",           CFUNCTYPE(handle_typ,
                                             handle_typ,
                                             c_char_p)),
            ("getCell",            CFUNCTYPE(handle_typ,
                                             handle_typ,
                                             asyInt)),
            ("addField",           CFUNCTYPE(None,
                                             handle_typ,
                                             c_char_p,
                                             handle_typ)),
            ("newArguments",       CFUNCTYPE(arguments_typ)),
            ("releaseArguments",   CFUNCTYPE(None, arguments_typ)),
            ("addArgument",        CFUNCTYPE(None,
                                             arguments_typ,
                                             c_char_p,
                                             handle_typ,
                                             asyInt)),
            ("call",               CFUNCTYPE(handle_typ,
                                             handle_typ,
                                             arguments_typ)),
            ("globals",            CFUNCTYPE(handle_typ, state_typ)),
            ("numParams",          CFUNCTYPE(asyInt, state_typ)),
            ("getParam",           CFUNCTYPE(handle_typ, state_typ, asyInt)),
            ("setReturnValue",     CFUNCTYPE(None, state_typ, handle_typ)),
            ("setErrorCallback",   CFUNCTYPE(None, ErrorCallbackFUNC)),
            ]

policy = None
baseState = None
def initPolicyAndBaseState():
    global policy, baseState
    lib = CDLL("asymptote.so")

    getPolicy = lib._asy_getPolicy
    getPolicy.restype = POINTER(Policy)
    policy = getPolicy()

    getState = lib._asy_getState
    getState.restype = state_typ
    baseState = getState()

initPolicyAndBaseState()

def pyStringFromAsyString(st):
    #TODO: Handle strings with null-terminators.
    return str(st.buf)

def pyStringFromHandle(h):
    #TODO: Handle strings with null-terminators.
    st = policy.contents.stringFromHandle(h)
    checkForErrors()
    return pyStringFromAsyString(st)

def handleFromPyString(s):
    st = string_typ(s, len(s))
    h = policy.contents.handleFromString(st)
    checkForErrors()
    return h

def ensureDatum(val):
    return val if type(val) is Datum else Datum(val)

# The error detection scheme.
# policyError is set to a string when an error occurs.
policyError = []
def pyErrorCallback(s):
    global policyError
    policyError.append(pyStringFromAsyString(s))

cErrorCallback = ErrorCallbackFUNC(pyErrorCallback)
policy.contents.setErrorCallback(cErrorCallback)

class AsyException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def checkForErrors():
    """Raises an exception if an error occured."""
    global policyError
    if policyError != []:
        s = policyError[0]
        if len(policyError) > 1:
            s += ' (and other errors)'
        policyError = []
        raise AsyException(s)

class Datum(object):

    def _setHandle(self, handle):
        object.__setattr__(self, 'handle', handle)

    def __init__(self, val):
        self._setHandle(0)

        if val is None:
            return

        if type(val) is int:
            self._setHandle(policy.contents.handleFromInt(val))
            checkForErrors()
        elif type(val) is bool:
            self._setHandle(policy.contents.handleFromBool(1 if val else 0))
            checkForErrors()
        elif type(val) is float:
            self._setHandle(policy.contents.handleFromDouble(val))
        elif type(val) is str:
            self._setHandle(handleFromPyString(val))
            checkForErrors()
        elif type(val) is tuple:
            # Could do this more efficiently, and avoid a copyHandle
            ret = state.globals()["operator tuple"](*val)
            self._setHandle(policy.contents.copyHandle(ret.handle))
            checkForErrors()
        elif type(val) is Datum:
            self._setHandle(policy.contents.copyHandle(val.handle))
            checkForErrors()
        else:
            # TODO: check if val has a toDatum field
            raise TypeError("cannot initialize Datum from '%s'" %
                    type(val).__name__)

    def __repr__(self):
        # TODO: Add type-checking to policy.
        return '<Datum with handle %s>' % hex(self.handle)

    def __int__(self):
        l = policy.contents.IntFromHandle(self.handle)
        checkForErrors()
        return int(l)

    def __nonzero__(self):
        # This will throw an exception for anything but an underlying bool
        # type.  Perhaps we should be more pythonic.
        l = policy.contents.boolFromHandle(self.handle)
        checkForErrors()
        assert l in [0,1]
        return l == 1

    def __float__(self):
        x = policy.contents.doubleFromHandle(self.handle)
        checkForErrors()
        return float(x)

    def __str__(self):
        return pyStringFromHandle(self.handle)

    def __getattr__(self, name):
        field = policy.contents.getField(self.handle, name)
        checkForErrors()
        return DatumFromHandle(field)

    def __getitem__(self, name):
        assert type(name) == str
        return self.__getattr__(name)
        #TODO: raise an IndexError when appropriate.
        #TODO: implement array indices

    def __setattr__(self, name, val):
        # TODO: Resolve setting versus declaring.
        # One idea: d.x = f or d["x"] = f sets and d["int x()"] = f declares
        # anew.
        policy.contents.addField(self.handle,
                name, ensureDatum(val).handle)
        checkForErrors()

    def __setitem__(self, name, val):
        assert type(name) == str
        self.__setattr__(name, val)
        #TODO: raise an IndexError when appropriate.
        #TODO: implement array indices

    def __call__(self, *args, **namedArgs):
        alist = policy.contents.newArguments()
        checkForErrors()


        for arg in args:
            d = ensureDatum(arg)
            policy.contents.addArgument(alist, "", d.handle, NORMAL_ARG)
            checkForErrors()

        for name,arg in namedArgs.items():
            d = ensureDatum(arg)
            policy.contents.addArgument(alist, name, d.handle, NORMAL_ARG)
            checkForErrors()

        ret = policy.contents.call(self.handle, alist)
        checkForErrors()

        policy.contents.releaseArguments(alist)
        checkForErrors()

        if ret != None:
            return DatumFromHandle(ret)

    def __add__(self, other):
        return state.globals()["operator +"](self, other)
    def __sub__(self, other):
        return state.globals()["operator -"](self, other)
    def __mul__(self, other):
        return state.globals()["operator *"](self, other)
    def __div__(self, other):
        return state.globals()["operator /"](self, other)
    def __truediv__(self, other):
        return state.globals()["operator /"](self, other)
    def __mod__(self, other):
        return state.globals()["operator %"](self, other)
    def __pow__(self, other):
        return state.globals()["operator ^"](self, other)
    def __and__(self, other):
        return state.globals()["operator &"](self, other)
    def __or__(self, other):
        return state.globals()["operator |"](self, other)
    def __neg__(self, other):
        return state.globals()["operator -"](self)

    def __lt__(self, other):
        return state.globals()["operator <"](self, other)
    def __le__(self, other):
        return state.globals()["operator <="](self, other)
    def __eq__(self, other):
        return state.globals()["operator =="](self, other)
    def __ne__(self, other):
        return state.globals()["operator !="](self, other)
    def __gt__(self, other):
        return state.globals()["operator >"](self, other)
    def __ge__(self, other):
        return state.globals()["operator >="](self, other)

def DatumFromHandle(handle):
    """Initializes a Datum from a given low-level handle.  Does not invoke
    copyHandle."""
    d = Datum(None)
    d._setHandle(handle)
    return d

class State(object):
    def __init__(self, base):
        self.base = base

    def globals(self):
        handle = policy.contents.globals(self.base)
        checkForErrors()
        return DatumFromHandle(handle)

    def params(self):
        p = []

        numParams = policy.contents.numParams(self.base)
        checkForErrors()

        for i in range(numParams):
            h = policy.contents.getParam(self.base, i)
            checkForErrors()
            p.append(DatumFromHandle(h))

        assert len(p) == numParams
        return p

    def setReturnValue(self, val):
        policy.contents.setReturnValue(self.base, ensureDatum(val).handle)
        checkForErrors()

# Keep a link to all of the callbacks created, so they aren't garbage
# collected.  TODO: See if this is neccessary.
storedCallbacks = []

def DatumFromCallable(f):
    def wrapped(s, d):
        state = State(s)
        params = state.params()
        r = f(*params)
        if r != None:
            state.setReturnValue(r)

    cf = function_typ(wrapped)
    storedCallbacks.append(cf)

    h = policy.contents.handleFromFunction(f.__name__, cf, None)
    checkForErrors()

    return DatumFromHandle(h)

print ("version", policy.contents.version)

state = State(baseState)

# An example
def runExample():
    g = state.globals()

    g.eval("path p = (0,0) -- (100,100) -- (200,0)", embedded=True)
    g.draw(g.p)
    g.shipout("frompython")

    g.draw(g.circle(100), g.red)

