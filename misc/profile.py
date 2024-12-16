#!/usr/bin/env python3

import sys
from pprint import pprint

# Unused line numbers required by kcachegrind.
POS = '1'

def nameFromNode(tree):
    name = tree['name']
    pos = tree['pos']
    if pos.endswith(": "):
        pos = pos[:-2]
    return (name, pos)

def addFuncNames(tree, s):
    s.add(nameFromNode(tree))
    for child in tree['children']:
        addFuncNames(child, s)

def funcNames(tree):
    s = set()
    addFuncNames(tree, s)
    return s

def computeTotals(tree):
    for child in tree['children']:
        computeTotals(child)
    tree['instTotal'] = (tree['instructions']
                     + sum(child['instTotal'] for child in tree['children']))
    tree['nsecsTotal'] = (tree['nsecs']
                     + sum(child['nsecsTotal'] for child in tree['children']))

def printName(name, prefix=''):
    print (prefix+"fl=", name[1])
    print (prefix+"fn=", name[0])

class Arc:
    def __init__(self):
        self.calls = 0
        self.instTotal = 0
        self.nsecsTotal = 0

    def add(self, tree):
        self.calls += tree['calls']
        self.instTotal += tree['instTotal']
        self.nsecsTotal += tree['nsecsTotal']

class Func:
    def __init__(self):
        self.instructions = 0
        self.nsecs = 0
        self.arcs = {}

    def addChildTime(self, tree):
        arc = self.arcs.setdefault(nameFromNode(tree), Arc())
        arc.add(tree)

    def analyse(self, tree):
        self.instructions += tree['instructions']
        self.nsecs += tree['nsecs']
        for child in tree['children']:
            self.addChildTime(child)

    def dump(self):
        print (POS, self.instructions, self.nsecs)
        for name in self.arcs:
            printName(name, prefix='c')
            arc = self.arcs[name]
            print ("calls="+str(arc.calls), POS)
            print (POS, arc.instTotal, arc.nsecsTotal)
        print ()

def analyse(funcs, tree):
    funcs[nameFromNode(tree)].analyse(tree)
    for child in tree['children']:
        analyse(funcs, child)

def dump(funcs):
    print ("events: Instructions Nanoseconds")
    for name in funcs:
        printName(name)
        funcs[name].dump()

rawdata = __import__("asyprof")
profile = rawdata.profile

computeTotals(profile)
names = funcNames(profile)

funcs = {}
for name in names:
    funcs[name] = Func()

analyse(funcs, profile)
dump(funcs)

#pprint(names)
