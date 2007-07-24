#!/usr/bin/env python
###########################################################################
#
# UndoRedoStack implements the usual undo/redo capabilities of a GUI
#
# Author: Orest Shardt
# Created: July 23, 2007
#
###########################################################################

class action:
  def __init__(self,act,inv):
    self.redo = act
    self.undo = inv

class actionStack:
  def __init__(self):
    self.clear()

  def add(self,action):
    self.undoStack.append(action)
    #print "Added",action
    self.redoStack = []

  def undo(self):
    if len(self.undoStack) > 0:
      op = self.undoStack.pop()
      self.redoStack.append(op)
      op.undo()
      #print "undid",op
    else:
      pass #print "nothing to undo"

  def redo(self):
    if len(self.redoStack) > 0:
      op = self.redoStack.pop()
      self.undoStack.append(op)
      op.redo()
      #print "redid",op
    else:
      pass #print "nothing to redo"

  def changesMade(self):
    if len(self.undoStack)>0:
      return True
    else:
      return False

  def clear(self):
    self.redoStack = []
    self.undoStack = []

if __name__=='__main__':
  import sys
  def opq():
    print "action1"
  def unopq():
    print "inverse1"
  q = action(opq,unopq)
  w = action(lambda:sys.stdout.write("action2\n"),lambda:sys.stdout.write("inverse2\n"))
  e = action(lambda:sys.stdout.write("action3\n"),lambda:sys.stdout.write("inverse3\n"))
  s = actionStack()
  s.add(q)
  s.add(w)
  s.add(e)
