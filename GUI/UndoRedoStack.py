#!/usr/bin/env python3
###########################################################################
#
# UndoRedoStack implements the usual undo/redo capabilities of a GUI
#
# Author: Orest Shardt
# Created: July 23, 2007
#
###########################################################################


class action:
    def __init__(self, actions):
        act, inv = actions
        self.act = act
        self.inv = inv

    def undo(self):
        # print ("Undo:",self)
        self.inv()

    def redo(self):
        # print ("Redo:",self)
        self.act()

    def __str__(self):
        return "A generic action"


class beginActionGroup:
    pass


class endActionGroup:
    pass


class actionStack:
    def __init__(self):
        self.clear()

    def add(self, action):
        self.undoStack.append(action)
        # print ("Added",action)
        self.redoStack = []

    def undo(self):
        if len(self.undoStack) > 0:
            op = self.undoStack.pop()
            if op is beginActionGroup:
                level = 1
                self.redoStack.append(endActionGroup)
                while level > 0:
                    op = self.undoStack.pop()
                    if op is endActionGroup:
                        level -= 1
                        self.redoStack.append(beginActionGroup)
                    elif op is beginActionGroup:
                        level += 1
                        self.redoStack.append(endActionGroup)
                    else:
                        op.undo()
                        self.redoStack.append(op)
            elif op is endActionGroup:
                raise Exception("endActionGroup without previous beginActionGroup")
            else:
                self.redoStack.append(op)
                op.undo()
                # print ("undid",op)
        else:
            pass  # print ("nothing to undo")

    def redo(self):
        if len(self.redoStack) > 0:
            op = self.redoStack.pop()
            if op is beginActionGroup:
                level = 1
                self.undoStack.append(endActionGroup)
                while level > 0:
                    op = self.redoStack.pop()
                    if op is endActionGroup:
                        level -= 1
                        self.undoStack.append(beginActionGroup)
                    elif op is beginActionGroup:
                        level += 1
                        self.undoStack.append(endActionGroup)
                    else:
                        op.redo()
                        self.undoStack.append(op)
            elif op is endActionGroup:
                raise Exception("endActionGroup without previous beginActionGroup")
            else:
                self.undoStack.append(op)
                op.redo()
                # print ("redid",op)
        else:
            pass  # print ("nothing to redo")

    def setCommitLevel(self):
        self.commitLevel = len(self.undoStack)

    def changesMade(self):
        if len(self.undoStack) != self.commitLevel:
            return True
        else:
            return False

    def clear(self):
        self.redoStack = []
        self.undoStack = []
        self.commitLevel = 0


if __name__ == '__main__':
    import sys


    def opq():
        print("action1")


    def unopq():
        print("inverse1")


    q = action(opq, unopq)
    w = action(lambda: sys.stdout.write("action2\n"), lambda: sys.stdout.write("inverse2\n"))
    e = action(lambda: sys.stdout.write("action3\n"), lambda: sys.stdout.write("inverse3\n"))
    s = actionStack()
    s.add(q)
    s.add(w)
    s.add(e)
