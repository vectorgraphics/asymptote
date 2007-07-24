#!/usr/bin/env python
###########################################################################
#
# xasyActions implements the possible actions and their inverses
# for the undo/redo stack in xasy
#
# Author: Orest Shardt
# Created: July 23, 2007
#
###########################################################################
import math
import UndoRedoStack

class translationAction(UndoRedoStack.action):
  def __init__(self,owner,IDList,translation):
    self.translation = translation
    self.owner = owner
    self.IDList = IDList
    UndoRedoStack.action.__init__(self,self.transF,self.unTransF)

  def transF(self):
    for ID in self.IDList:
      self.owner.translateSomething(ID,self.translation)
      self.owner.mainCanvas.move(ID,self.translation[0],-self.translation[1])
      self.owner.updateSelection()
      self.owner.updateCanvasSize()

  def unTransF(self):
    for ID in self.IDList:
      self.owner.translateSomething(ID,(-self.translation[0],-self.translation[1]))
      self.owner.mainCanvas.move(ID,-self.translation[0],self.translation[1])
      self.owner.updateSelection()
      self.owner.updateCanvasSize()

  def __str__(self):
    return "translation of "+str(self.IDList)+" by "+str(self.translation)

class rotationAction(UndoRedoStack.action):
  def __init__(self,owner,itemList,indexList,angle,origin):
    self.owner = owner
    self.itemList = itemList
    self.indexList = indexList
    self.angle = angle
    self.origin = origin
    UndoRedoStack.action.__init__(self,self.rotF,self.unRotF)

  def rotF(self):
    for i in range(len(self.itemList)):
      for index in self.indexList[i]:
        self.owner.rotateSomething(-1,self.angle,self.origin,self.itemList[i],index)
    for item in self.itemList:
      item.drawOnCanvas(self.owner.mainCanvas)
      self.owner.bindItemEvents(item)
    self.owner.updateSelection()
    self.owner.updateCanvasSize()

  def unRotF(self):
    for i in range(len(self.itemList)):
      for index in self.indexList[i]:
        self.owner.rotateSomething(-1,-self.angle,self.origin,self.itemList[i],index)
    for item in self.itemList:
      item.drawOnCanvas(self.owner.mainCanvas)
      self.owner.bindItemEvents(item)
    self.owner.updateSelection()
    self.owner.updateCanvasSize()

  def __str__(self):
    return "rotation of "+str(self.itemList)+str(self.indexList)+" by "+"%.3f"%(self.angle*180.0/math.pi)+" about "+str(self.origin)
