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
  def __init__(self,owner,itemList,indexList,translation):
    self.translation = translation
    self.owner = owner
    self.itemList = itemList
    self.indexList = indexList
    UndoRedoStack.action.__init__(self,self.transF,self.unTransF)

  def transF(self):
    for i in range(len(self.itemList)):
      for index in self.indexList[i]:
        self.owner.translateSomething(-1,(self.translation[0],self.translation[1]),self.itemList[i],index)
        if index==None:
          index = 0
        try:
          self.owner.mainCanvas.move(self.itemList[i].imageList[index].IDTag,self.translation[0],-self.translation[1])
        except:
          self.owner.mainCanvas.move(self.itemList[i].IDTag,self.translation[0],-self.translation[1])
    self.owner.updateSelection()
    self.owner.updateCanvasSize()

  def unTransF(self):
    for i in range(len(self.itemList)):
      for index in self.indexList[i]:
        self.owner.translateSomething(-1,(-self.translation[0],-self.translation[1]),self.itemList[i],index)
        try:
          self.owner.mainCanvas.move(self.itemList[i].imageList[index].IDTag,-self.translation[0],self.translation[1])
        except:
          self.owner.mainCanvas.move(self.itemList[i].IDTag,-self.translation[0],self.translation[1])
    self.owner.updateSelection()
    self.owner.updateCanvasSize()

  def __str__(self):
    return "translation of "+str(self.itemList)+str(self.indexList)+" by "+str(self.translation)

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

class addLabelAction(UndoRedoStack.action):
  def __init__(self,owner,label):
    self.owner = owner
    self.label = label
    UndoRedoStack.action.__init__(self,self.addF,self.unAddF)

  def addF(self):
    self.owner.addItemToFile(self.label)
    self.label.drawOnCanvas(self.owner.mainCanvas)
    self.owner.bindItemEvents(self.label)

  def unAddF(self):
    self.label.removeFromCanvas()
    del self.owner.fileItems[-1]
    self.owner.propList.delete(0)
    self.owner.clearSelection()

  def __str__(self):
    return "addition of a label"

class deleteLabelAction(UndoRedoStack.action):
  def __init__(self,owner,label,index):
    self.owner = owner
    self.label = label
    self.index = index
    UndoRedoStack.action.__init__(self,self.delF,self.unDelF)

  def delF(self):
    self.owner.fileItems[self.index].removeFromCanvas()
    self.owner.propList.delete(len(self.owner.fileItems)-self.index-1)
    del self.owner.fileItems[self.index]

  def unDelF(self):
    self.owner.fileItems.insert(self.index,self.label)
    self.owner.fileItems[self.index].drawOnCanvas(self.owner.mainCanvas)
    self.owner.propList.insert(len(self.owner.fileItems)-self.index-1,self.owner.describeItem(self.label))
    self.owner.bindItemEvents(self.label)

class editLabelAction(UndoRedoStack.action):
  def __init__(self,owner,label,newText,oldText):
    self.owner = owner
    self.label = label
    self.newText = newText
    self.oldText = oldText
    UndoRedoStack.action.__init__(self,self.modT,self.unModT)

  def modT(self):
    self.label.label.setText(self.newText)
    self.label.drawOnCanvas(self.owner.mainCanvas)
    self.owner.bindItemEvents(self.label)

  def unModT(self):
    self.label.label.setText(self.oldText)
    self.label.drawOnCanvas(self.owner.mainCanvas)
    self.owner.bindItemEvents(self.label)
