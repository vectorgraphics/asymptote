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
import sys
import UndoRedoStack
import xasy2asy

from tkinter import *


class translationAction(UndoRedoStack.action):
    def __init__(self, owner, itemList, indexList, translation):
        self.translation = translation
        self.owner = owner
        self.itemList = itemList
        self.indexList = indexList
        UndoRedoStack.action.__init__(self, self.transF, self.unTransF)

    def transF(self):
        mag = self.owner.magnification
        for i in range(len(self.itemList)):
            for index in self.indexList[i]:
                self.owner.translateSomething(-1, (self.translation[0] / mag, self.translation[1] / mag),
                                              self.itemList[i], index)
                if index == None:
                    index = 0
                try:
                    self.owner.mainCanvas.move(self.itemList[i].imageList[index].IDTag, self.translation[0] * mag,
                                               -self.translation[1] * mag)
                except:
                    self.owner.mainCanvas.move(self.itemList[i].IDTag, self.translation[0] * mag,
                                               -self.translation[1] * mag)
        self.owner.updateSelection()
        self.owner.updateCanvasSize()

    def unTransF(self):
        mag = self.owner.magnification
        for i in range(len(self.itemList)):
            for index in self.indexList[i]:
                self.owner.translateSomething(-1, (-self.translation[0] / mag, -self.translation[1] / mag),
                                              self.itemList[i], index)
                try:
                    self.owner.mainCanvas.move(self.itemList[i].imageList[index].IDTag, -self.translation[0] * mag,
                                               self.translation[1] * mag)
                except:
                    self.owner.mainCanvas.move(self.itemList[i].IDTag, -self.translation[0] * mag,
                                               self.translation[1] * mag)
        self.owner.updateSelection()
        self.owner.updateCanvasSize()

    def __str__(self):
        return "Translation of " + str(self.itemList) + str(self.indexList) + " by " + str(self.translation)


class rotationAction(UndoRedoStack.action):
    def __init__(self, owner, itemList, indexList, angle, origin):
        self.owner = owner
        self.itemList = itemList
        self.indexList = indexList
        self.angle = angle
        self.origin = origin
        UndoRedoStack.action.__init__(self, self.rotF, self.unRotF)

    def rotF(self):
        for i in range(len(self.itemList)):
            for index in self.indexList[i]:
                self.owner.rotateSomething(-1, self.angle, self.origin, self.itemList[i], index)
        for item in self.itemList:
            item.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
            self.owner.bindItemEvents(item)
        self.owner.updateSelection()
        self.owner.updateCanvasSize()

    def unRotF(self):
        for i in range(len(self.itemList)):
            for index in self.indexList[i]:
                self.owner.rotateSomething(-1, -self.angle, self.origin, self.itemList[i], index)
        for item in self.itemList:
            item.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
            self.owner.bindItemEvents(item)
        self.owner.updateSelection()
        self.owner.updateCanvasSize()

    def __str__(self):
        return "Rotation of " + str(self.itemList) + str(self.indexList) + " by " + "%.3f" % (
        self.angle * 180.0 / math.pi) + " about " + str(self.origin)


class addLabelAction(UndoRedoStack.action):
    def __init__(self, owner, label):
        self.owner = owner
        self.label = label
        UndoRedoStack.action.__init__(self, self.addF, self.unAddF)

    def addF(self):
        self.owner.addItemToFile(self.label)
        self.label.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.bindItemEvents(self.label)

    def unAddF(self):
        self.label.removeFromCanvas()
        del self.owner.fileItems[-1]
        self.owner.propList.delete(0)
        self.owner.clearSelection()

    def __str__(self):
        return "Addition of a label"


class deleteLabelAction(UndoRedoStack.action):
    def __init__(self, owner, label, index):
        self.owner = owner
        self.label = label
        self.index = index
        UndoRedoStack.action.__init__(self, self.delF, self.unDelF)

    def delF(self):
        self.owner.fileItems[self.index].removeFromCanvas()
        self.owner.propList.delete(len(self.owner.fileItems) - self.index - 1)
        del self.owner.fileItems[self.index]

    def unDelF(self):
        self.owner.fileItems.insert(self.index, self.label)
        self.owner.fileItems[self.index].drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.propList.insert(len(self.owner.fileItems) - self.index - 1, self.owner.describeItem(self.label))
        self.owner.bindItemEvents(self.label)

    def __str__(self):
        return "Deletion of a label"


class editLabelTextAction(UndoRedoStack.action):
    def __init__(self, owner, label, newText, oldText):
        self.owner = owner
        self.label = label
        self.newText = newText
        self.oldText = oldText
        UndoRedoStack.action.__init__(self, self.modT, self.unModT)

    def modT(self):
        self.label.label.setText(self.newText)
        self.label.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.bindItemEvents(self.label)

    def unModT(self):
        self.label.label.setText(self.oldText)
        self.label.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.bindItemEvents(self.label)

    def __str__(self):
        return "Editing a label's text"


class editLabelPenAction(UndoRedoStack.action):
    def __init__(self, owner, oldPen, newPen, index):
        self.owner = owner
        self.newPen = newPen
        self.oldPen = oldPen
        self.index = index
        UndoRedoStack.action.__init__(self, self.editF, self.unEditF)

    def editF(self):
        self.owner.fileItems[self.index].removeFromCanvas()
        self.owner.fileItems[self.index].label.pen = self.newPen
        self.owner.fileItems[self.index].drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.bindItemEvents(self.owner.fileItems[self.index])

    def unEditF(self):
        self.owner.fileItems[self.index].removeFromCanvas()
        self.owner.fileItems[self.index].label.pen = self.oldPen
        self.owner.fileItems[self.index].drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.bindItemEvents(self.owner.fileItems[self.index])

    def __str__(self):
        return "Changing a label's pen"


class addScriptAction(UndoRedoStack.action):
    def __init__(self, owner, script):
        self.owner = owner
        self.script = script
        UndoRedoStack.action.__init__(self, self.addF, self.unAddF)

    def addF(self):
        self.owner.addItemToFile(self.script)
        self.script.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.bindItemEvents(self.script)

    def unAddF(self):
        self.script.removeFromCanvas()
        del self.owner.fileItems[-1]
        self.owner.propList.delete(0)
        self.owner.clearSelection()

    def __str__(self):
        return "Addition of a script"


class deleteScriptAction(UndoRedoStack.action):
    def __init__(self, owner, script, index):
        self.owner = owner
        self.script = script
        self.index = index
        UndoRedoStack.action.__init__(self, self.delF, self.unDelF)

    def delF(self):
        self.owner.fileItems[self.index].removeFromCanvas()
        self.owner.propList.delete(len(self.owner.fileItems) - self.index - 1)
        del self.owner.fileItems[self.index]

    def unDelF(self):
        self.owner.fileItems.insert(self.index, self.script)
        self.owner.fileItems[self.index].drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.propList.insert(len(self.owner.fileItems) - self.index - 1, self.owner.describeItem(self.script))
        self.owner.bindItemEvents(self.script)

    def __str__(self):
        return "Deletion of a script"


class deleteScriptItemAction(UndoRedoStack.action):
    def __init__(self, owner, script, indices, oldTransforms):
        self.owner = owner
        self.script = script
        self.indices = indices[:]
        UndoRedoStack.action.__init__(self, self.delI, self.unDelI)

    def delI(self):
        for index in self.indices:
            self.script.transform[index].deleted = True
            self.owner.mainCanvas.delete(self.script.imageList[index].IDTag)

    def unDelI(self):
        for i in range(len(self.indices)):
            index = self.indices[i]
            self.script.transform[index].deleted = False
            bbox = self.script.imageList[index].originalImage.bbox
            self.script.imageList[index].IDTag = self.owner.mainCanvas.create_image(bbox[0], -bbox[3], anchor=NW,
                                                                                    tags=("image"),
                                                                                    image=self.script.imageList[
                                                                                        index].itk)
            self.owner.bindEvents(self.script.imageList[index].IDTag)
        self.owner.resetStacking()

    def __str__(self):
        return "Deletion of item " + str(self.indices) + " in " + str(self.script)


class editScriptAction(UndoRedoStack.action):
    def __init__(self, owner, script, newText, oldText):
        self.owner = owner
        self.script = script
        self.newText = newText
        self.oldText = oldText
        UndoRedoStack.action.__init__(self, self.modS, self.unModS)

    def modS(self):
        self.script.setScript(self.newText)
        self.script.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.bindItemEvents(self.script)

    def unModS(self):
        self.script.setScript(self.oldText)
        self.script.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)
        self.owner.bindItemEvents(self.script)

    def __str__(self):
        return "Modification of a script"


class clearItemTransformsAction(UndoRedoStack.action):
    def __init__(self, owner, item, oldTransforms):
        self.owner = owner
        self.item = item
        self.oldTransforms = oldTransforms
        UndoRedoStack.action.__init__(self, self.clearF, self.unClearF)

    def clearF(self):
        for i in range(len(self.oldTransforms)):
            self.item.transform[i] = xasy2asy.identity()
        self.item.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)

    def unClearF(self):
        for i in range(len(self.oldTransforms)):
            self.item.transform[i] = self.oldTransforms[i]
        self.item.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification)

    def __str__(self):
        return "Clear the transforms of " + str(self.item) + " from " + str(self.oldTransforms)


class itemRaiseAction(UndoRedoStack.action):
    def __init__(self, owner, items, oldPositions):
        self.owner = owner
        self.items = items[:]
        self.oldPositions = oldPositions[:]
        UndoRedoStack.action.__init__(self, self.raiseI, self.unRaiseI)

    def raiseI(self):
        for item in self.items:
            self.owner.raiseSomething(item)

    def unRaiseI(self):
        length = len(self.owner.fileItems)
        indices = self.oldPositions[:]
        indices = [length - i - 1 for i in indices]
        indices.reverse()
        for index in indices:
            for i in range(index):
                self.owner.raiseSomething(self.owner.fileItems[length - index - 1])

    def __str__(self):
        return "Raise items " + str(self.items) + " from positions " + str(self.oldPositions)


class itemLowerAction(UndoRedoStack.action):
    def __init__(self, owner, items, oldPositions):
        self.owner = owner
        self.items = items[:]
        self.oldPositions = oldPositions[:]
        UndoRedoStack.action.__init__(self, self.lowerI, self.unLowerI)

    def lowerI(self):
        for item in self.items:
            self.owner.lowerSomething(item)

    def unLowerI(self):
        indices = self.oldPositions[:]
        indices.reverse()
        for index in indices:
            for i in range(index):
                self.owner.lowerSomething(self.owner.fileItems[index])

    def __str__(self):
        return "Lower items " + str(self.items) + " from positions " + str(self.oldPositions)


class addDrawnItemAction(UndoRedoStack.action):
    def __init__(self, owner, item):
        self.owner = owner
        self.item = item
        UndoRedoStack.action.__init__(self, self.drawF, self.unDrawF)

    def drawF(self):
        self.owner.addItemToFile(self.item)
        self.item.drawOnCanvas(self.owner.mainCanvas, self.owner.magnification, forceAddition=True)
        self.owner.bindItemEvents(self.item)

    def unDrawF(self):
        self.item.removeFromCanvas(self.owner.mainCanvas)
        del self.owner.fileItems[-1]
        self.owner.propList.delete(0)
        self.owner.clearSelection()

    def __str__(self):
        return "Drawing of an item"


class deleteDrawnItemAction(UndoRedoStack.action):
    def __init__(self, owner, item, index):
        self.owner = owner
        self.item = item
        self.index = index
        UndoRedoStack.action.__init__(self, self.delF, self.unDelF)

    def delF(self):
        self.owner.fileItems[self.index].removeFromCanvas(self.owner.mainCanvas)
        self.owner.propList.delete(len(self.owner.fileItems) - self.index - 1)
        del self.owner.fileItems[self.index]

    def unDelF(self):
        self.owner.fileItems.insert(self.index, self.item)
        self.owner.fileItems[self.index].drawOnCanvas(self.owner.mainCanvas, self.owner.magnification,
                                                      forceAddition=True)
        self.owner.propList.insert(len(self.owner.fileItems) - self.index - 1, self.owner.describeItem(self.item))
        self.owner.bindItemEvents(self.item)

    def __str__(self):
        return "Deletion of a drawn item"


class editDrawnItemAction(UndoRedoStack.action):
    def __init__(self, owner, oldItem, newItem, index):
        self.owner = owner
        self.oldItem = oldItem
        self.newItem = newItem
        self.index = index
        UndoRedoStack.action.__init__(self, self.editF, self.unEditF)

    def editF(self):
        self.owner.fileItems[self.index].removeFromCanvas(self.owner.mainCanvas)
        self.owner.fileItems[self.index].path = self.newItem.path
        self.owner.fileItems[self.index].pen = self.newItem.pen
        self.owner.fileItems[self.index].transform = self.newItem.transform
        self.owner.fileItems[self.index].IDTag = self.newItem.IDTag
        self.owner.fileItems[self.index].drawOnCanvas(self.owner.mainCanvas, self.owner.magnification,
                                                      forceAddition=True)
        self.owner.bindItemEvents(self.owner.fileItems[self.index])

    def unEditF(self):
        self.owner.fileItems[self.index].removeFromCanvas(self.owner.mainCanvas)
        self.owner.fileItems[self.index].path = self.oldItem.path
        self.owner.fileItems[self.index].pen = self.oldItem.pen
        self.owner.fileItems[self.index].transform = self.oldItem.transform
        self.owner.fileItems[self.index].IDTag = self.oldItem.IDTag
        self.owner.fileItems[self.index].drawOnCanvas(self.owner.mainCanvas, self.owner.magnification,
                                                      forceAddition=True)
        self.owner.bindItemEvents(self.owner.fileItems[self.index])

    def __str__(self):
        return "Modification of a drawn item"
