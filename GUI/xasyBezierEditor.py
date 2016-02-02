#!/usr/bin/env python
###########################################################################
#
# xasyBezierEditor implements the ability to graphically edit the location
# of the nodes and control points of a bezier curve.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
###########################################################################

import math
import sys
from CubicBezier import *
import xasy2asy

if sys.version_info >= (3, 0):
  from tkinter import *
else:
  from Tkinter import *

class node:
  def __init__(self,precontrol,node,postcontrol,uid,isTied = True):
    self.node = node
    self.precontrol = precontrol
    self.postcontrol = postcontrol
    self.isTied = isTied
    self.uid = uid
    self.nodeID = self.precontrolID = self.prelineID = self.postcontrolID = self.postlineID = None

  def shiftNode(self,delta):
    self.node = (self.node[0]+delta[0],self.node[1]+delta[1])
    if self.precontrol != None:
      self.precontrol = (self.precontrol[0]+delta[0],self.precontrol[1]+delta[1])
    if self.postcontrol != None:
      self.postcontrol = (self.postcontrol[0]+delta[0],self.postcontrol[1]+delta[1])

  def shiftPrecontrol(self,delta):
    self.precontrol = (self.precontrol[0]+delta[0],self.precontrol[1]+delta[1])
    if self.isTied and self.postcontrol != None:
      self.rotatePostControl(self.precontrol)

  def shiftPostcontrol(self,delta):
    self.postcontrol = (self.postcontrol[0]+delta[0],self.postcontrol[1]+delta[1])
    if self.isTied and self.precontrol != None:
      self.rotatePrecontrol(self.postcontrol)

  def rotatePrecontrol(self,after):
    vx,vy = after[0]-self.node[0],after[1]-self.node[1]
    l = norm((vx,vy))
    if l == 0:
      return
    m = norm((self.precontrol[0]-self.node[0],self.precontrol[1]-self.node[1]))
    vx = -m*vx/l
    vy = -m*vy/l
    self.precontrol = self.node[0]+vx,self.node[1]+vy

  def rotatePostControl(self,after):
    vx,vy = after[0]-self.node[0],after[1]-self.node[1]
    l = norm((vx,vy))
    if l == 0:
      return
    m = norm((self.postcontrol[0]-self.node[0],self.postcontrol[1]-self.node[1]))
    vx = -m*vx/l
    vy = -m*vy/l
    self.postcontrol = self.node[0]+vx,self.node[1]+vy

  def draw(self,canvas):
    width = 3
    if self.precontrol != None:
      if self.prelineID == None:
        self.prelineID = canvas.create_line(self.precontrol[0],-self.precontrol[1],self.node[0],-self.node[1],tags=("preline",self.uid))
      else:
        canvas.coords(self.prelineID,self.precontrol[0],-self.precontrol[1],self.node[0],-self.node[1])
      if self.precontrolID == None:
        self.precontrolID = canvas.create_oval(self.precontrol[0]-width,-self.precontrol[1]-width,self.precontrol[0]+width,-self.precontrol[1]+width,
            fill="red",outline="black",tags=("precontrol",self.uid))
      else:
        canvas.coords(self.precontrolID,self.precontrol[0]-width,-self.precontrol[1]-width,self.precontrol[0]+width,-self.precontrol[1]+width)
    if self.postcontrol != None:
      if self.postlineID == None:
        self.postlineID = canvas.create_line(self.postcontrol[0],-self.postcontrol[1],self.node[0],-self.node[1],tags=("postline",self.uid))
      else:
        canvas.coords(self.postlineID,self.postcontrol[0],-self.postcontrol[1],self.node[0],-self.node[1])
      if self.postcontrolID == None:
        self.postcontrolID = canvas.create_oval(self.postcontrol[0]-width,-self.postcontrol[1]-width,self.postcontrol[0]+width,-self.postcontrol[1]+width,
            fill="red",outline="black",tags=("postcontrol",self.uid))
      else:
        canvas.coords(self.postcontrolID,self.postcontrol[0]-width,-self.postcontrol[1]-width,self.postcontrol[0]+width,-self.postcontrol[1]+width)
    if self.isTied:
      color = "blue"
    else:
      color = "green"
    if self.nodeID == None:
      self.nodeID = canvas.create_oval(self.node[0]-width,-self.node[1]-width,self.node[0]+width,-self.node[1]+width,
          fill=color,outline="black",tags=("node",self.uid))
    else:
      canvas.coords(self.nodeID,self.node[0]-width,-self.node[1]-width,self.node[0]+width,-self.node[1]+width)
      canvas.itemconfigure(self.nodeID,fill=color)

class xasyBezierEditor:
  def __init__(self,parent,shape,canvas):
    self.parent = parent
    self.shape = shape
    self.transform = self.shape.transform[0]
    self.path = self.shape.path
    self.canvas = canvas
    self.modified = False
    self.path.computeControls()
    isCyclic = self.path.nodeSet[-1] == 'cycle'
    segments = len(self.path.controlSet)
    self.nodeList = []
    for i in range(segments):
      if i == 0:
        node0 = self.transform*self.path.nodeSet[i]
        control = self.transform*self.path.controlSet[i][0]
        self.nodeList.append(node(None,node0,control,len(self.nodeList)))
      else:
        node0 = self.transform*self.path.nodeSet[i]
        precontrol = self.transform*self.path.controlSet[i-1][1]
        postcontrol = self.transform*self.path.controlSet[i][0]
        self.nodeList.append(node(precontrol,node0,postcontrol,len(self.nodeList)))
    if not isCyclic:
      node0 = self.transform*self.path.nodeSet[-1]
      precontrol = self.transform*self.path.controlSet[-1][1]
      self.nodeList.append(node(precontrol,node0,None,len(self.nodeList)))
    else:
      self.nodeList[0].precontrol = self.transform*self.path.controlSet[-1][1]
    self.showControls()
    self.bindNodeEvents()
    self.bindControlEvents()

  def showControls(self):
    for n in self.nodeList:
      n.draw(self.canvas)
    self.bindNodeEvents()
    self.bindControlEvents()
    self.parent.updateCanvasSize()

  def bindNodeEvents(self):
    self.canvas.tag_bind("node","<B1-Motion>",self.nodeDrag)
    self.canvas.tag_bind("node","<Button-1>",self.buttonDown)
    self.canvas.tag_bind("node","<Double-Button-1>",self.toggleNode)

  def unbindNodeEvents(self):
    self.canvas.tag_unbind("node","<B1-Motion>")
    self.canvas.tag_unbind("node","<Button-1>")
    self.canvas.tag_unbind("node","<Double-Button-1>")

  def bindControlEvents(self):
    self.canvas.tag_bind("precontrol || postcontrol","<B1-Motion>",self.controlDrag)
    self.canvas.tag_bind("precontrol || postcontrol","<Button-1>",self.buttonDown)

  def unbindControlEvents(self):
    self.canvas.tag_unbind("precontrol || postcontrol","<B1-Motion>")
    self.canvas.tag_unbind("precontrol || postcontrol","<Button-1>")

  def buttonDown(self,event):
    self.parent.freeMouseDown = False
    self.startx,self.starty = event.x,event.y

  def toggleNode(self,event):
    self.parent.freeMouseDown = False
    tags = self.canvas.gettags(CURRENT)
    obj = tags[0]
    uid = int(tags[1])
    self.nodeList[uid].isTied = not self.nodeList[uid].isTied
    self.showControls()

  def nodeDrag(self,event):
    self.parent.freeMouseDown = False
    deltax = event.x-self.startx
    deltay = event.y-self.starty
    tags = self.canvas.gettags(CURRENT)
    obj = tags[0]
    uid = int(tags[1])
    self.nodeList[uid].shiftNode((deltax,-deltay))
    self.startx,self.starty = event.x,event.y
    self.applyChanges()
    self.showControls()
    self.shape.drawOnCanvas(self.canvas,self.parent.magnification)

  def controlDrag(self,event):
    self.parent.freeMouseDown = False
    deltax = event.x-self.startx
    deltay = event.y-self.starty
    tags = self.canvas.gettags(CURRENT)
    obj = tags[0]
    uid = int(tags[1])
    if obj == "precontrol":
      self.nodeList[uid].shiftPrecontrol((deltax,-deltay))
    elif obj == "postcontrol":
      self.nodeList[uid].shiftPostcontrol((deltax,-deltay))
    self.startx,self.starty = event.x,event.y
    self.applyChanges()
    self.showControls()
    self.shape.drawOnCanvas(self.canvas,self.parent.magnification)

  def applyChanges(self):
    self.modified = True
    self.shape.transform[0] = xasy2asy.asyTransform((0,0,1,0,0,1))
    for i in range(len(self.nodeList)):
      self.path.nodeSet[i] = self.nodeList[i].node
      if self.nodeList[i].postcontrol != None:
        self.path.controlSet[i][0] = self.nodeList[i].postcontrol
      if self.nodeList[i].precontrol != None:
        self.path.controlSet[i-1][1] = self.nodeList[i].precontrol

  def endEdit(self):
    self.unbindNodeEvents()
    self.unbindControlEvents()
    self.canvas.delete("node || precontrol || postcontrol || preline || postline")
