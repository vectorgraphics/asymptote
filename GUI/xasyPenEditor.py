#!/usr/bin/env python
#
# Author: Orest Shardt
# Created: June 29, 2007
#

from Tkinter import *
import tkMessageBox
from xasy2asy import *

class xasyPenEditor:
  """A dialog for getting the properties of an asyPen"""
  def __init__(self,master,callback,pen=asyPen()):
    """Initialize the dialog with a parent, a callback, and an initial pen"""
    master.title("Xasy Pen Editor")
    self.parent = master
    self.pen = pen
    self.callback = callback
    self.props = {"Color":pen.color,"Width":pen.width,"Options":pen.options}
    Label(text="Edit Pen Properties").grid(row=0,column=0,columnspan=4)

    self.texts = {}
    rowCount = 1
    for key in self.pen.__dict__.keys():
      Label(self.parent,text=key).grid(row=rowCount,column=0,sticky=E)
      self.texts[key]=Entry(self.parent)
      self.texts[key].insert(0,str(pen.__dict__[key]))
      self.texts[key].grid(row=rowCount,column=1)
      rowCount += 1

    f = Frame(self.parent)
    okButton = Button(f,text="OK",command=self.okCmd)
    okButton.grid(row=0,column=0)
    applyButton = Button(f,text="Apply",command=self.applyCmd)
    applyButton.grid(row=0,column=1)
    cancelButton = Button(f,text="Cancel",command=self.cancelCmd)
    cancelButton.grid(row=0,column=2)
    f.grid(row=10,column=1,rowspan=3)

  def applyCmd(self):
    """Respond to the user clicking the applyButton"""
    for key in self.pen.__dict__.keys():
      if type(self.pen.__dict__[key]) == type(0.5):
        exec("self.pen.%s = %f"%(key,float(self.texts[key].get())))
      else:
        exec("self.pen.%s = \"%s\""%(key,self.texts[key].get()))
    self.callback(self.pen)
  def okCmd(self):
    """Respond to the user clicking the okButton"""
    self.callback(self.pen)
  def cancelCmd(self):
    """Respond to the user clicking the cancelButton"""
    self.parent.destroy() 

def defCallback(a):
  print a.__dict__

if __name__ == '__main__':
  #run a test
  root=Tk()
  pe = xasyPenEditor(root,defCallback)
  root.mainloop()
  asy.quit()
