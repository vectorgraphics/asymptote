#!/usr/bin/env python
###########################################################################
#
# xasyOptionsDialog implements a dialog window to allow users to edit
# their preferences and specify program options
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
###########################################################################

from Tkinter import *
import xasyOptions
import tkSimpleDialog
import xasyColorPicker
import tkMessageBox
import tkFileDialog
import tkColorChooser
import os
import sys

class xasyOptionsDlg(tkSimpleDialog.Dialog):
  """A dialog to interact with users about their preferred settings"""
  def __init__(self,master=None):
    tkSimpleDialog.Dialog.__init__(self,master,"xasy Options")

  def body(self,master):
    optFrame = Frame(master)
    optFrame.grid(row=0,column=0,sticky=N+S+E+W)

    asyGrp = LabelFrame(optFrame,text="Asymptote",padx=5,pady=5)
    asyGrp.grid(row=0,column=0,sticky=E+W)
    asyGrp.rowconfigure(0,weight=1)
    asyGrp.rowconfigure(1,weight=1)
    asyGrp.columnconfigure(0,weight=1)
    asyGrp.columnconfigure(0,weight=2)
    Label(asyGrp,text="Command").grid(row=0,column=0,sticky=W)
    self.ap = Entry(asyGrp)
    self.ap.insert(END,xasyOptions.options['asyPath'])
    self.ap.grid(row=0,column=1,sticky=E+W)
    Button(asyGrp,text="...",command=self.findAsyPath).grid(row=0,column=2,sticky=E+W)
    self.showDebug = BooleanVar()
    self.showDebug.set(xasyOptions.options['showDebug'])
    self.sd = Checkbutton(asyGrp,text="Show debugging info in console",var=self.showDebug)
    self.sd.grid(row=1,column=0,columnspan=2,sticky=W)

    editGrp = LabelFrame(optFrame,text="External Editor",padx=5,pady=5)
    editGrp.grid(row=1,column=0,sticky=E+W)
    editGrp.rowconfigure(0,weight=1)
    editGrp.rowconfigure(1,weight=1)
    editGrp.columnconfigure(0,weight=1)
    editGrp.columnconfigure(0,weight=2)
    Label(editGrp,text="Program").grid(row=0,column=0,sticky=W)
    self.ee = Entry(editGrp)
    self.ee.insert(END,xasyOptions.options['externalEditor'])
    self.ee.grid(row=0,column=1,sticky=E+W)
    Button(editGrp,text="...",command=self.findEEPath).grid(row=0,column=2,sticky=E+W)

    penGrp = LabelFrame(optFrame,text="Default Pen",padx=5,pady=5)
    penGrp.grid(row=2,column=0,sticky=E+W)
    penGrp.rowconfigure(0,weight=1)
    penGrp.rowconfigure(1,weight=1)
    penGrp.rowconfigure(2,weight=1)
    penGrp.columnconfigure(1,weight=1)
    Label(penGrp,text="Color").grid(row=0,column=0,sticky=E)
    self.pc = xasyOptions.options['defPenColor']
    Button(penGrp,text="Change",command=self.changePenColor).grid(row=0,column=1,sticky=W)
    Label(penGrp,text="Width").grid(row=1,column=0,sticky=E)
    self.pw = Entry(penGrp)
    self.pw.insert(END,str(xasyOptions.options['defPenWidth']))
    self.pw.grid(row=1,column=1,sticky=E+W)
    Label(penGrp,text="Options").grid(row=2,column=0,sticky=E)
    self.po = Entry(penGrp)
    self.po.insert(END,xasyOptions.options['defPenOptions'])
    self.po.grid(row=2,column=1,sticky=E+W)

    dispGrp = LabelFrame(optFrame,text="Display Options",padx=5,pady=5)
    dispGrp.grid(row=3,column=0,sticky=E+W)
    dispGrp.rowconfigure(0,weight=1)
    dispGrp.rowconfigure(1,weight=1)
    dispGrp.rowconfigure(2,weight=1)
    dispGrp.rowconfigure(3,weight=1)
    dispGrp.columnconfigure(0,weight=1)
    dispGrp.columnconfigure(1,weight=1)
    dispGrp.columnconfigure(2,weight=1)
    self.showAxes = BooleanVar()
    self.showAxes.set(xasyOptions.options['showAxes'])
    self.sa = Checkbutton(dispGrp,text="Show Axes",var=self.showAxes)
    self.sa.grid(row=0,column=0,sticky=W)
    self.ac = xasyOptions.options['axesColor']
    Button(dispGrp,text="Color...",command=self.changeAxesColor).grid(row=1,column=0)
    Label(dispGrp,text="x").grid(row=0,column=1,padx=5,sticky=E)
    self.axs = Entry(dispGrp,width=6)
    self.axs.insert(END,xasyOptions.options['axisX'])
    self.axs.grid(row=0,column=2,sticky=W+E)
    Label(dispGrp,text="y").grid(row=1,column=1,padx=5,sticky=E)
    self.ays = Entry(dispGrp,width=6)
    self.ays.insert(END,xasyOptions.options['axisY'])
    self.ays.grid(row=1,column=2,sticky=W+E)

    self.showGrid = BooleanVar()
    self.showGrid.set(xasyOptions.options['showGrid'])
    self.sg = Checkbutton(dispGrp,text="Show Grid",var=self.showGrid)
    self.sg.grid(row=4,column=0,sticky=W)
    self.gc = xasyOptions.options['gridColor']
    Button(dispGrp,text="Color...",command=self.changeGridColor).grid(row=3,column=0)
    Label(dispGrp,text="x").grid(row=2,column=1,padx=5,sticky=E)
    self.gxs = Entry(dispGrp,width=6)
    self.gxs.insert(END,xasyOptions.options['gridX'])
    self.gxs.grid(row=2,column=2,sticky=W+E)
    Label(dispGrp,text="y").grid(row=3,column=1,padx=5,sticky=E)
    self.gys = Entry(dispGrp,width=6)
    self.gys.insert(END,xasyOptions.options['gridY'])
    self.gys.grid(row=3,column=2,sticky=W+E)

  def findEEPath(self):
    if sys.platform[:3] == 'win': #for windows, wince, win32, etc
      file=tkFileDialog.askopenfile(filetypes=[("Programs","*.exe"),("All files","*")],title="Choose External Editor",parent=self)
    else:
      file=tkFileDialog.askopenfile(filetypes=[("All files","*")],title="Choose External Editor",parent=self)
    if file != None:
      name = os.path.abspath(file.name)
      file.close()
      self.ee.delete(0,END)
      self.ee.insert(END,name)
      self.validate()

  def findAsyPath(self):
    if sys.platform[:3] == 'win': #for windows, wince, win32, etc
      file=tkFileDialog.askopenfile(filetypes=[("Programs","*.exe"),("All files","*")],title="Find Asymptote Executable",parent=self)
    else:
      file=tkFileDialog.askopenfile(filetypes=[("All files","*")],title="Find Asymptote Executable",parent=self)
    if file != None:
      name = os.path.abspath(file.name)
      file.close()
      self.ap.delete(0,END)
      self.ap.insert(END,name)
      self.validate()

  def getAColor(self,color):
    result = xasyColorPicker.xasyColorDlg(self).getColor(xasyColorPicker.makeRGBfromTkColor(color))
    return xasyColorPicker.RGB255hex(xasyColorPicker.RGBreal255(result))

  def changeAxesColor(self):
    self.ac = self.getAColor(self.ac)

  def changeGridColor(self):
    self.gc = self.getAColor(self.gc)

  def changePenColor(self):
    self.pc = self.getAColor(self.pc)

  def apply(self):
    xasyOptions.options['externalEditor'] = self.ee.get()
    xasyOptions.options['asyPath'] = self.ap.get()
    xasyOptions.options['showDebug'] = bool(self.showDebug.get())

    xasyOptions.options['defPenColor'] = self.pc
    xasyOptions.options['defPenWidth'] = float(self.pw.get())
    xasyOptions.options['defPenOptions'] = self.po.get()

    xasyOptions.options['showAxes'] = bool(self.showAxes.get())
    xasyOptions.options['axesColor'] = self.ac
    xasyOptions.options['tickColor'] = self.ac
    xasyOptions.options['axisX'] = int(self.axs.get())
    xasyOptions.options['axisY'] = int(self.ays.get())
    xasyOptions.options['showGrid'] = bool(self.showGrid.get())
    xasyOptions.options['gridColor'] = self.gc
    xasyOptions.options['gridX'] = int(self.gxs.get())
    xasyOptions.options['gridY'] = int(self.gys.get())
    xasyOptions.save()

  def validateAColor(self,color):
    hexdigits = '0123456789abcdef'
    if len(self.pc) != 7 or self.pc[0] != '#' or sum([1 for a in self.pc[1:] if a in hexdigits]) != 6:
      return False
    else:
      return True

  def validate(self):
    """Validate the data entered into the dialog"""
    #validate the color
    hexdigits = '0123456789abcdef'
    if not self.validateAColor(self.pc):
      tkMessageBox.showerror("xasy Options","Invalid pen color.\r\n"+self.pc,parent=self)
      return False
    #validate the width
    try:
      test = float(self.pw.get())
    except:
      tkMessageBox.showerror("xasy Options","Pen width must be a number.",parent=self)
      return False

    #validate the options
    #nothing to do

    #validate the axis spacing
    try:
      test = int(self.axs.get())
      test = int(self.ays.get())
    except:
      tkMessageBox.showerror("xasy Options","Axes' x- and y-spacing must be numbers.",parent=self)
      return False

    #validate the grid spacing
    try:
      test = int(self.gxs.get())
      test = int(self.gys.get())
    except:
      tkMessageBox.showerror("xasy Options","Grid's x- and y-spacing  must be numbers.",parent=self)
      return False

    if not self.validateAColor(self.ac):
      tkMessageBox.showerror("xasy Options","Invalid axis color.\r\n"+self.ac,parent=self)
      return False

    if not self.validateAColor(self.gc):
      tkMessageBox.showerror("xasy Options","Invalid grid color.\r\n"+self.gc,parent=self)
      return False

    return True

if __name__ == '__main__':
  root = Tk()
  xasyOptions.load()
  d = xasyOptionsDlg(root)
  print d.result
