#!/usr/bin/env python
###########################################################################
#
# xasyMainWin implements the functionality of the GUI. It depends on
# xasy2asy for its interaction with Asymptote.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
###########################################################################

import os
from string import *
import subprocess
import math
import copy

from Tkinter import *
import tkMessageBox
import tkFileDialog
import tkSimpleDialog
import threading
import time

from xasyVersion import xasyVersion
import xasyCodeEditor
from xasy2asy import *
import xasyFile
import xasyOptions
import xasyOptionsDialog
import CubicBezier
from xasyBezierEditor import xasyBezierEditor
from xasyGUIIcons import iconB64
from xasyColorPicker import *

from UndoRedoStack import *
from xasyActions import *

import string

try:
  import ImageTk
  import Image
  PILAvailable = True
except:
  PILAvailable = False

class xasyMainWin:
  def __init__(self,master,file=None,magnification=1.0):
    self.opLock = threading.Lock()
    self.parent = master
    self.magnification = magnification
    self.previousZoom = self.magnification
    self.magList = [0.1,0.25,1.0/3,0.5,1,2,3,4,5,10]
    self.bindGlobalEvents()
    self.createWidgets()
    self.resetGUI()
    if sys.platform[:3] == "win":
      site="http://effbot.org/downloads/PIL-1.1.7.win32-py2.7.exe"
    else:
      site="http://effbot.org/downloads/Imaging-1.1.7.tar.gz" 
    if not PILAvailable:
      tkMessageBox.showerror("Failed Dependencies","An error occurred loading the required PIL library. Please install "+site)
      self.parent.destroy()
      sys.exit(1)
    if file != None:
      self.loadFile(file)
    self.parent.after(100,self.tickHandler)

  def testOrAcquireLock(self):
    val = self.opLock.acquire(False)
    if val:
      self.closeDisplayLock()
    return val

  def acquireLock(self):
    self.closeDisplayLock()
    self.opLock.acquire()

  def releaseLock(self):
    self.opLock.release()
    self.openDisplayLock()

  def tickHandler(self):
    self.tickCount += 1
    self.mainCanvas.itemconfigure("outlineBox",dashoffset=self.tickCount%9)
    self.parent.after(100,self.tickHandler)

  def closeDisplayLock(self):
    self.status.config(text="Busy")
    self.parent.update_idletasks()

  def openDisplayLock(self):
    self.status.config(text="Ready")

  def bindGlobalEvents(self):
    #global bindings
    self.parent.bind_all("<Control-z>",lambda q:self.editUndoCmd())# z -> no shift
    self.parent.bind_all("<Control-Z>",lambda q:self.editRedoCmd())# Z -> with shift
    self.parent.bind_all("<Control-o>",lambda q:self.fileOpenCmd())
    self.parent.bind_all("<Control-n>",lambda q:self.fileNewCmd())
    self.parent.bind_all("<Control-s>",lambda q:self.fileSaveCmd())
    self.parent.bind_all("<Control-q>",lambda q:self.fileExitCmd())
    self.parent.bind_all("<F1>",lambda q:self.helpHelpCmd())

  def unbindGlobalEvents(self):
    #global bindings
    self.parent.unbind("<Control-z>")
    self.parent.unbind("<Control-Z>")
    self.parent.unbind("<Control-o>")
    self.parent.unbind("<Control-n>")
    self.parent.unbind("<Control-s>")
    self.parent.unbind("<Control-q>")
    self.parent.unbind("<F1>")

  def createWidgets(self):
    #first some configuration
    self.parent.geometry("800x600")
    self.parent.title("Xasy")
    self.parent.resizable(True,True)

    #try to capture the closing of the window
    #find a better way to do this since the widgets may
    #already be destroyed when this is called
    self.parent.protocol("WM_DELETE_WINDOW",self.canQuit)

    #the main menu
    self.mainMenu = Menu(self.parent)
    self.parent.config(menu=self.mainMenu)

    #the file menu
    self.fileMenu = Menu(self.mainMenu,tearoff=0)
    self.fileMenu.add_command(label="New",command=self.fileNewCmd,accelerator="Ctrl+N",underline=0)
    self.fileMenu.add_command(label="Open",command=self.fileOpenCmd,accelerator="Ctrl+O",underline=0)
    self.fileMenu.add_separator()
    self.fileMenu.add_command(label="Save",command=self.fileSaveCmd,accelerator="Ctrl+S",underline=0)
    self.fileMenu.add_command(label="Save As",command=self.fileSaveAsCmd,underline=5)
    self.fileMenu.add_separator()

    #an export menu
    self.exportMenu = Menu(self.fileMenu,tearoff=0)
    self.exportMenu.add_command(label="EPS...",command=self.exportEPS,underline=0)
    self.exportMenu.add_command(label="PDF...",command=self.exportPDF,underline=0)
    self.exportMenu.add_command(label="GIF...",command=self.exportGIF,underline=0)
    self.exportMenu.add_command(label="PNG...",command=self.exportPNG,underline=1)
    self.exportMenu.add_command(label="SVG...",command=self.exportSVG,underline=0)
    self.fileMenu.add_cascade(label="Export",menu=self.exportMenu,underline=1)
    self.fileMenu.add_separator()

    self.fileMenu.add_command(label="Quit",command=self.fileExitCmd,accelerator="Ctrl+Q",underline=0)

    self.mainMenu.add_cascade(label="File",menu=self.fileMenu,underline=0)

    #the edit menu
    self.editMenu = Menu(self.mainMenu,tearoff=0)
    self.editMenu.add_command(label="Undo",command=self.editUndoCmd,accelerator="Ctrl+Z",underline=0)
    self.editMenu.add_command(label="Redo",command=self.editRedoCmd,accelerator="Shift+Ctrl+Z",underline=0)
    self.mainMenu.add_cascade(label="Edit",menu=self.editMenu,underline=0)

    #the tools menu
    self.toolsMenu = Menu(self.mainMenu,tearoff=0)
    self.mainMenu.add_cascade(label="Tools",menu=self.toolsMenu,underline=0)

    #the options menu
    self.optionsMenu = Menu(self.toolsMenu,tearoff=0)
    self.toolsMenu.add_cascade(label="Options",menu=self.optionsMenu,underline=0)
    self.optionsMenu.add_command(label="Edit...",command=self.editOptions,underline=0)
    self.optionsMenu.add_command(label="Reset defaults",command=self.resetOptions,underline=6)

    #the help menu
    self.helpMenu = Menu(self.mainMenu,tearoff=0)
    self.helpMenu.add_command(label="Help",command=self.helpHelpCmd,state=DISABLED,accelerator="F1",underline=0)
    self.helpMenu.add_command(label="Asymptote Documentation",command=self.helpAsyDocCmd,underline=10)
    self.helpMenu.add_separator()
    self.helpMenu.add_command(label="About xasy",command=self.helpAboutCmd,underline=0)
    self.mainMenu.add_cascade(label="Help",menu=self.helpMenu,underline=0)

    #status bar
    self.statusBar = Frame(self.parent,relief=FLAT)

    self.magVal = DoubleVar()
    self.magVal.set(round(100*self.magnification,1))
    self.magVal.trace('w',self.zoomViewCmd)
    zoomList = self.magList
    if self.magnification not in zoomList:
      zoomList.append(self.magnification)
    zoomList.sort()
    zoomList = [round(100*i,1) for i in zoomList]
    self.zoomMenu = OptionMenu(self.statusBar,self.magVal,*zoomList)
    self.zoomMenu.pack(side=RIGHT)
    Label(self.statusBar,text="Zoom:",anchor=E,width=7).pack(side=RIGHT)

    self.coords = Label(self.statusBar,text="(0,0)",relief=SUNKEN,anchor=W)
    self.coords.pack(side=RIGHT,anchor=S)
    self.status = Label(self.statusBar,text="Ready",relief=SUNKEN,anchor=W)
    self.status.pack(side=RIGHT,fill=X,expand=1,anchor=SW)
    self.statusBar.pack(side=BOTTOM,fill=X)

    #toolbar for transformation, drawing, and adjustment commands
    self.toolBar = Frame(self.parent,relief=FLAT,borderwidth=3)

    #let's load some images
    self.toolIcons = {}
    for x in iconB64.keys():
      self.toolIcons[x] = PhotoImage(data=iconB64[x])

    self.transformLbl = Label(self.toolBar,text="",anchor=W)
    self.transformLbl.grid(row=0,column=0,columnspan=2,sticky=W)
    self.toolSelectButton = Button(self.toolBar,command=self.toolSelectCmd,image=self.toolIcons["select"])
    self.toolSelectButton.grid(row=1,column=0,sticky=N+S+E+W)
    self.toolMoveButton = Button(self.toolBar,command=self.toolMoveCmd,image=self.toolIcons["move"])
    self.toolMoveButton.grid(row=2,column=0,sticky=N+S+E+W)
    self.toolRotateButton = Button(self.toolBar,command=self.toolRotateCmd,image=self.toolIcons["rotate"])
    self.toolRotateButton.grid(row=2,column=1,sticky=N+S+E+W)
    self.toolVertiMoveButton = Button(self.toolBar,command=self.toolVertiMoveCmd,image=self.toolIcons["vertiMove"])
    self.toolVertiMoveButton.grid(row=3,column=0,sticky=N+S+E+W)
    self.toolHorizMoveButton = Button(self.toolBar,command=self.toolHorizMoveCmd,image=self.toolIcons["horizMove"])
    self.toolHorizMoveButton.grid(row=3,column=1,sticky=N+S+E+W)

    self.drawLbl = Label(self.toolBar,text="",anchor=W)
    self.drawLbl.grid(row=4,column=0,columnspan=2,sticky=W)
    self.toolDrawLinesButton = Button(self.toolBar,command=self.toolDrawLinesCmd,image=self.toolIcons["drawLines"])
    self.toolDrawLinesButton.grid(row=5,column=0,sticky=N+S+E+W)
    self.toolDrawBeziButton = Button(self.toolBar,command=self.toolDrawBeziCmd,image=self.toolIcons["drawBezi"])
    self.toolDrawBeziButton.grid(row=5,column=1,sticky=N+S+E+W)
    self.toolDrawPolyButton = Button(self.toolBar,command=self.toolDrawPolyCmd,image=self.toolIcons["drawPoly"])
    self.toolDrawPolyButton.grid(row=6,column=0,sticky=N+S+E+W)
    self.toolFillPolyButton = Button(self.toolBar,command=self.toolFillPolyCmd,image=self.toolIcons["fillPoly"])
    self.toolFillPolyButton.grid(row=6,column=1,sticky=N+S+E+W)
    self.toolDrawEllipButton = Button(self.toolBar,command=self.toolDrawEllipCmd,image=self.toolIcons["drawEllip"],state=DISABLED,relief=FLAT)
    #self.toolDrawEllipButton.grid(row=7,column=0,sticky=N+S+E+W)
    self.toolFillEllipButton = Button(self.toolBar,command=self.toolFillEllipCmd,image=self.toolIcons["fillEllip"],state=DISABLED,relief=FLAT)
    #self.toolFillEllipButton.grid(row=7,column=1,sticky=N+S+E+W)
    self.toolDrawShapeButton = Button(self.toolBar,command=self.toolDrawShapeCmd,image=self.toolIcons["drawShape"])
    self.toolDrawShapeButton.grid(row=8,column=0,sticky=N+S+E+W)
    self.toolFillShapeButton = Button(self.toolBar,command=self.toolFillShapeCmd,image=self.toolIcons["fillShape"])
    self.toolFillShapeButton.grid(row=8,column=1,sticky=N+S+E+W)
    self.toolTextButton = Button(self.toolBar,command=self.toolTextCmd,image=self.toolIcons["text"])
    self.toolTextButton.grid(row=9,column=0,sticky=N+S+E+W)
    self.toolAsyButton = Button(self.toolBar,command=self.toolAsyCmd,image=self.toolIcons["asy"])
    self.toolAsyButton.grid(row=9,column=1,sticky=N+S+E+W)

    self.adjLbl = Label(self.toolBar,text="",anchor=W)
    self.adjLbl.grid(row=10,column=0,columnspan=2,sticky=W)
    self.toolRaiseButton = Button(self.toolBar,command=self.toolRaiseCmd,image=self.toolIcons["raise"])
    self.toolRaiseButton.grid(row=11,column=0,sticky=N+S+E+W)
    self.toolLowerButton = Button(self.toolBar,command=self.toolLowerCmd,image=self.toolIcons["lower"])
    self.toolLowerButton.grid(row=11,column=1,sticky=N+S+E+W)

    self.toolBar.pack(side=LEFT,anchor=NW)

    #documentation for the tool bar buttons
    self.toolDocs = {
      self.toolSelectButton : "Click an item to select it. Control-Click will select/deselect additional items. Use mouse scroller (or Up/Down keys) to raise/lower highlighted items.",
      self.toolMoveButton : "Drag a selected item.",
      self.toolHorizMoveButton : "Drag a selected item. Only horizontal translation will be applied.",
      self.toolVertiMoveButton : "Drag a selected item. Only vertical translation will be applied.",
      self.toolRotateButton : "Drag a selected item to rotate it.",
      self.toolDrawLinesButton : "Click to draw line segments. Double click to place last point.",
      self.toolDrawBeziButton : "Click to place points. Double click to place last point.",
      self.toolDrawPolyButton : "Click to place vertices. Double click to place last point.",
      self.toolFillPolyButton : "Click to place vertices. Double click to place last point.",
      self.toolDrawEllipButton : "(UNIMPLEMENTED)Click to place center. Move mouse to achieve correct shape and double click.",
      self.toolFillEllipButton : "(UNIMPLEMENTED)Click to place center. Move mouse to achieve correct shape and double click.",
      self.toolDrawShapeButton : "Click to place points. Double click to place last point.",
      self.toolFillShapeButton : "Click to place points. Double click to place last point.",
      self.toolTextButton : "Click location of top left label position and enter text in dialog.",
      self.toolRaiseButton : "Raise selected items to top.",
      self.toolLowerButton : "Lower selected items to bottom.",
      self.toolAsyButton : "Insert/Edit Asymptote code."
    }

    #Current pen settings
    self.optionsBar = Frame(self.parent,height=100,relief=FLAT,borderwidth=3)
    self.penDisp = Canvas(self.optionsBar,width=100,height=25,bg="white",relief=SUNKEN,borderwidth=3)
    self.penDisp.grid(row=0,column=0,padx=3,pady=3)
    self.penDisp.create_line(10,25,30,10,60,20,80,10,smooth=True,tags="penDisp")
    self.penDisp.create_text(100,30,text="x1",tags="penMag",anchor=SE,font=("times","8"))
    self.penColButton = Button(self.optionsBar,text="Color...",width=5,command=self.setPenColCmd,relief=FLAT)
    self.penColButton.grid(row=0,column=1,padx=3,pady=3)
    Label(self.optionsBar,text="Width",anchor=E).grid(row=0,column=2)
    self.penWidthEntry = Entry(self.optionsBar,width=5)
    self.penWidthEntry.bind("<KeyRelease>",self.penWidthChanged)
    self.penWidthEntry.bind("<FocusOut>",self.applyPenWidthEvt)
    self.penWidthEntry.bind("<Return>",self.applyPenWidthEvt)
    self.penWidthEntry.grid(row=0,column=3)
    Label(self.optionsBar,text="Options",anchor=E).grid(row=0,column=4)
    self.penOptEntry = Entry(self.optionsBar)
    self.penOptEntry.bind("<FocusOut>",self.applyPenOptEvt)
    self.penOptEntry.bind("<Return>",self.applyPenOptEvt)
    self.penOptEntry.grid(row=0,column=5)
    self.optionsBar.pack(side=BOTTOM,anchor=NW)

    #a paned window for the canvas and propert explorer
    self.windowPane = PanedWindow(self.parent)

    #a property explorer
    self.propFrame = Frame(self.parent)
    self.propFrame.rowconfigure(1,weight=1)
    self.propFrame.columnconfigure(0,weight=1)
    Label(self.propFrame,text="Item List").grid(row=0,column=0,columnspan=2)
    self.itemScroll = Scrollbar(self.propFrame,orient=VERTICAL)
    self.propList = Listbox(self.propFrame, yscrollcommand=self.itemScroll.set)
    self.itemScroll.config(command=self.propList.yview)
    self.itemScroll.grid(row=1,column=1,sticky=N+S)
    self.propList.grid(row=1,column=0,sticky=N+S+E+W)
    self.propList.bind("<Double-Button-1>",self.propSelect)
    self.propList.bind("<Button-3>",self.itemPropMenuPopup)

    #the canvas's frame
    self.canvFrame = Frame(self.parent,relief=FLAT,borderwidth=0)
    self.canvFrame.rowconfigure(0,weight=1)
    self.canvFrame.columnconfigure(0,weight=1)
    self.canvVScroll = Scrollbar(self.canvFrame,orient=VERTICAL)
    self.canvHScroll = Scrollbar(self.canvFrame,orient=HORIZONTAL)
    self.canvHScroll.grid(row=1,column=0,sticky=E+W)
    self.canvVScroll.grid(row=0,column=1,sticky=N+S)

    #add the frames to the window pane
    self.windowPane.pack(side=RIGHT,fill=BOTH,expand=True)
    self.windowPane.add(self.canvFrame)
    self.windowPane.add(self.propFrame)
    self.windowPane.paneconfigure(self.propFrame,minsize=50,sticky=N+S+E+W)
    self.windowPane.bind("<Double-Button-1>",self.togglePaneEvt)

    #the highly important canvas!
    self.mainCanvas = Canvas(self.canvFrame,relief=SUNKEN,background="white",borderwidth=3,
                  highlightthickness=0,closeenough=1.0,yscrollcommand=self.canvVScroll.set,
                  xscrollcommand=self.canvHScroll.set)
    self.mainCanvas.grid(row=0,column=0,sticky=N+S+E+W)

    self.canvVScroll.config(command=self.mainCanvas.yview)
    self.canvHScroll.config(command=self.mainCanvas.xview)

    self.mainCanvas.bind("<Motion>",self.canvMotion)
    self.mainCanvas.bind("<Button-1>",self.canvLeftDown)
    self.mainCanvas.bind("<Double-Button-1>",self.endDraw)
    self.mainCanvas.bind("<ButtonRelease-1>",self.canvLeftUp)
    self.mainCanvas.bind("<B1-Motion>",self.canvDrag)

    self.mainCanvas.bind("<Enter>",self.canvEnter)
    self.mainCanvas.bind("<Leave>",self.canvLeave)
    self.mainCanvas.bind("<Delete>",self.itemDelete)
    #self.mainCanvas.bind("<Button-3>",self.canvRightDown)
    #self.mainCanvas.bind("<ButtonRelease-3>",self.canvRightUp)
    self.mainCanvas.bind("<Button-4>",self.itemRaise)
    self.mainCanvas.bind("<Button-5>",self.itemLower)
    self.mainCanvas.bind("<Up>",self.itemRaise)
    self.mainCanvas.bind("<Down>",self.itemLower)
    self.mainCanvas.bind("<Configure>",self.configEvt)

  def foregroundPenColor(self,hex):
    hex = hex[1:]
    rgb = max(hex[0:2], hex[2:4], hex[4:6])
    if(rgb >= "80"):
      return "black"
    else:
      return "white"

  def resetGUI(self):
    #set up the main window
    self.filename = None
    self.fileToOpen = None
    self.retitle()

    #set up the paned window
    self.paneVisible = True

    #setup the pen entries
    self.pendingPenWidthChange = None
    self.pendingPenOptChange = None

    #load one-time configs
    xasyOptions.load()
    self.tkPenColor = xasyOptions.options['defPenColor']
    self.penColor = makeRGBfromTkColor(self.tkPenColor)
    self.penColButton.config(activebackground=self.tkPenColor,
                             activeforeground=self.foregroundPenColor(self.tkPenColor))
    self.penWidth = xasyOptions.options['defPenWidth']
    self.penWidthEntry.select_range(0,END)
    self.penWidthEntry.delete(0,END)
    self.penWidthEntry.insert(END,str(self.penWidth))
    self.penOptions = xasyOptions.options['defPenOptions']
    self.penOptEntry.select_range(0,END)
    self.penOptEntry.delete(0,END)
    self.penOptEntry.insert(END,str(self.penOptions))
    self.showCurrentPen()

    #load modifiable configs
    self.applyOptions()

    #set up editing
    self.editor = None

    #set up drawing
    self.pathInProgress = asyPath()
    self.currentIDTag = -1
    self.inDrawingMode = False
    self.freeMouseDown = True
    self.dragSelecting = False
    self.itemsBeingRotated = []
    self.inRotatingMode = False

    #set up the toolbar
    try:
      self.updateSelectedButton(self.toolSelectButton)
    except:
      self.selectedButton = self.toolSelectButton
      self.updateSelectedButton(self.toolSelectButton)

    #set up the canvas
    self.mainCanvas.delete(ALL)
    self.mainCanvas.create_rectangle(0,0,0,0,tags="outlineBox",width=0,outline="#801111",dash=(3,6))
    self.backColor = "white" #in future, load this from an options file. Or, should this really be an option?
    self.mainCanvas.configure(background=self.backColor)

    #set up the xasy item list
    self.fileItems = []
    self.propList.delete(0,END)
    self.updateCanvasSize()

    #setup timer
    self.tickCount = 0

    #setup undo/redo!
    self.undoRedoStack = actionStack()
    self.amDragging = False

  def retitle(self):
    if self.filename == None:
      self.parent.title("Xasy - New File")
    else:
      name = os.path.abspath(self.filename)
      name = os.path.basename(name)
      self.parent.title("Xasy - %s"%name)

  def applyOptions(self):
    self.gridcolor = xasyOptions.options['gridColor']
    self.tickcolor = xasyOptions.options['tickColor']
    self.axiscolor = xasyOptions.options['axesColor']
    self.gridVisible = xasyOptions.options['showGrid']
    self.gridxspace = xasyOptions.options['gridX']
    self.gridyspace = xasyOptions.options['gridY']
    self.axesVisible = xasyOptions.options['showAxes']
    self.axisxspace = xasyOptions.options['axisX']
    self.axisyspace = xasyOptions.options['axisY']
    self.updateCanvasSize()
    #test the asyProcess
    startQuickAsy()
    if not quickAsyRunning():
      if tkMessageBox.askyesno("Xasy Error","Asymptote could not be executed.\r\nTry to find Asymptote automatically?"):
        xasyOptions.setAsyPathFromWindowsRegistry()
        xasyOptions.save()
        startQuickAsy()
    while not quickAsyRunning():
      if tkMessageBox.askyesno("Xasy Error","Asymptote could not be executed.\r\nEdit settings?"):
        xasyOptionsDialog.xasyOptionsDlg(self.parent)
        xasyOptions.save()
        startQuickAsy()
      else:
        self.parent.destroy()
        sys.exit(1)

  def drawGrid(self):
    self.mainCanvas.delete("grid")
    if not self.gridVisible:
      return
    left,top,right,bottom = map(int,self.mainCanvas.cget("scrollregion").split())
    gridyspace = int(self.magnification*self.gridyspace)
    gridxspace = int(self.magnification*self.gridxspace)
    if gridxspace >= 3 and gridyspace >= 3:
      for i in range(0,right,gridxspace):
        self.mainCanvas.create_line(i,top,i,bottom,tags=("grid","vertical"),fill=self.gridcolor)
      for i in range(-gridxspace,left,-gridxspace):
        self.mainCanvas.create_line(i,top,i,bottom,tags=("grid","vertical"),fill=self.gridcolor)
      for i in range(-gridyspace,top,-gridyspace):
        self.mainCanvas.create_line(left,i,right,i,tags=("grid","horizontal"),fill=self.gridcolor)
      for i in range(0,bottom,gridyspace):
        self.mainCanvas.create_line(left,i,right,i,tags=("grid","horizontal"),fill=self.gridcolor)
    self.mainCanvas.tag_lower("grid")

  def drawAxes(self):
    self.mainCanvas.delete("axes")
    if not self.axesVisible:
      return
    left,top,right,bottom = map(int,self.mainCanvas.cget("scrollregion").split())
    self.mainCanvas.create_line(0,top,0,bottom,tags=("axes","yaxis"),fill=self.axiscolor)
    self.mainCanvas.create_line(left,0,right,0,tags=("axes","xaxis"),fill=self.axiscolor)
    axisxspace = int(self.magnification*self.axisxspace)
    axisyspace = int(self.magnification*self.axisyspace)
    if axisxspace >= 3 and axisyspace >= 3:
      for i in range(axisxspace,right,axisxspace):
        self.mainCanvas.create_line(i,-5,i,5,tags=("axes","xaxis-ticks"),fill=self.tickcolor)
      for i in range(-axisxspace,left,-axisxspace):
        self.mainCanvas.create_line(i,-5,i,5,tags=("axes","xaxis-ticks"),fill=self.tickcolor)
      for i in range(-axisyspace,top,-axisyspace):
        self.mainCanvas.create_line(-5,i,5,i,tags=("axes","yaxis-ticks"),fill=self.tickcolor)
      for i in range(axisyspace,bottom,axisyspace):
        self.mainCanvas.create_line(-5,i,5,i,tags=("axes","yaxis-ticks"),fill=self.tickcolor)
    self.mainCanvas.tag_lower("axes")

  def updateCanvasSize(self,left=-200,top=-200,right=200,bottom=200):
    self.parent.update_idletasks()
    bbox = self.mainCanvas.bbox("drawn || image || node || precontrol || postcontrol")
    if bbox == None:
      bbox = (0,0,0,0)
    #(topleft, bottomright)
    left = min(bbox[0],left)
    top = min(bbox[1],top)
    right = max(bbox[2],right)
    bottom = max(bbox[3],bottom)
    w,h = self.mainCanvas.winfo_width(),self.mainCanvas.winfo_height()
    if right-left < w:
      extraw = w-(right-left)
      right += extraw/2
      left -= extraw/2
    if bottom-top < h:
      extrah = h-(bottom-top)
      bottom += extrah/2
      top -= extrah/2
    self.mainCanvas.config(scrollregion=(left,top,right,bottom))
    #self.mainCanvas.xview(MOVETO,float(split(self.mainCanvas["scrollregion"])[0]))
    #self.mainCanvas.yview(MOVETO,float(split(self.mainCanvas["scrollregion"])[1]))
    #self.mainCanvas.xview(MOVETO,(left+right)/2)
    #self.mainCanvas.yview(MOVETO,(top+bottom)/2)
    self.drawAxes()
    self.drawGrid()

  def bindEvents(self,tagorID):
    if tagorID == None:
      return
    self.mainCanvas.tag_bind(tagorID,"<Control-Button-1>",self.itemToggleSelect)
    self.mainCanvas.tag_bind(tagorID,"<Button-1>",self.itemSelect)
    self.mainCanvas.tag_bind(tagorID,"<ButtonRelease-1>",self.itemMouseUp)
    self.mainCanvas.tag_bind(tagorID,"<Double-Button-1>",self.itemEditEvt)
    self.mainCanvas.tag_bind(tagorID,"<B1-Motion>",self.itemDrag)
    self.mainCanvas.tag_bind(tagorID,"<Delete>",self.itemDelete)
    self.mainCanvas.tag_bind(tagorID,"<Enter>",self.itemHighlight)
    self.mainCanvas.tag_bind(tagorID,"<Button-3>",self.itemCanvasMenuPopup)

  def bindItemEvents(self,item):
    if item == None:
      return
    if isinstance(item,xasyScript) or isinstance(item,xasyText):
      for image in item.imageList:
        self.bindEvents(image.IDTag)
    else:
      self.bindEvents(item.IDTag)

  def canQuit(self,force=False):
    #print "Quitting"
    if not force and not self.testOrAcquireLock():
      return
    try:
      self.releaseLock()
    except:
      pass
    if self.undoRedoStack.changesMade():
      result = tkMessageBox._show("xasy","File has been modified.\nSave changes?",icon=tkMessageBox.QUESTION,type=tkMessageBox.YESNOCANCEL)
      if str(result) == tkMessageBox.CANCEL:
        return
      elif result == tkMessageBox.YES:
        self.fileSaveCmd()
    try:
      os.rmdir(getAsyTempDir())
    except:
      pass
    self.parent.destroy()

  def openFile(self,name):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock() #release the lock for loadFile
    self.resetGUI()
    self.loadFile(name)

  def loadFile(self,name):
    self.status.config(text="Loading "+name)
    self.filename = os.path.abspath(name)
    startQuickAsy()
    self.retitle()
    try:
      try:
        f = open(self.filename,'rt')
      except:
        if self.filename[-4:] == ".asy":
          raise
        else:
          f = open(self.filename+".asy",'rt')
          self.filename += ".asy"
          self.retitle()
      self.fileItems = xasyFile.parseFile(f)
      f.close()
    except IOError:
      tkMessageBox.showerror("File Opening Failed.","File could not be opened.")
      self.fileItems = []
    except:
      self.fileItems = []
      self.autoMakeScript = True
      if self.autoMakeScript or tkMessageBox.askyesno("Error Opening File", "File was not recognized as an xasy file.\nLoad as a script item?"):
        try:
          item = xasyScript(self.mainCanvas)
          f.seek(0)
          item.setScript(f.read())
          self.addItemToFile(item)
        except:
          tkMessageBox.showerror("File Opening Failed.","Could not load as a script item.")
          self.fileItems = []
    self.populateCanvasWithItems()
    self.populatePropertyList()
    self.updateCanvasSize()

  def populateCanvasWithItems(self):
    if(not self.testOrAcquireLock()):
      return
    self.mainCanvas.delete("drawn || image")
    self.itemCount = 0
    for item in self.fileItems:
      item.drawOnCanvas(self.mainCanvas,self.magnification,forceAddition=True)
      self.bindItemEvents(item)
    self.releaseLock()

  def propListCountItem(self,item):
    plist = self.propList.get(0,END)
    count = 1
    for text in plist:
      if text.startswith(item):
        count += 1
    return count

  def describeItem(self,item):
    if isinstance(item,xasyScript):
      return "Code Module "+str(self.propListCountItem("Code Module"))
    elif isinstance(item,xasyText):
      return "Text Label "+str(self.propListCountItem("Text Label"))
    elif isinstance(item,xasyFilledShape):
      return "Filled Shape "+str(self.propListCountItem("Filled Shape"))
    elif isinstance(item,xasyShape):
      return "Outline "+str(self.propListCountItem("Outline"))
    else:
      return "If this happened, the program is corrupt!"

  def populatePropertyList(self):
    self.propList.delete(0,END)
    for item in self.fileItems:
      self.propList.insert(0,self.describeItem(item))

  def saveFile(self,name):
    if(not self.testOrAcquireLock()):
      return
    f = open(name,"wt")
    xasyFile.saveFile(f,self.fileItems)
    f.close()
    self.undoRedoStack.setCommitLevel()
    self.retitle()
    self.releaseLock()

  #menu commands
  def fileNewCmd(self):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    #print "Create New File"
    if self.undoRedoStack.changesMade():
      result = tkMessageBox._show("xasy","File has been modified.\nSave changes?",icon=tkMessageBox.QUESTION,type=tkMessageBox.YESNOCANCEL)
      if str(result) == tkMessageBox.CANCEL:
        return
      elif result == tkMessageBox.YES:
        self.fileSaveCmd()
    self.resetGUI()

  def fileOpenCmd(self):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    #print "Open a file"
    if self.undoRedoStack.changesMade():
      result = tkMessageBox._show("xasy","File has been modified.\nSave changes?",icon=tkMessageBox.QUESTION,type=tkMessageBox.YESNOCANCEL)
      if str(result) == tkMessageBox.CANCEL:
        return
      elif result == tkMessageBox.YES:
        self.fileSaveCmd()
    filename=tkFileDialog.askopenfilename(filetypes=[("asy files","*.asy"),("All files","*")],title="Open File",parent=self.parent)
    if type(filename) != type((0,)) and filename != None and filename != '':
      self.filename = filename
      self.openFile(self.filename)

  def fileSaveCmd(self):
    #print "Save current file"
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    if self.filename == None:
      filename=tkFileDialog.asksaveasfilename(defaultextension=".asy",filetypes=[("asy files","*.asy")],initialfile="newDrawing.asy",parent=self.parent,title="Save File")
      if type(filename) != type((0,)) and filename != None and filename != '':
        self.filename = filename
    if self.filename != None:
      self.saveFile(self.filename)

  def fileSaveAsCmd(self):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    #print "Save current file as"
    filename=tkFileDialog.asksaveasfilename(defaultextension=".asy",filetypes=[("asy files","*.asy")],initialfile="newDrawing.asy",parent=self.parent,title="Save File")
    if type(filename) != type((0,)) and filename != None and filename != '':
      self.filename = filename
      self.saveFile(self.filename)

  #export the file
  def exportEPS(self):
    self.exportFile(self.filename,"eps")

  def exportPDF(self):
    self.exportFile(self.filename,"pdf")

  def exportGIF(self):
    self.exportFile(self.filename,"gif")

  def exportPNG(self):
    self.exportFile(self.filename,"png")

  def exportSVG(self):
    self.exportFile(self.filename,"svg")

  def exportFile(self,inFile, outFormat):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    if inFile == None:
      if tkMessageBox.askyesno("xasy","File has not been saved.\nSave?"):
        self.fileSaveAsCmd()
        inFile = self.filename
      else:
        return
    elif self.undoRedoStack.changesMade():
      choice = tkMessageBox._show("xasy","File has been modified.\nOnly saved changes can be exported.\nDo you want to save changes?",icon=tkMessageBox.QUESTION,type=tkMessageBox.YESNOCANCEL)
      choice = str(choice)
      if choice != tkMessageBox.YES:
        return
      else:
        self.fileSaveCmd()
    name = os.path.splitext(os.path.basename(self.filename))[0]+'.'+outFormat
    outfilename = tkFileDialog.asksaveasfilename(defaultextension = '.'+outFormat,filetypes=[(outFormat+" files","*."+outFormat)],initialfile=name,parent=self.parent,title="Choose output file")
    if type(outfilename)==type((0,)) or not outfilename or outfilename == '':
      return
    fullname = os.path.abspath(outfilename)
    outName = os.path.basename(outfilename)
    command=[xasyOptions.options['asyPath'],"-f"+outFormat,"-o"+fullname,inFile]
    saver = subprocess.Popen(command,stdin=PIPE,stdout=PIPE,stderr=PIPE)
    saver.wait()
    if saver.returncode != 0:
      tkMessageBox.showerror("Export Error","Export Error:\n"+saver.stdout.read()+saver.stderr.read())
      self.status.config(text="Error exporting file")
    else:
      self.status.config(text="File exported successfully")

  def fileExitCmd(self):
    #print "Exit xasy"
    self.canQuit()

  def editUndoCmd(self):
    if not self.editor == None:
      return
    if(not self.testOrAcquireLock()):
      return
    self.undoOperation()
    self.releaseLock()

  def editRedoCmd(self):
    if not self.editor == None:
      return
    if(not self.testOrAcquireLock()):
      return
    self.redoOperation()
    self.releaseLock()

  def helpHelpCmd(self):
    print "Get help on xasy"

  def helpAsyDocCmd(self):
    #print "Open documentation about Asymptote"
    asyExecute("help;\n")

  def helpAboutCmd(self):
    tkMessageBox.showinfo("About xasy","A graphical interface for Asymptote "+xasyVersion)

  def updateSelectedButton(self,newB):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    #disable switching modes during an incomplete drawing operation
    if self.inDrawingMode:
      return
    self.selectedButton.config(relief = RAISED)
    if newB == self.toolSelectButton or self.selectedButton == self.toolSelectButton:
      self.mainCanvas.delete("highlightBox")
    if self.editor != None:
      self.editor.endEdit()
      if self.editor.modified:
        self.undoRedoStack.add(editDrawnItemAction(self,self.itemBeingEdited,copy.deepcopy(self.editor.shape),self.fileItems.index(self.editor.shape)))
    if newB not in (self.toolSelectButton,self.toolMoveButton,self.toolHorizMoveButton,self.toolVertiMoveButton,self.toolRotateButton):
      self.clearSelection()
    self.selectedButton = newB
    self.selectedButton.config(relief = SUNKEN)
    self.status.config(text=self.toolDocs[newB])

  #toolbar commands
  def toolSelectCmd(self):
    self.updateSelectedButton(self.toolSelectButton)
  def toolMoveCmd(self):
    self.updateSelectedButton(self.toolMoveButton)
  def toolRotateCmd(self):
    self.updateSelectedButton(self.toolRotateButton)
  def toolVertiMoveCmd(self):
    self.updateSelectedButton(self.toolVertiMoveButton)
  def toolHorizMoveCmd(self):
    self.updateSelectedButton(self.toolHorizMoveButton)
  def toolDrawLinesCmd(self):
    self.updateSelectedButton(self.toolDrawLinesButton)
  def toolDrawBeziCmd(self):
    self.updateSelectedButton(self.toolDrawBeziButton)
  def toolDrawPolyCmd(self):
    self.updateSelectedButton(self.toolDrawPolyButton)
  def toolFillPolyCmd(self):
    self.updateSelectedButton(self.toolFillPolyButton)
  def toolDrawEllipCmd(self):
    self.updateSelectedButton(self.toolDrawEllipButton)
  def toolFillEllipCmd(self):
    self.updateSelectedButton(self.toolFillEllipButton)
  def toolDrawShapeCmd(self):
    self.updateSelectedButton(self.toolDrawShapeButton)
  def toolFillShapeCmd(self):
    self.updateSelectedButton(self.toolFillShapeButton)
  def toolTextCmd(self):
    self.updateSelectedButton(self.toolTextButton)
  def toolAsyCmd(self):
    # ignore the command if we are too busy to process it
    if not self.testOrAcquireLock():
      return
    self.updateSelectedButton(self.toolSelectButton)
    self.clearSelection()
    self.clearHighlight()
    self.unbindGlobalEvents()
    try:
      self.getNewText("// enter your code here")
    except Exception, e:
      tkMessageBox.showerror('xasy Error',e.message)
    else:
      self.addItemToFile(xasyScript(self.mainCanvas))
      text = self.newText
      self.undoRedoStack.add(addScriptAction(self,self.fileItems[-1]))
      self.fileItems[-1].setScript(text)
      self.fileItems[-1].drawOnCanvas(self.mainCanvas,self.magnification)
      self.bindItemEvents(self.fileItems[-1])
    self.bindGlobalEvents()
    self.releaseLock()
  def toolRaiseCmd(self):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    if not self.inDrawingMode and self.editor == None:
      itemList = []
      indexList = []
      for ID in self.mainCanvas.find_withtag("selectedItem"):
        item = self.findItem(ID)
        if item not in itemList:
          itemList.append(item)
          indexList.append(self.fileItems.index(item))
          self.raiseSomething(item)
      self.undoRedoStack.add(itemRaiseAction(self,itemList,indexList))
  def toolLowerCmd(self):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    if not self.inDrawingMode and self.editor == None:
      itemList = []
      indexList = []
      for ID in self.mainCanvas.find_withtag("selectedItem"):
        item = self.findItem(ID)
        if item not in itemList:
          itemList.append(item)
          indexList.append(self.fileItems.index(item))
          self.lowerSomething(item)
      self.undoRedoStack.add(itemLowerAction(self,itemList,indexList))
  def itemRaise(self,event):
    self.mainCanvas.tag_raise(CURRENT)
  def itemLower(self,event):
    self.mainCanvas.tag_lower(CURRENT)

  #options bar commands
  def setPenColCmd(self):
    if not self.testOrAcquireLock():
      return
    old = self.penColor
    self.penColor = xasyColorDlg(self.parent).getColor(self.penColor)
    if self.penColor != old:
      self.tkPenColor = RGB255hex(RGBreal255(self.penColor))
      self.penColButton.config(activebackground=self.tkPenColor,
                               activeforeground=self.foregroundPenColor(self.tkPenColor))
      self.showCurrentPen()
    self.releaseLock()

  def clearSelection(self):
    self.hideSelectionBox()
    self.mainCanvas.dtag("selectedItem","selectedItem")

  def hideSelectionBox(self):
    self.mainCanvas.itemconfigure("outlineBox",width=1,outline=self.backColor)
    self.mainCanvas.tag_lower("outlineBox")
    self.mainCanvas.coords("outlineBox",self.mainCanvas.bbox(ALL))

  def showSelectionBox(self):
    self.mainCanvas.itemconfigure("outlineBox",width=2,outline="#801111")
    self.mainCanvas.tag_raise("outlineBox")

  def setSelection(self,what):
    self.mainCanvas.addtag_withtag("selectedItem",what)
    self.updateSelection()
    if self.selectedButton == self.toolSelectButton and len(self.mainCanvas.find_withtag("selectedItem")) > 0:
      self.updateSelectedButton(self.toolMoveButton)

  def unSelect(self,what):
    self.mainCanvas.dtag(what,"selectedItem")
    self.updateSelection()

  def updateSelection(self):
    self.clearHighlight()
    theBbox = self.mainCanvas.bbox("selectedItem")
    if theBbox != None:
      theBbox = (theBbox[0]-2,theBbox[1]-2,theBbox[2]+2,theBbox[3]+2)
      self.mainCanvas.coords("outlineBox",theBbox)
      self.showSelectionBox()
    else:
      self.clearSelection()

  #event handlers
  def updateZoom(self):
    self.zoomMenu.config(state=DISABLED)
    self.magnification = self.magVal.get()/100.0
    if self.magnification != self.previousZoom:
      self.populateCanvasWithItems()
      self.updateCanvasSize()
      self.updateSelection()
      self.drawAxes()
      self.drawGrid()
      self.previousZoom = self.magnification
    self.zoomMenu.config(state=NORMAL)

  def zoomViewCmd(self,*args):
    magnification = self.magVal.get()/100.0
    self.updateZoom();

  def selectItem(self,item):
    self.clearSelection()
    if isinstance(item,xasyScript) or isinstance(item,xasyText):
      for image in item.imageList:
        self.setSelection(image.IDTag)
    else:
      self.setSelection(item.IDTag)

  def propSelect(self,event):
    items = map(int, self.propList.curselection())
    if len(items)>0:
      try:
        self.selectItem(self.fileItems[len(self.fileItems)-items[0]-1])
      except:
        raise

  def findItem(self,ID):
    for item in self.fileItems:
      if isinstance(item,xasyScript) or isinstance(item,xasyText):
        for image in item.imageList:
          if image.IDTag == ID:
            return item
      else:
        if item.IDTag == ID:
          return item
    raise Exception,"Illegal operation: Item with matching ID could not be found."

  def findItemImageIndex(self,item,ID):
    count = 0
    for image in item.imageList:
      if image.IDTag == ID:
        return count
      else:
        count += 1
    raise Exception,"Illegal operation: Image with matching ID could not be found."
    return None

  def raiseSomething(self,item,force=False):
    if self.fileItems[-1] != item or force:
      index = len(self.fileItems)-self.fileItems.index(item)-1
      text = self.propList.get(index)
      self.propList.delete(index)
      self.propList.insert(0,text)
      for i in range(self.fileItems.index(item),len(self.fileItems)-1):
        self.fileItems[i] = self.fileItems[i+1]
      self.fileItems[-1] = item
      if isinstance(item,xasyScript) or isinstance(item,xasyText):
        for im in item.imageList:
          if im.IDTag != None:
            self.mainCanvas.tag_raise(im.IDTag)
      else:
        if item.IDTag != None:
          self.mainCanvas.tag_raise(item.IDTag)

  def lowerSomething(self,item):
    if self.fileItems[0] != item:
      index = len(self.fileItems)-self.fileItems.index(item)-1
      text = self.propList.get(index)
      self.propList.delete(index)
      self.propList.insert(END,text)
      indices = range(self.fileItems.index(item))
      indices.reverse()
      for i in indices:
        self.fileItems[i+1] = self.fileItems[i]
      self.fileItems[0] = item
      if isinstance(item,xasyScript) or isinstance(item,xasyText):
        item.imageList.reverse()
        for im in item.imageList:
          if im.IDTag != None:
            self.mainCanvas.tag_lower(im.IDTag)
        item.imageList.reverse()
      else:
        if item.IDTag != None:
          self.mainCanvas.tag_lower(item.IDTag)
      self.mainCanvas.tag_lower("axes || grid")

  def translateSomething(self,ID,translation,specificItem=None,specificIndex=None):
    transform = asyTransform((translation[0],translation[1],1,0,0,1))
    if ID == -1:
      item = specificItem
    else:
      item = self.findItem(ID)
    if isinstance(item,xasyText) or isinstance(item,xasyScript):
      if ID == -1:
        index = specificIndex
      else:
        index = self.findItemImageIndex(item,ID)
      try:
        original = item.transform[index]
      except:
        original = identity()
      item.transform[index] = transform*original
      bbox = item.imageList[index].originalImage.bbox
      item.imageList[index].originalImage.bbox = bbox[0]+translation[0],bbox[1]+translation[1],bbox[2]+translation[0],bbox[3]+translation[1]
    else:
      item.transform = [transform*item.transform[0]]

  def makeRotationMatrix(self,theta,origin):
    rotMat = (math.cos(theta),-math.sin(theta),math.sin(theta),math.cos(theta))
    shift = asyTransform((0,0,1-rotMat[0],-rotMat[1],-rotMat[2],1-rotMat[3]))*origin
    return asyTransform((shift[0],shift[1],rotMat[0],rotMat[1],rotMat[2],rotMat[3]))

  def rotateSomething(self,ID,theta,origin,specificItem=None,specificIndex=None):
    #print "Rotating by",theta*180.0/math.pi,"around",origin
    rotMat = self.makeRotationMatrix(theta,(origin[0]/self.magnification,origin[1]/self.magnification))
    #print rotMat
    if ID == -1:
      item = specificItem
    else:
      item = self.findItem(ID)
    if isinstance(item,xasyText) or isinstance(item,xasyScript):
      #transform the image
      if ID == -1:
        index = specificIndex
      else:
        index = self.findItemImageIndex(item,ID)
      try:
        original = item.transform[index]
      except:
        original = identity()
      oldBbox = item.imageList[index].originalImage.bbox
      oldBbox = (oldBbox[0],-oldBbox[1],oldBbox[2],-oldBbox[3])
      item.transform[index] = rotMat*item.transform[index]
      item.transform[index] = rotMat*original
      item.imageList[index].originalImage.theta += theta
      item.imageList[index].image = item.imageList[index].originalImage.rotate(item.imageList[index].originalImage.theta*180.0/math.pi,expand=True,resample=Image.BICUBIC)
      item.imageList[index].itk = ImageTk.PhotoImage(item.imageList[index].image)
      self.mainCanvas.itemconfigure(ID,image=item.imageList[index].itk)
      #the image has been rotated in place
      #now, compensate for any resizing and shift to the correct location
      #
      #  p0 --- p1               p1
      #  |      |     --->      /  \
      #  p2 --- p3             p0  p3
      #                         \ /
      #                          p2
      #
      rotMat2 = self.makeRotationMatrix(item.imageList[index].originalImage.theta,origin)
      p0 = rotMat2*(oldBbox[0],-oldBbox[3])#switch to usual coordinates
      p1 = rotMat2*(oldBbox[2],-oldBbox[3])
      p2 = rotMat2*(oldBbox[0],-oldBbox[1])
      p3 = rotMat2*(oldBbox[2],-oldBbox[1])
      newTopLeft = (min(p0[0],p1[0],p2[0],p3[0]),-max(p0[1],p1[1],p2[1],p3[1]))#switch back to screen coords
      shift = (newTopLeft[0]-oldBbox[0],newTopLeft[1]-oldBbox[3])
      #print theta*180.0/math.pi,origin,oldBbox,newTopLeft,shift
      #print item.imageList[index].originalImage.size
      #print item.imageList[index].image.size
      #print
      self.mainCanvas.coords(ID,oldBbox[0]+shift[0],oldBbox[3]+shift[1])
    else:
      #transform each point of the object
      xform = rotMat*item.transform[0]
      item.transform = [identity()]
      for i in range(len(item.path.nodeSet)):
        if item.path.nodeSet[i] != 'cycle':
          item.path.nodeSet[i] = xform*item.path.nodeSet[i]
      for i in range(len(item.path.controlSet)):
        item.path.controlSet[i][0] = xform*item.path.controlSet[i][0]
        item.path.controlSet[i][1] = xform*item.path.controlSet[i][1]
      item.drawOnCanvas(self.mainCanvas,self.magnification)

  def deleteItem(self,item):
    if isinstance(item,xasyScript) or isinstance(item,xasyText):
      if isinstance(item,xasyScript):
        self.undoRedoStack.add(deleteScriptAction(self,item,self.fileItems.index(item)))
      else:
        self.undoRedoStack.add(deleteLabelAction(self,item,self.fileItems.index(item)))
      for image in item.imageList:
        self.mainCanvas.delete(image.IDTag)
    else:
      if isinstance(item,xasyDrawnItem):
        self.undoRedoStack.add(deleteDrawnItemAction(self,item,self.fileItems.index(item)))
      self.mainCanvas.delete(item.IDTag)
    self.fileItems.remove(item)
    self.populatePropertyList()
    self.clearSelection()

  def deleteSomething(self,ID):
    self.clearSelection()
    self.clearHighlight()
    if self.editor != None:
      self.editor.endEdit()
      if self.editor.modified:
        self.undoRedoStack.add(editDrawnItemAction(self,self.itemBeingEdited,copy.deepcopy(self.editor.shape),self.fileItems.index(self.editor.shape)))
    item = self.findItem(ID)
    #save an event on the undoredo stack
    if isinstance(item,xasyScript):
      index = self.findItemImageIndex(item,ID)
      item.transform[index].deleted = True
    else:
      if isinstance(item,xasyText):
        self.undoRedoStack.add(deleteLabelAction(self,item,self.fileItems.index(item)))
      elif isinstance(item,xasyDrawnItem):
        self.undoRedoStack.add(deleteDrawnItemAction(self,item,self.fileItems.index(item)))
      self.fileItems.remove(item)
    self.mainCanvas.delete(ID)
    self.populatePropertyList()

  def scriptEditThread(self,oldText):
    try:
      self.newText = xasyCodeEditor.getText(oldText)
    except:
      self.newText = -1

  def getNewText(self,oldText):
    editThread = threading.Thread(target=self.scriptEditThread,args=(oldText,))
    editThread.start()
    while editThread.isAlive():
      time.sleep(0.05)
      self.parent.update()
    editThread.join()
    if type(self.newText)==type(-1):
      self.newText = ''
      raise Exception('Error launching external editor. Please check xasy options.')

  def itemEdit(self,item):
    # are we too busy?
    if not self.testOrAcquireLock():
      return
    self.updateSelectedButton(self.toolSelectButton)
    if isinstance(item,xasyScript):
      self.unbindGlobalEvents()
      oldText = item.script
      try:
        self.getNewText(oldText)
      except Exception,e:
        tkMessageBox.showerror('xasy Error',e.message)
      else:
        if self.newText != oldText:
          self.undoRedoStack.add(editScriptAction(self,item,self.newText,oldText))
          item.setScript(self.newText)
          item.drawOnCanvas(self.mainCanvas,self.magnification)
          self.bindItemEvents(item)
      self.bindGlobalEvents()
    elif isinstance(item,xasyText):
      theText = tkSimpleDialog.askstring(title="Xasy - Text",prompt="Enter text to display:",initialvalue=item.label.text,parent=self.parent)
      if theText != None and theText != "":
        self.undoRedoStack.add(editLabelTextAction(self,item,theText,item.label.text))
        item.label.text = theText
        item.drawOnCanvas(self.mainCanvas,self.magnification)
        self.bindItemEvents(item)
    elif isinstance(item,xasyShape):
      self.clearSelection()
      self.clearHighlight()
      self.itemBeingEdited = copy.deepcopy(item)
      self.editor = xasyBezierEditor(self,item,self.mainCanvas)
    self.updateSelection()
    self.releaseLock()

  def itemEditEvt(self,event):
    if not self.inDrawingMode:
      ID = self.mainCanvas.find_withtag(CURRENT)[0]
      item = self.findItem(ID)
      self.itemEdit(item)

  def itemDrag(self,event):
    x0,y0 = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    x = x0/self.magnification
    y = y0/self.magnification
    if self.selectedButton not in [self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton]:
      return
    if "selectedItem" in self.mainCanvas.gettags(CURRENT):
      self.amDragging = True
      for ID in self.mainCanvas.find_withtag("selectedItem"):
        transform = identity()
        if self.selectedButton == self.toolMoveButton:
          translation = (x0-self.dragStartx,-(y0-self.dragStarty))
        elif self.selectedButton == self.toolVertiMoveButton:
          translation = (0,-(y0-self.dragStarty))
        elif self.selectedButton == self.toolHorizMoveButton:
          translation = (x0-self.dragStartx,0)
        self.translateSomething(ID,(translation[0]/self.magnification,translation[1]/self.magnification))
        self.mainCanvas.move(ID,translation[0],-translation[1])
        self.updateSelection()
        self.updateCanvasSize()
      self.distanceDragged = (self.distanceDragged[0]+translation[0],self.distanceDragged[1]-translation[1])
    self.dragStartx,self.dragStarty = x0,y0

  def itemMouseUp(self,event):
    self.freeMouseDown = True
    if self.amDragging:
      IDList = self.mainCanvas.find_withtag("selectedItem")
      itemList = []
      indexList = []
      for ID in IDList:
        item = self.findItem(ID)
        if item not in itemList:
          itemList.append(item)
          try:
            indexList.append([self.findItemImageIndex(item,ID)])
          except:
            indexList.append([None])
        else:
          indexList[itemList.index(item)].append(self.findItemImageIndex(item,ID))
      self.undoRedoStack.add(translationAction(self,itemList,indexList,(self.distanceDragged[0],-self.distanceDragged[1])))
      self.amDragging = False

  def itemSelect(self,event):
    x0,y0 = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    x = x0/self.magnification
    y = y0/self.magnification
    self.dragStartx,self.dragStarty = x0,y0
    self.distanceDragged = (0,0)
    if self.selectedButton in [self.toolSelectButton,self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton,self.toolRotateButton]:
      self.freeMouseDown = False
    if self.selectedButton == self.toolSelectButton or (len(self.mainCanvas.find_withtag("selectedItem"))<=1 and self.selectedButton in [self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton,self.toolRotateButton]):
      self.clearSelection()
      self.setSelection(CURRENT)

  def itemToggleSelect(self,event):
    #print "control click"
    x0,y0 = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    x = x0/self.magnification
    y = y0/self.magnification
    if self.selectedButton in [self.toolSelectButton,self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton,self.toolRotateButton]:
      self.freeMouseDown = False
      self.dragStartx,self.dragStarty = x0,y0
      if "selectedItem" in self.mainCanvas.gettags(CURRENT):
        self.unSelect(CURRENT)
      else:
        self.setSelection(CURRENT)

  def itemDelete(self,event):
    if(not self.testOrAcquireLock()):
      return
    itemList = []
    self.undoRedoStack.add(endActionGroup)
    for ID in self.mainCanvas.find_withtag("selectedItem"):
      item = self.findItem(ID)
      if isinstance(item,xasyScript):
        index = self.findItemImageIndex(item,ID)
        if item not in itemList:
          itemList.append([item,[index],[item.transform[index]]])
        else:
          x = None
          for i in itemList:
            if i[0] == item:
              x = i
          x[1].append(index)
          x[2].append(item.transform[index])
      self.deleteSomething(ID)
    for entry in itemList:
      self.undoRedoStack.add(deleteScriptItemAction(self,entry[0],entry[1],entry[2]))
    self.undoRedoStack.add(beginActionGroup)
    self.clearSelection()
    self.releaseLock()

  def itemMotion(self,event):
    pass

  def itemHighlight(self,event):
    if self.selectedButton in [self.toolSelectButton] and self.editor == None:
      box = self.mainCanvas.bbox(CURRENT)
      box = (box[0]-2,box[1]-2,box[2]+2,box[3]+2)
      if len(self.mainCanvas.find_withtag("highlightBox"))==0:
        self.mainCanvas.create_rectangle(box,tags="highlightBox",width=2,outline="red")
      else:
        self.mainCanvas.tag_raise("highlightBox")
        self.mainCanvas.coords("highlightBox",*box)
      self.mainCanvas.tag_bind("highlightBox","<Leave>",self.itemUnHighlight)

  def itemUnHighlight(self,event):
    self.clearHighlight()

  def clearHighlight(self):
    self.mainCanvas.delete("highlightBox")

  def itemLeftDown(self,event):
    pass

  def itemLeftUp(self,event):
    pass

  def itemRightDown(self,event):
    pass

  def itemRightUp(self,event):
    pass

  def canvMotion(self,event):
    self.coords.config(
    text="(%.3f,%.3f)"%(self.mainCanvas.canvasx(event.x)/self.magnification,-self.mainCanvas.canvasy(event.y)/self.magnification)
    )

  def addItemToFile(self,item):
    self.fileItems.append(item)
    self.propList.insert(0,self.describeItem(item))
    self.updateCanvasSize()

  def startDraw(self,event):
    # don't start if we can't finish
    if not self.testOrAcquireLock() and not self.inDrawingMode:
      return
    x0,y0 = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    x = x0/self.magnification
    y = y0/self.magnification
    #self.mainCanvas.create_oval(x,y,x,y,width=5)
    if self.selectedButton == self.toolDrawEllipButton:
      pass
    elif self.selectedButton == self.toolFillEllipButton:
      pass
    elif self.selectedButton == self.toolTextButton:
      theText = tkSimpleDialog.askstring(title="Xasy - Text",prompt="Enter text to display:",initialvalue="",parent=self.parent)
      if theText != None and theText != "":
        theItem = xasyText(theText,(x,-y),asyPen(self.penColor,self.penWidth,self.penOptions))
        theItem.drawOnCanvas(self.mainCanvas,self.magnification)
        self.bindItemEvents(theItem)
        self.addItemToFile(theItem)
        self.undoRedoStack.add(addLabelAction(self,theItem))
      self.releaseLock()
      self.updateSelectedButton(self.toolSelectButton)
    elif self.selectedButton in [self.toolDrawLinesButton,self.toolDrawBeziButton,self.toolDrawPolyButton,self.toolDrawShapeButton,self.toolFillPolyButton,self.toolFillShapeButton]:
      self.inDrawingMode = True
      try:
        if len(self.itemBeingDrawn.path.nodeSet) == 0:
          raise Exception
        else:
          if self.selectedButton in [self.toolDrawLinesButton,self.toolDrawPolyButton,self.toolFillPolyButton]:
            self.itemBeingDrawn.appendPoint((x,-y),'--')
          else:#drawBezi,drawShape,fillShape
            self.itemBeingDrawn.appendPoint((x,-y),'..')
      except:
        path = asyPath()
        if self.selectedButton == self.toolDrawLinesButton:
          path.initFromNodeList([(x,-y),(x,-y)],['--'])
        elif self.selectedButton == self.toolDrawBeziButton:
          path.initFromNodeList([(x,-y),(x,-y)],['..'])
        elif self.selectedButton == self.toolDrawPolyButton or self.selectedButton == self.toolFillPolyButton:
          path.initFromNodeList([(x,-y),(x,-y),'cycle'],['--','--'])
        elif self.selectedButton == self.toolDrawShapeButton or self.selectedButton == self.toolFillShapeButton:
          path.initFromNodeList([(x,-y),(x,-y),'cycle'],['..','..'])
        if self.selectedButton in [self.toolDrawLinesButton,self.toolDrawBeziButton,self.toolDrawPolyButton,self.toolDrawShapeButton]:
          self.itemBeingDrawn = xasyShape(path,pen=asyPen(self.penColor,self.penWidth,self.penOptions))
        else:
          if self.penOptions.find("fillrule") != -1 or self.penOptions.find("evenodd") != -1 or self.penOptions.find("zerowinding") != -1:
            options = self.penOptions
          else:
            options = "evenodd"
          self.itemBeingDrawn = xasyFilledShape(path,pen=asyPen(self.penColor,self.penWidth,options))
        self.itemBeingDrawn.drawOnCanvas(self.mainCanvas,self.magnification)
        self.bindItemEvents(self.itemBeingDrawn)
        self.mainCanvas.bind("<Motion>",self.extendDraw)

  def extendDraw(self,event):
    x0,y0 = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    x = x0/self.magnification
    y = y0/self.magnification
    tags = self.mainCanvas.gettags("itemBeingDrawn")
    self.itemBeingDrawn.setLastPoint((x,-y))
    self.itemBeingDrawn.drawOnCanvas(self.mainCanvas,self.magnification)
    self.canvMotion(event)

  def endDraw(self,event):
    if not self.inDrawingMode or self.itemBeingDrawn == None:
      return
    x0,y0 = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    x = x0/self.magnification
    y = y0/self.magnification
    #if self.selectedButton in [self.toolDrawLinesButton,self.toolDrawPolyButton,self.toolFillPolyButton]:
      #self.itemBeingDrawn.appendPoint((x,-y),'--')
    #else:
      #self.itemBeingDrawn.appendPoint((x,-y),'..')

    #only needed for certain key bindings when startDraw is triggered right before an endDraw
    #e.g.: single click: startDraw, double click: endDraw
    self.itemBeingDrawn.removeLastPoint()
    self.itemBeingDrawn.setLastPoint((x,-y))
    self.itemBeingDrawn.drawOnCanvas(self.mainCanvas,self.magnification)
    self.addItemToFile(self.itemBeingDrawn)
    self.undoRedoStack.add(addDrawnItemAction(self,self.itemBeingDrawn))
    self.itemBeingDrawn = None
    self.mainCanvas.dtag("itemBeingDrawn","itemBeingDrawn")
    self.mainCanvas.bind("<Motion>",self.canvMotion)
    self.inDrawingMode = False
    self.releaseLock()

  def canvLeftDown(self,event):
    #print "Left Mouse Down"
    self.selectDragStart = (self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y))
    theBbox = self.mainCanvas.bbox("selectedItem")
    if theBbox != None:
      self.selectBboxMidpoint = (theBbox[0]+theBbox[2])/2.0,-(theBbox[1]+theBbox[3])/2.0
    if self.freeMouseDown and self.editor != None:
      self.editor.endEdit()
      if self.editor.modified:
        self.undoRedoStack.add(editDrawnItemAction(self,self.itemBeingEdited,copy.deepcopy(self.editor.shape),self.fileItems.index(self.editor.shape)))
      self.editor = None
    elif self.selectedButton in (self.toolSelectButton,self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton,self.toolRotateButton):
      if self.freeMouseDown:
        self.clearSelection()
        self.dragSelecting = False
    else:
      self.startDraw(event)

  def canvLeftUp(self,event):
    #print "Left Mouse Up"
    # if we're busy, ignore it
    if not self.testOrAcquireLock():
      return
    self.freeMouseDown = True
    if self.inRotatingMode:
      for item in self.itemsBeingRotated:
        item.drawOnCanvas(self.mainCanvas,self.magnification)
        self.bindItemEvents(item)
      self.updateSelection()
      self.itemsBeingRotated = []
      self.inRotatingMode = False
    if self.dragSelecting:
      self.hideSelectionBox()
      self.dragSelecting = False
      self.mainCanvas.addtag_enclosed("enclosed",self.selectDragStart[0],self.selectDragStart[1],self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y))
      for item in self.mainCanvas.find_withtag("enclosed"):
        tags = self.mainCanvas.gettags(item)
        if "drawn" not in tags and "image" not in tags:
          self.mainCanvas.dtag(item,"enclosed")
      self.mainCanvas.addtag_withtag("selectedItem","enclosed")
      self.mainCanvas.dtag("enclosed","enclosed")
      if self.selectedButton == self.toolSelectButton and len(self.mainCanvas.find_withtag("selectedItem")) > 0:
        self.updateSelectedButton(self.toolMoveButton)
      self.updateSelection()
    self.releaseLock()

  def canvDrag(self,event):
    x0,y0 = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    x = x0/self.magnification
    y = y0/self.magnification
    if self.selectedButton == self.toolSelectButton and self.editor == None:
      self.mainCanvas.coords("outlineBox",self.selectDragStart[0],self.selectDragStart[1],x0,y0)
      self.showSelectionBox()
      self.dragSelecting = True
    elif self.selectedButton == self.toolRotateButton and self.editor == None:
      bbox = self.mainCanvas.bbox("selectedItem")
      if bbox != None:
        p1 = self.selectDragStart[0]-self.selectBboxMidpoint[0],-self.selectDragStart[1]-self.selectBboxMidpoint[1]
        mp1 = math.sqrt(p1[0]**2+p1[1]**2)
        p2 = x0-self.selectBboxMidpoint[0],-y0-self.selectBboxMidpoint[1]
        mp2 = math.sqrt(p2[0]**2+p2[1]**2)
        if mp1 != 0:
          t1 = math.acos(p1[0]/mp1)
          if p1[1] < 0:
            t1 *= -1
        else:
          t1 = 0
        if mp2 != 0:
          t2 = math.acos(p2[0]/mp2)
          if p2[1] < 0:
            t2 *= -1
        else:
          t2 = 0
        theta = t2-t1
        self.selectDragStart = x0,y0
        self.itemsBeingRotated = []
        for ID in self.mainCanvas.find_withtag("selectedItem"):
          self.rotateSomething(ID,theta,self.selectBboxMidpoint)
          item = self.findItem(ID)
          if not item in self.itemsBeingRotated:
            self.itemsBeingRotated.append(item)
        self.updateSelection()
        self.updateCanvasSize()
        if not self.inRotatingMode:
          self.currentRotationAngle = theta
          IDList = self.mainCanvas.find_withtag("selectedItem")
          itemList = []
          indexList = []
          for ID in IDList:
            item = self.findItem(ID)
            if item not in itemList:
              itemList.append(item)
              try:
                indexList.append([self.findItemImageIndex(item,ID)])
              except:
                indexList.append([None])
            else:
              indexList[itemList.index(item)].append(self.findItemImageIndex(item,ID))
          self.undoRedoStack.add(rotationAction(self,itemList,indexList,self.currentRotationAngle,self.selectBboxMidpoint))
          self.inRotatingMode = True
        else:
          self.currentRotationAngle += theta
          self.undoRedoStack.undoStack[-1].angle = self.currentRotationAngle

  def canvEnter(self,event):
    self.freeMouseDown = True
    event.widget.focus_set()

  def canvLeave(self,event):
    self.freeMouseDown = False

  def canvRightDown(self,event):
    pass
    #print "Right Mouse Down"

  def canvRightUp(self,event):
    pass
    #print "Right Mouse Up"

  def configEvt(self,event):
    self.updateCanvasSize()
    self.sizePane()

  def sizePane(self):
    width = self.windowPane.winfo_width()-10
    cwidth = min(int(0.87*self.windowPane.winfo_width()),width-75)
    if self.paneVisible:
      self.windowPane.paneconfigure(self.canvFrame,minsize=cwidth)
    else:
      self.windowPane.paneconfigure(self.canvFrame,minsize=width)
    self.windowPane.paneconfigure(self.propFrame,minsize=75)

  def togglePaneEvt(self,event):
    self.paneVisible = not self.paneVisible
    self.sizePane()

  def popupDelete(self):
    self.deleteItem(self.itemPopupMenu.item)

  def popupEdit(self):
    self.itemEdit(self.itemPopupMenu.item)

  def popupViewCode(self):
    tkMessageBox.showinfo("Item Code",self.itemPopupMenu.item.getCode())

  def popupClearTransform(self):
    self.undoRedoStack.add(clearItemTransformsAction(self,self.itemPopupMenu.item,copy.deepcopy(self.itemPopupMenu.item.transform)))
    if isinstance(self.itemPopupMenu.item,xasyScript) or isinstance(self.itemPopupMenu.item,xasyText):
      for i in range(len(self.itemPopupMenu.item.transform)):
        self.itemPopupMenu.item.transform[i] = identity()
    else:
      self.itemPopupMenu.item.transform = [identity()]
    self.popupRedrawItem()

  def popupRedrawItem(self):
    if not self.testOrAcquireLock():
      return
    self.clearSelection()
    self.clearHighlight()
    self.itemPopupMenu.item.drawOnCanvas(self.mainCanvas,self.magnification)
    self.bindItemEvents(self.itemPopupMenu.item)
    self.updateCanvasSize()
    self.releaseLock()

  def hidePopupMenu(self):
    try:
      self.itemPopupMenu.unpost()
    except:
      pass

  def itemMenuPopup(self,parent,item,x,y):
    self.hidePopupMenu()
    self.itemPopupMenu = Menu(parent,tearoff=False)
    self.itemPopupMenu.add_command(label="Edit",command=self.popupEdit)
    self.itemPopupMenu.add_command(label="Clear Transforms",command=self.popupClearTransform)
    self.itemPopupMenu.add_command(label="Redraw",command=self.popupRedrawItem)
    self.itemPopupMenu.add_command(label="View code",command=self.popupViewCode)
    self.itemPopupMenu.add_separator()
    self.itemPopupMenu.add_command(label="Delete",command=self.popupDelete)
    self.itemPopupMenu.item = item
    #self.itemPopupMenu.bind("<Leave>",lambda a:self.itemPopupMenu.unpost())
    #self.itemPopupMenu.bind("<FocusOut>",lambda a:self.itemPopupMenu.unpost())
    self.itemPopupMenu.post(x,y)

  def itemPropMenuPopup(self,event):
    try:
      item = self.fileItems[len(self.fileItems)-int(self.propList.curselection()[0])-1]
      self.itemMenuPopup(self.propList,item,event.x_root,event.y_root)
    except:
      pass

  def itemCanvasMenuPopup(self,event):
    if self.selectedButton in (self.toolSelectButton,self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton,self.toolRotateButton):
      try:
        item = self.findItem(self.mainCanvas.find_withtag(CURRENT)[0])
      except:
        item = None
      if item != None:
        self.itemMenuPopup(self.mainCanvas,item,event.x_root,event.y_root)

  def editOptions(self):
    if(not self.testOrAcquireLock()):
      return
    self.releaseLock()
    xasyOptionsDialog.xasyOptionsDlg(self.parent)
    self.applyOptions()

  def resetOptions(self):
    xasyOptions.setDefaults()
    self.applyOptions()

  def applyPenWidth(self):
    self.pendingPenWidthChange = None
    if self.validatePenWidth():
      old = self.penWidth
      self.penWidth = float(self.penWidthEntry.get())
      if old != self.penWidth:
        self.showCurrentPen()

  def validatePenWidth(self):
    text = self.penWidthEntry.get()
    try:
      width = float(text)
      if width <= 0:
        return False
      else:
        return True
    except:
      return False

  def showCurrentPen(self):
    mag = 1
    width = self.penWidth
    while width > 10:
      width /= 2
      mag *= 2
    self.penDisp.itemconfigure("penDisp",width=width,fill=self.tkPenColor)
    self.penDisp.itemconfigure("penMag",text="x%d"%mag)
    #apply the new pen to any selected items
    IDs = self.mainCanvas.find_withtag("selectedItem")
    madeAChange = False
    for ID in IDs:
      item = self.findItem(ID)
      if not isinstance(item,xasyScript):
        if not madeAChange:
          self.undoRedoStack.add(endActionGroup)
          madeAChange = True
        if isinstance(item,xasyText):
          temp = item.label.pen
          item.label.pen = asyPen(self.penColor,self.penWidth,self.penOptions)
          item.drawOnCanvas(self.mainCanvas,self.magnification)
          self.bindItemEvents(item)
          self.setSelection(item.imageList[0].IDTag)
          self.undoRedoStack.add(editLabelPenAction(self,temp,asyPen(self.penColor,self.penWidth,self.penOptions),self.fileItems.index(item)))
        else:
          temp = copy.deepcopy(item)
          item.pen = asyPen(self.penColor,self.penWidth,self.penOptions)
          item.drawOnCanvas(self.mainCanvas,self.magnification)
          self.undoRedoStack.add(editDrawnItemAction(self,temp,copy.deepcopy(item),self.fileItems.index(item)))
    if madeAChange:
      self.undoRedoStack.add(beginActionGroup)

  def applyPenWidthEvt(self,event):
    if not self.testOrAcquireLock():
      return
    self.applyPenWidth()
    self.releaseLock()

  def penWidthChanged(self,event):
    if self.pendingPenWidthChange is not None:
      self.penWidthEntry.after_cancel(self.pendingPenWidthChange)
    self.pendingPenWidthChange = self.penWidthEntry.after(1000,self.applyPenWidth)

  def applyPenOptEvt(self,event):
    if not self.testOrAcquireLock():
      return
    self.applyPenOpt()
    self.releaseLock()

  def validatePenOpt(self):
    try:
      penTest = asyPen(self.penColor,self.penWidth,self.penOptEntry.get())
      return True
    except:
      self.penOptEntry.select_range(0,END)
      self.penOptEntry.delete(0,END)
      self.penOptEntry.insert(END,"Invalid Pen Options")
      self.penOptEntry.after(5000,self.clearInvalidOptEntry)
      self.penOptions = ""
      return False

  def clearInvalidOptEntry(self):
    self.penOptEntry.select_range(0,END)
    self.penOptEntry.delete(0,END)

  def applyPenOpt(self):
    if self.validatePenOpt():
      old = self.penOptions
      self.penOptions = self.penOptEntry.get()
      if old != self.penOptions:
        self.showCurrentPen()

  def undoOperation(self):
    self.undoRedoStack.undo()

  def redoOperation(self):
    self.undoRedoStack.redo()

  def resetStacking(self):
    for item in self.fileItems:
      self.raiseSomething(item,force=True)
