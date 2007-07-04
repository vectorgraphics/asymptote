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

from Tkinter import *
import tkMessageBox
import tkFileDialog
import tkSimpleDialog
import threading
import time

from xasyCodeEditor import *
from xasy2asy import *
import xasyFile
import xasyOptions
import xasyOptionsDialog
import CubicBezier
from xasyBezierEditor import xasyBezierEditor
from xasyGUIIcons import iconB64
from xasyColorPicker import *
try:
  from PIL import ImageTk
  import Image
  PILAvailable = True
except:
  PILAvailable = False

class xasyMainWin:
  def __init__(self,master,file=None):
    self.parent = master
    self.createWidgets()
    self.resetGUI()
    if file != None:
      self.loadFile(file)
    self.ticker = threading.Thread(target=self.tickHandler)
    self.ticker.start()

  def tickHandler(self):
    while not(self.quitting):
      self.tickCount += 1
      self.mainCanvas.itemconfigure("outlineBox",dashoffset=self.tickCount%9)
      time.sleep(0.1)

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
    self.fileMenu = Menu(self.mainMenu)
    self.fileMenu.add_command(label="New",command=self.fileNewCmd)
    self.fileMenu.add_command(label="Open",command=self.fileOpenCmd)
    self.fileMenu.add_separator()
    self.fileMenu.add_command(label="Save",command=self.fileSaveCmd)
    self.fileMenu.add_command(label="Save As",command=self.fileSaveAsCmd)
    self.fileMenu.add_separator()

    #an export menu
    self.exportMenu = Menu(self.fileMenu)
    self.exportMenu.add_command(label="EPS...",command=self.exportEPS)
    self.exportMenu.add_command(label="PDF...",command=self.exportPDF)
    self.exportMenu.add_command(label="GIF...",command=self.exportGIF)
    self.exportMenu.add_command(label="PNG...",command=self.exportPNG)
    self.fileMenu.add_cascade(label="Export",menu=self.exportMenu)
    self.fileMenu.add_separator()

    self.fileMenu.add_command(label="Quit",command=self.fileExitCmd)

    self.mainMenu.add_cascade(label="File",menu=self.fileMenu)

    #the edit menu
    self.editMenu = Menu(self.mainMenu)
    self.editMenu.add_command(label="Undo",command=self.editUndoCmd,state=DISABLED)
    self.editMenu.add_command(label="Redo",command=self.editRedoCmd,state=DISABLED)
    self.mainMenu.add_cascade(label="Edit",menu=self.editMenu)

    #the tools menu
    self.toolsMenu = Menu(self.mainMenu)
    self.mainMenu.add_cascade(label="Tools",menu=self.toolsMenu)

    #the options menu
    self.optionsMenu = Menu(self.toolsMenu)
    self.toolsMenu.add_cascade(label="Options",menu=self.optionsMenu)
    self.optionsMenu.add_command(label="Edit...",command=self.editOptions)
    self.optionsMenu.add_command(label="Reset defaults",command=self.resetOptions)

    #the help menu
    self.helpMenu = Menu(self.mainMenu)
    self.helpMenu.add_command(label="Help",command=self.helpHelpCmd,state=DISABLED)
    self.helpMenu.add_command(label="Asymptote Documentation",command=self.helpAsyDocCmd)
    self.helpMenu.add_separator()
    self.helpMenu.add_command(label="About xasy",command=self.helpAboutCmd)
    self.mainMenu.add_cascade(label="Help",menu=self.helpMenu)

    #status bar
    self.statusBar = Frame(self.parent,relief=FLAT)

    Label(self.statusBar,text="+").pack(side=RIGHT)
    self.zoomBar = Scale(self.statusBar,orient=HORIZONTAL,length=150,width=10,from_=-5,to=5,command=self.zoomViewCmd,showvalue=False,state=DISABLED,relief=FLAT)
    self.zoomBar.pack(side=RIGHT,fill=X)
    Label(self.statusBar,text="Zoom: - ").pack(side=RIGHT)
    self.coords = Label(self.statusBar,text="(0,0)",relief=SUNKEN,anchor=W)
    self.coords.pack(side=RIGHT)
    self.status = Label(self.statusBar,text="Ready",relief=SUNKEN,anchor=W)
    self.status.pack(side=RIGHT,fill=X,expand=1)

    self.statusBar.pack(side=BOTTOM,fill=X)

    #toolbar for transformation, drawing, and adjustment commands
    self.toolBar = Frame(self.parent,relief=FLAT)

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
    self.toolRotateButton = Button(self.toolBar,command=self.toolRotateCmd,image=self.toolIcons["rotate"],state=DISABLED,relief=FLAT)
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
    self.toolDrawEllipButton.grid(row=7,column=0,sticky=N+S+E+W)
    self.toolFillEllipButton = Button(self.toolBar,command=self.toolFillEllipCmd,image=self.toolIcons["fillEllip"],state=DISABLED,relief=FLAT)
    self.toolFillEllipButton.grid(row=7,column=1,sticky=N+S+E+W)
    self.toolDrawShapeButton = Button(self.toolBar,command=self.toolDrawShapeCmd,image=self.toolIcons["drawShape"])
    self.toolDrawShapeButton.grid(row=8,column=0,sticky=N+S+E+W)
    self.toolFillShapeButton = Button(self.toolBar,command=self.toolFillShapeCmd,image=self.toolIcons["fillShape"])
    self.toolFillShapeButton.grid(row=8,column=1,sticky=N+S+E+W)
    self.toolTextButton = Button(self.toolBar,command=self.toolTextCmd,image=self.toolIcons["text"])
    self.toolTextButton.grid(row=9,column=0,sticky=N+S+E+W)		


    self.adjLbl = Label(self.toolBar,text="",anchor=W)
    self.adjLbl.grid(row=10,column=0,columnspan=2,sticky=W)
    self.toolRaiseButton = Button(self.toolBar,command=self.toolRaiseCmd,image=self.toolIcons["raise"],state=DISABLED,relief=FLAT)
    self.toolRaiseButton.grid(row=11,column=0,sticky=N+S+E+W)
    self.toolLowerButton = Button(self.toolBar,command=self.toolLowerCmd,image=self.toolIcons["lower"],state=DISABLED,relief=FLAT)
    self.toolLowerButton.grid(row=11,column=1,sticky=N+S+E+W)

    Label(self.toolBar,text="",anchor=W).grid(row=11,column=0,columnspan=2,sticky=W)
    self.toolAsyButton = Button(self.toolBar,command=self.toolAsyCmd,image=self.toolIcons["asy"])
    self.toolAsyButton.grid(row=12,column=0,sticky=N+S+E+W)

    self.toolBar.pack(side=LEFT,anchor=NW)

    #documentation for the tool bar buttons
    self.toolDocs = {
      self.toolSelectButton : "Click an item to select it. Control-Click will select/deselect additional items. Use mouse scroller (or Up/Down keys) to raise/lower highlighted items.",
      self.toolMoveButton : "Drag a selected item.",
      self.toolHorizMoveButton : "Drag a selected item. Only horizontal translation will be applied.",
      self.toolVertiMoveButton : "Drag a selected item. Only vertical translation will be applied.",
      self.toolRotateButton : "Drag a selected item to rotate. (Not yet implemented)",
      self.toolDrawLinesButton : "Click to draw line segments. Double click to place last point.",
      self.toolDrawBeziButton : "Click to place points. Double click to place last point.",
      self.toolDrawPolyButton : "Click to place vertices. Double click to place last point.",
      self.toolFillPolyButton : "Click to place vertices. Double click to place last point.",
      self.toolDrawEllipButton : "(UNIMPLEMENTED)Click to place center. Move mouse to achieve correct shape and double click.",
      self.toolFillEllipButton : "(UNIMPLEMENTED)Click to place center. Move mouse to achieve correct shape and double click.",
      self.toolDrawShapeButton : "Click to place points. Double click to place last point.",
      self.toolFillShapeButton : "Click to place points. Double click to place last point.",
      self.toolTextButton : "Click location of top left label position and enter text in dialog.",
      self.toolRaiseButton : "(UNIMPLEMENTED)Raise selected items to top.",
      self.toolLowerButton : "(UNIMPLEMENTED)Lower selected items to bottom.",
      self.toolAsyButton : "Insert/Edit Asymptote code."
    }

    #an options bar
    self.optionsBar = Frame(self.parent,relief=FLAT,height=100)
    Label(self.optionsBar,text="Current Pen",anchor=W).grid(row=0,column=0)
    Label(self.optionsBar,text="Color",anchor=E).grid(row=1,column=0)
    self.penColButton = Button(self.optionsBar,text=" ",width=5,bg="black",activebackground="black",command=self.setPenColCmd)
    self.penColButton.grid(row=1,column=1)
    self.optionsBar.pack(side=BOTTOM,anchor=NW)

    #a paned window for the canvas and propert explorer
    self.windowPane = PanedWindow(self.parent)

    #a property explorer
    self.propFrame = Frame(self.parent)
    #self.propFrame.pack(side=RIGHT,fill=Y,expand=False)
    self.propFrame.rowconfigure(1,weight=1)
    self.propFrame.columnconfigure(0,weight=1)
    Label(self.propFrame,text="Item List").grid(row=0,column=0,columnspan=2)
    self.itemScroll = Scrollbar(self.propFrame,orient=VERTICAL)
    #self.itemHScroll = Scrollbar(self.propFrame,orient=HORIZONTAL)
    self.propList = Listbox(self.propFrame, yscrollcommand=self.itemScroll.set)#, xscrollcommand=self.itemHScroll.set)
    self.itemScroll.config(command=self.propList.yview)
    #self.itemHScroll.config(command=self.propList.xview)
    self.itemScroll.grid(row=1,column=1,sticky=N+S)
    #self.itemHScroll.grid(row=2,column=0,stick=E+W)
    self.propList.grid(row=1,column=0,sticky=N+S+E+W)
    self.propList.bind("<Double-Button-1>",self.propSelect)
    self.propList.bind("<Button-3>",self.itemPropMenuPopup)

    #the canvas's frame
    self.canvFrame = Frame(self.parent)
    #self.canvFrame.pack(side=RIGHT,fill=BOTH,expand=True)
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
    self.mainCanvas = Canvas(self.canvFrame,relief=SUNKEN,background="white",borderwidth=0,
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


  def resetGUI(self):
    #set up the main window
    self.filename = None
    self.fileToOpen = None
    self.retitle()

    #set up the paned window
    self.paneVisible = True

    #load configuration
    xasyOptions.load()
    self.applyOptions()

    #set up editing
    self.editor = None

    #set up the toolbar
    try:
      self.updateSelectedButton(self.toolSelectButton)
    except:
      self.selectedButton = self.toolSelectButton
      self.updateSelectedButton(self.toolSelectButton)

    #set up the canvas
    self.mainCanvas.delete(ALL)
    self.mainCanvas.create_rectangle(0,0,0,0,tags="outlineBox",width=0,outline="#801111",dash=(3,6))
    self.backColor = "white" #in future, load this from an options file
    self.mainCanvas.configure(background=self.backColor)

    #set up drawing
    self.pathInProgress = asyPath()
    self.currentIDTag = -1
    self.inDrawingMode = False
    self.freeMouseDown = True
    self.dragSelecting = False

    #set up the xasy item list
    self.fileItems = []
    self.propList.delete(0,END)
    self.updateCanvasSize()

    #setup timer
    self.quitting = False
    self.tickCount = 0

    self.magnification = 1

  def retitle(self):
    if self.filename == None:
      self.parent.title("Xasy - New File")
    else:
      name = os.path.abspath(self.filename)
      name = os.path.basename(name)
      self.parent.title("Xasy - %s"%name)

  def applyOptions(self):
    self.tkPenColor = xasyOptions.options['defPenColor']
    self.penColor = makeRGBfromTkColor(self.tkPenColor)
    self.penColButton.config(bg=self.tkPenColor,activebackground=self.tkPenColor)
    self.penWidth = xasyOptions.options['defPenWidth']
    self.penOptions = xasyOptions.options['defPenOptions']
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

  def drawGrid(self):
    self.mainCanvas.delete("grid")
    if not self.gridVisible:
      return
    left,top,right,bottom = map(int,self.mainCanvas.cget("scrollregion").split())
    for i in range(0,right,self.gridxspace):
      self.mainCanvas.create_line(i,top,i,bottom,tags=("grid","vertical"),fill=self.gridcolor)
    for i in range(-self.gridxspace,left,-self.gridxspace):
      self.mainCanvas.create_line(i,top,i,bottom,tags=("grid","vertical"),fill=self.gridcolor)
    for i in range(-self.gridyspace,top,-self.gridyspace):
      self.mainCanvas.create_line(left,i,right,i,tags=("grid","horizontal"),fill=self.gridcolor)
    for i in range(0,bottom,self.gridyspace):
      self.mainCanvas.create_line(left,i,right,i,tags=("grid","horizontal"),fill=self.gridcolor)
    self.mainCanvas.tag_lower("grid")

  def drawAxes(self):
    self.mainCanvas.delete("axes")
    if not self.axesVisible:
      return
    left,top,right,bottom = map(int,self.mainCanvas.cget("scrollregion").split())
    self.mainCanvas.create_line(0,top,0,bottom,tags=("axes","yaxis"),fill=self.axiscolor)
    self.mainCanvas.create_line(left,0,right,0,tags=("axes","xaxis"),fill=self.axiscolor)
    self.mainCanvas.create_rectangle(left,top,right,bottom,tags=("axes","scrolloutline"),outline=self.axiscolor,fill="")
    for i in range(10,right,10):
      self.mainCanvas.create_line(i,-5,i,5,tags=("axes","xaxis-ticks"),fill=self.tickcolor)
    for i in range(-10,left,-10):
      self.mainCanvas.create_line(i,-5,i,5,tags=("axes","xaxis-ticks"),fill=self.tickcolor)
    for i in range(-10,top,-10):
      self.mainCanvas.create_line(-5,i,5,i,tags=("axes","yaxis-ticks"),fill=self.tickcolor)
    for i in range(10,bottom,10):
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
    self.mainCanvas.tag_bind(tagorID,"<Control-Button-1>",self.itemToggleSelect)
    self.mainCanvas.tag_bind(tagorID,"<Button-1>",self.itemSelect)
    #self.mainCanvas.tag_bind(tagorID,"<Button-1>",self.itemMouseDown)
    #self.mainCanvas.tag_bind(tagorID,"<ButtonRelease-1>",self.itemMouseUp)
    self.mainCanvas.tag_bind(tagorID,"<Double-Button-1>",self.itemEditEvt)
    self.mainCanvas.tag_bind(tagorID,"<B1-Motion>",self.itemDrag)
    self.mainCanvas.tag_bind(tagorID,"<Delete>",self.itemDelete)
    self.mainCanvas.tag_bind(tagorID,"<Enter>",self.itemHighlight)
    self.mainCanvas.tag_bind(tagorID,"<Button-3>",self.itemCanvasMenuPopup)

  def bindItemEvents(self,item):
    if isinstance(item,xasyScript):
      for image in item.imageList:
        self.bindEvents(image.IDTag)
    else:
      self.bindEvents(item.IDTag)

  def canQuit(self):
    #print "Quitting"
    self.quitting = True
    self.ticker.join()
    asy.quit()
    self.parent.destroy()

  def openFile(self,name):
    self.resetGUI()
    self.loadFile(name)

  def loadFile(self,name):
    self.status.config(text="Loading "+name)
    fullName = os.path.abspath(name)
    self.filename = fullName
    fileName = os.path.basename(fullName)
    fileDir = os.path.dirname(fullName)
    os.chdir(fileDir)
    self.filePrefix,none = os.path.splitext(fileName)
    #print "opening: full:" + fullName + " file:"+fileName+" dir:"+fileDir+" pref:"+self.filePrefix
    self.retitle()
    try:
      f = open(fullName,'rt')
      self.fileItems = xasyFile.parseFile(f)
      f.close()
    except IOError:
      raise
      tkMessageBox.showerror("File Opening Failed.","File coult not be opened.")
      self.fileItems = []
    except:
      self.fileItems = []
      self.autoMakeScript = True
      if self.autoMakeScript or tkMessageBox.askyesno("Error Opening File", "File was not recognized as an xasy file.\nLoad as a script item?"):
        try:
          item = xasyScript()
          f.seek(0)
          item.setScript(f.read())
          self.addItemToFile(item)
        except:
          raise
          tkMessageBox.showerror("File Opening Failed.","Could not load as a script item.")
          self.fileItems = []
    self.populateCanvasWithItems()
    self.populatePropertyList()

  def populateCanvasWithItems(self):
    self.mainCanvas.delete("drawn || image")
    self.itemCount = 0
    for item in self.fileItems:
      item.drawOnCanvas(self.mainCanvas);
      self.bindItemEvents(item)

  def swapFileItems(self,index1,index2):
    temp = self.fileItems[index1]
    self.fileItems[index1] = self.fileItems[index2]
    self.fileItems[index2] = temp

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
      self.propList.insert(END,self.describeItem(item))

  def saveFile(self,name):
    f = open(name,"wt")
    xasyFile.saveFile(f,self.fileItems)
    f.close()
    self.retitle()

  #menu commands
  def fileNewCmd(self):
    #print "Create New File"
    self.resetGUI()

  def fileOpenCmd(self):
    #print "Open a file"
    file=tkFileDialog.askopenfile(filetypes=[("asy GUI files","*.asy"),("All files","*")],title="Open File",parent=self.parent)
    if file != None:
      self.filename = file.name
      file.close()
      self.openFile(self.filename)

  def fileSaveCmd(self):
    #print "Save current file"
    if self.filename == None:
      file=tkFileDialog.asksaveasfile(defaultextension=".asy",filetypes=[("asy GUI files","*.asy")],initialfile="newDrawing.asy",parent=self.parent,title="Save File")
      if file != None:
        self.filename = file.name
        file.close()
    if self.filename != None:
      self.saveFile(self.filename)

  def fileSaveAsCmd(self):
    #print "Save current file as"
    file=tkFileDialog.asksaveasfile(defaultextension=".asy",filetypes=[("asy GUI files","*.asy")],initialfile="newDrawing.asy",parent=self.parent,title="Save File")
    if file != None:
      self.filename = file.name
      file.close()
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

  def exportFile(self,inFile, outFormat):
    if inFile == None:
      if tkMessageBox.askyesno("File has not been saved.","Save?"):
        self.fileSaveAsCmd()
    else:
      self.fileSaveCmd()
    outfile = tkFileDialog.asksaveasfile(defaultextension = '.'+outFormat,filetypes=[(outFormat+" files","*."+outFormat)],initialfile="out."+outFormat,parent=self.parent,title="Choose output file")
    if not outfile:
      return
    fullname = os.path.abspath(outfile.name)
    outName = os.path.basename(outfile.name)
    dirname = os.path.dirname(fullname)
    outfile.close()
    os.chdir(dirname)
    command = "asy -f %s -o %s %s"%(outFormat,outName,inFile)
    print command
    saver = subprocess.Popen(split(command),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    saver.wait()
    if saver.returncode != 0:
      tkMessageBox.showerror("Error Exporting File",saver.stdout.read()+saver.stderr.read())
    else:
      tkMessageBox.showinfo("Export","File Exported Successfully")

  def fileExitCmd(self):
    #print "Exit xasy"
    self.canQuit()

  def editUndoCmd(self):
    print "Undo"

  def editRedoCmd(self):
    print "Redo"

  def toolsOptionsCmd(self):
    print "Display options dialog"

  def helpHelpCmd(self):
    print "Get help on xasy"

  def helpAsyDocCmd(self):
    #print "Open documentation about Asymptote"
    asy.execute("help;")

  def helpAboutCmd(self):
    tkMessageBox.showinfo("About xasy","A graphical interface for Asymptote")

  def updateSelectedButton(self,newB):
    self.selectedButton.config(relief = RAISED)
    if newB == self.toolSelectButton or self.selectedButton == self.toolSelectButton:
      self.mainCanvas.delete("highlightBox")
    if self.editor != None:
      self.editor.endEdit()
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
    self.updateSelectedButton(self.toolAsyButton)
  def toolRaiseCmd(self):
    self.mainCanvas.tag_raise("selectedItem")
  def toolLowerCmd(self):
    self.mainCanvas.tag_lower("selectedItem")
  def itemRaise(self,event):
    self.mainCanvas.tag_raise(CURRENT)
  def itemLower(self,event):
    self.mainCanvas.tag_lower(CURRENT)

  #options bar commands
  def setPenColCmd(self):
    self.penColor = xasyColorDlg(self.parent).getColor(self.penColor)
    self.tkPenColor = RGB255hex(RGBreal255(self.penColor))
    self.penColButton.config(bg=self.tkPenColor,activebackground=self.tkPenColor)

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

  def unSelect(self,what):
    self.mainCanvas.dtag(what,"selectedItem")
    self.updateSelection()

  def updateSelection(self):
    self.clearHighlight()
    theBbox = self.mainCanvas.bbox("selectedItem")
    if theBbox != None:
      self.mainCanvas.coords("outlineBox",self.mainCanvas.bbox("selectedItem"))
      self.showSelectionBox()
    else:
      self.clearSelection()

  #event handlers
  def zoomViewCmd(self,where):
    pass
    #print "Zooming the view!"

  def selectItem(self,item):
    self.clearSelection()
    if isinstance(item,xasyScript):
      for image in item.imageList:
        self.setSelection(image.IDTag)
    else:
      self.setSelection(item.IDTag)

  def propSelect(self,event):
    items = map(int, self.propList.curselection())
    if len(items)>0:
      try:
        self.selectItem(self.fileItems[items[0]])
      except:
        pass

  def findItem(self,ID):
    for item in self.fileItems:
      if isinstance(item,xasyScript):
        for image in item.imageList:
          if image.IDTag == ID:
            return item
      else:
        if item.IDTag == ID:
          return item
    return None

  def findItemImageIndex(self,item,ID):
    count = 0
    for image in item.imageList:
      if image.IDTag == ID:
        return count
      else:
        count += 1
    return None

  def transformSomething(self,ID,transform):
    item = self.findItem(ID)
    if item == None:
      raise Exception,"fileList is corrupt!!!"
    if isinstance(item,xasyScript):
      index = self.findItemImageIndex(item,ID)
      if index == None:
        raise Exception,"imageList is corrupt!!!"
      else:
        try:
          original = item.transform[index]
        except:
          original = identity
        item.transform[index] = transform*original
    else:
      item.transform = transform*item.transform

  def deleteItem(self,item):
    if isinstance(item,xasyScript):
      for image in item.imageList:
        self.mainCanvas.delete(image.IDTag)
    else:
      self.mainCanvas.delete(item.IDTag)
    self.fileItems.remove(item)
    self.populatePropertyList()
    self.clearSelection()

  def deleteSomething(self,ID):
    self.clearSelection()
    self.clearHighlight()
    if self.editor != None:
      self.editor.endEdit()
    item = self.findItem(ID)
    if item == None:
      raise Exception,"fileList is corrupt!!!"
    if isinstance(item,xasyScript):
      index = self.findItemImageIndex(item,ID)
      if index == None:
        raise Exception,"imageList is corrupt!!!"
      else:
        item.transform[index] = asyTransform((0,0,0,0,0,0))
    else:
      self.fileItems.remove(item)
    self.mainCanvas.delete(ID)
    self.populatePropertyList()
    self.clearSelection()

  def itemEdit(self,item):
    self.updateSelectedButton(self.toolSelectButton)
    if isinstance(item,xasyScript):
      tl = Toplevel()
      xasyCodeEditor(tl,item.script,item.setScript)
      self.parent.wait_window(tl)
      item.drawOnCanvas(self.mainCanvas)
      self.bindItemEvents(item)
    elif isinstance(item,xasyText):
      theText = tkSimpleDialog.askstring(title="Xasy - Text",prompt="Enter text to display:",initialvalue=item.label.text,parent=self.parent)		
      if theText != None and theText != "":
        item.label.text = theText
        item.drawOnCanvas(self.mainCanvas) 
    elif isinstance(item,xasyShape):
      self.clearSelection()
      self.clearHighlight()
      self.editor = xasyBezierEditor(self,item,self.mainCanvas)

  def itemEditEvt(self,event):
    if not self.inDrawingMode:
      ID = self.mainCanvas.find_withtag(CURRENT)[0]
      item = self.findItem(ID)
      self.itemEdit(item)

  def itemDrag(self,event):
    x,y = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    if self.selectedButton not in [self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton]:
      return
    if "selectedItem" in self.mainCanvas.gettags(CURRENT):
      for ID in self.mainCanvas.find_withtag("selectedItem"):
        transform = identity
        if self.selectedButton == self.toolMoveButton:
          self.mainCanvas.move(ID,x-self.dragStartx,y-self.dragStarty)
          transform = asyTransform((x-self.dragStartx,-(y-self.dragStarty),1,0,0,1))
        elif self.selectedButton == self.toolVertiMoveButton:
          self.mainCanvas.move(ID,0,y-self.dragStarty)
          transform = asyTransform((0,-(y-self.dragStarty),1,0,0,1))
        elif self.selectedButton == self.toolHorizMoveButton:
          self.mainCanvas.move(ID,x-self.dragStartx,0)
          transform = asyTransform((x-self.dragStartx,0,1,0,0,1))
        self.transformSomething(ID,transform)
      self.updateSelection()
    self.updateCanvasSize()
    self.dragStartx,self.dragStarty = x,y

  def itemMouseUp(self,event):
    self.freeMouseDown = True

  def itemMouseDown(self,event):
    x,y = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    self.dragStartx,self.dragStarty = x,y
    self.freeMouseDown = False

  def itemSelect(self,event):
    x,y = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    self.dragStartx,self.dragStarty = x,y
    if self.selectedButton in (self.toolSelectButton,self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton):
      self.freeMouseDown = False
    if self.selectedButton == self.toolSelectButton:
      #print "selecting CURRENT"
      self.clearSelection()
      self.setSelection(CURRENT)

  def itemToggleSelect(self,event):
    #print "control click"
    x,y = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    if self.selectedButton in (self.toolSelectButton,self.toolMoveButton,self.toolVertiMoveButton,self.toolHorizMoveButton):
      self.freeMouseDown = False
    if self.selectedButton == self.toolSelectButton:
      self.dragStartx,self.dragStarty = x,y
      if "selectedItem" in self.mainCanvas.gettags(CURRENT):
        self.unSelect(CURRENT)
      else:
        self.setSelection(CURRENT)

  def itemDelete(self,event):
    for ID in self.mainCanvas.find_withtag("selectedItem"):
      self.deleteSomething(ID)
    self.clearSelection()

  def itemMotion(self,event):
    pass

  def itemHighlight(self,event):
    if self.selectedButton == self.toolSelectButton and self.editor == None:
      box = self.mainCanvas.bbox(CURRENT)
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
    self.coords.config(text=str((self.mainCanvas.canvasx(event.x),-self.mainCanvas.canvasy(event.y))))

  def addItemToFile(self,item):
    self.fileItems.append(item)
    self.propList.insert(END,self.describeItem(item))
    self.updateCanvasSize()

  def startDraw(self,event):
    x,y = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    #self.mainCanvas.create_oval(x,y,x,y,width=5)
    if self.selectedButton == self.toolDrawEllipButton:
      pass
    elif self.selectedButton == self.toolFillEllipButton:
      pass  
    elif self.selectedButton == self.toolTextButton:
      theText = tkSimpleDialog.askstring(title="Xasy - Text",prompt="Enter text to display:",initialvalue="",parent=self.parent)		
      if theText != None and theText != "":
        theItem = xasyText(theText,(x,-y))
        theItem.drawOnCanvas(self.mainCanvas)
        self.bindItemEvents(theItem)
        self.addItemToFile(theItem)
    elif self.selectedButton == self.toolAsyButton:
      self.addItemToFile(xasyScript())
      tl = Toplevel(self.parent)
      xasyCodeEditor(tl,"// enter your code here",self.fileItems[-1].setScript)
      self.parent.wait_window(tl)
      self.fileItems[-1].drawOnCanvas(self.mainCanvas)
      self.bindItemEvents(self.fileItems[-1])
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
          self.itemBeingDrawn = xasyShape(path,pen=asyPen(self.penColor,self.penWidth))
        else:
          self.itemBeingDrawn = xasyFilledShape(path,pen=asyPen(self.penColor,self.penWidth,"evenodd"))
        self.itemBeingDrawn.drawOnCanvas(self.mainCanvas)
        self.bindItemEvents(self.itemBeingDrawn)
        self.mainCanvas.bind("<Motion>",self.extendDraw)
      self.itemBeingDrawn.drawOnCanvas(self.mainCanvas)

  def extendDraw(self,event):
    x,y = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    tags = self.mainCanvas.gettags("itemBeingDrawn")
    self.itemBeingDrawn.setLastPoint((x,-y))
    self.itemBeingDrawn.drawOnCanvas(self.mainCanvas)
    self.canvMotion(event)

  def endDraw(self,event):
    if not self.inDrawingMode or self.itemBeingDrawn == None:
      return
    x,y = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    #if self.selectedButton in [self.toolDrawLinesButton,self.toolDrawPolyButton,self.toolFillPolyButton]:
      #self.itemBeingDrawn.appendPoint((x,-y),'--')
    #else:
      #self.itemBeingDrawn.appendPoint((x,-y),'..')

    #only needed for certain key bindings when startDraw is triggered right before an endDraw
    #e.g.: single click: startDraw, double click: endDraw
    self.itemBeingDrawn.removeLastPoint()
    self.itemBeingDrawn.setLastPoint((x,-y))
    self.itemBeingDrawn.drawOnCanvas(self.mainCanvas)
    self.addItemToFile(self.itemBeingDrawn)
    del self.itemBeingDrawn
    self.itemBeingDrawn = None
    self.mainCanvas.dtag("itemBeingDrawn","itemBeingDrawn")
    self.mainCanvas.bind("<Motion>",self.canvMotion)
    self.inDrawingMode = False


  def canvLeftDown(self,event):
    x,y = self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y)
    #print "Left Mouse Down"
    self.selectDragStart = (self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y))
    if self.freeMouseDown and self.editor != None:
      self.editor.endEdit()
      self.editor = None
    elif self.selectedButton == self.toolSelectButton:
      if self.freeMouseDown:
        self.clearSelection()
        self.dragSelecting = False
    else:
      self.startDraw(event)

  def canvLeftUp(self,event):
    #print "Left Mouse Up"
    self.freeMouseDown = True
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
      self.updateSelection()

  def canvDrag(self,event):
    if self.selectedButton == self.toolSelectButton and self.editor == None:
      self.mainCanvas.coords("outlineBox",self.selectDragStart[0],self.selectDragStart[1],self.mainCanvas.canvasx(event.x),self.mainCanvas.canvasy(event.y))
      self.showSelectionBox()
      self.dragSelecting = True

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

  def hidePopupMenu(self):
    try:
      self.itemPopupMenu.unpost()
    except:
      pass

  def itemMenuPopup(self,parent,item,x,y):
    self.hidePopupMenu()
    self.itemPopupMenu = Menu(parent,tearoff=False)
    self.itemPopupMenu.add_command(label="Edit",command=self.popupEdit)
    self.itemPopupMenu.add_command(label="View code",command=self.popupViewCode)
    self.itemPopupMenu.add_separator()
    self.itemPopupMenu.add_command(label="Delete",command=self.popupDelete)
    self.itemPopupMenu.item = item
    #self.itemPopupMenu.bind("<Leave>",lambda a:self.itemPopupMenu.unpost())
    #self.itemPopupMenu.bind("<FocusOut>",lambda a:self.itemPopupMenu.unpost())
    self.itemPopupMenu.post(x,y)

  def itemPropMenuPopup(self,event):
    try:
      item = self.fileItems[int(self.propList.curselection()[0])]
    except:
      item = None
    if item != None:
      self.itemMenuPopup(self.propList,item,event.x_root,event.y_root)

  def itemCanvasMenuPopup(self,event):
    if self.selectedButton == self.toolSelectButton:
      try:
        item = self.findItem(self.mainCanvas.find_withtag(CURRENT)[0])
      except:
        item = None
      if item != None:
        self.itemMenuPopup(self.mainCanvas,item,event.x_root,event.y_root)

  def editOptions(self):
    xasyOptionsDialog.xasyOptionsDlg(self.parent)
    self.applyOptions()

  def resetOptions(self):
    xasyOptions.setDefaults()
    xasyOptions.save()
    self.applyOptions()