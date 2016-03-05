#!/usr/bin/env python
###########################################################################
#
# xasyColorPicker implements a dialog that allows a user to choose a color
# from those already defined in Asymptote or a custom RGB color.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
############################################################################

import sys

if sys.version_info >= (3, 0):
  from tkinter import *
  from tkinter import colorchooser
else:
  from Tkinter import *
  import tkColorChooser as colorchooser

asyColors = { "black":(0,0,0),
  "white":(1,1,1),
  "gray":(0.5,0.5,0.5),
  "red":(1,0,0),
  "green":(0,1,0),
  "blue":(0,0,1),
  "cmyk":(1,1,1),
  "Cyan":(0,1,1),
  "Magenta":(1,0,1),
  "Yellow":(1,1,0),
  "Black":(0,0,0),
  "cyan":(0,1,1),
  "magenta":(1,0,1),
  "yellow":(1,1,0),
  "palered":(1,0.75,0.75),
  "palegreen":(0.75,1,0.75),
  "paleblue":(0.75,0.75,1),
  "palecyan":(0.75,1,1),
  "palemagenta":(1,0.75,1),
  "paleyellow":(1,1,0.75),
  "palegray":(0.95,0.95,0.95),
  "lightred":(1,0.5,0.5),
  "lightgreen":(0.5,1,0.5),
  "lightblue":(0.5,0.5,1),
  "lightcyan":(0.5,1,1),
  "lightmagenta":(1,0.5,1),
  "lightyellow":(1,1,0.5),
  "lightgray":(0.9,0.9,0.9),
  "mediumred":(1,0.25,0.25),
  "mediumgreen":(0.25,1,0.25),
  "mediumblue":(0.25,0.25,1),
  "mediumcyan":(0.25,1,1),
  "mediummagenta":(1,0.25,1),
  "mediumyellow":(1,1,0.25),
  "mediumgray":(0.75,0.75,0.75),
  "heavyred":(0.75,0,0),
  "heavygreen":(0,0.75,0),
  "heavyblue":(0,0,0.75),
  "heavycyan":(0,0.75,0.75),
  "heavymagenta":(0.75,0,0.75),
  "lightolive":(0.75,0.75,0),
  "heavygray":(0.25,0.25,0.25),
  "deepred":(0.5,0,0),
  "deepgreen":(0,0.5,0),
  "deepblue":(0,0,0.5),
  "deepcyan":(0,0.5,0.5),
  "deepmagenta":(0.5,0,0.5),
  "olive":(0.5,0.5,0),
  "deepgray":(0.1,0.1,0.1),
  "darkred":(0.25,0,0),
  "darkgreen":(0,0.25,0),
  "darkblue":(0,0,0.25),
  "darkcyan":(0,0.25,0.25),
  "darkmagenta":(0.25,0,0.25),
  "darkolive":(0.25,0.25,0),
  "darkgray":(0.05,0.05,0.05),
  "orange":(1,0.5,0),
  "fuchsia":(1,0,0.5),
  "chartreuse":(0.5,1,0),
  "springgreen":(0,1,0.5),
  "purple":(0.5,0,1),
  "royalblue":(0,0.5,1)
  }
colorLayout = [['palered',
  'lightred',
  'mediumred',
  'red',
  'heavyred',
  'deepred',
  'darkred',
  'palegreen',
  'lightgreen',
  'mediumgreen',
  'green',
  'heavygreen',
  'deepgreen',
  'darkgreen',
  'paleblue',
  'lightblue',
  'mediumblue',
  'blue',
  'heavyblue',
  'deepblue',
  'darkblue'],
  ['palecyan',
  'lightcyan',
  'heavycyan',
  'deepcyan',
  'darkcyan',
  'palemagenta',
  'lightmagenta',
  'mediummagenta',
  'magenta',
  'heavymagenta',
  'deepmagenta',
  'darkmagenta',
  'yellow',
  'lightyellow',
  'mediumyellow',
  'yellow',
  'lightolive',
  'olive',
  'darkolive',
  'palegray',
  'lightgray',
  'mediumgray',
  'gray',
  'heavygray',
  'deepgray',
  'darkgray'],
  ['black',
  'white',
  'orange',
  'fuchsia',
  'chartreuse',
  'springgreen',
  'purple',
  'royalblue',
  'Cyan',
  'Magenta',
  'Yellow',
  'Black']]

def makeRGBfromTkColor(tkColor):
  """Convert a Tk color of the form #rrggbb to an asy rgb color"""
  r = int('0x'+tkColor[1:3],16)
  g = int('0x'+tkColor[3:5],16)
  b = int('0x'+tkColor[5:7],16)
  r /= 255.0
  g /= 255.0
  b /= 255.0
  return (r,g,b)

def RGBreal255(rgb):
  """Convert an RGB color from 0-1 to 0-255"""
  return [min(int(256*a),255) for a in rgb]

def RGB255hex(rgb):
  """Make a color in the form #rrggbb in hex from r,g,b in 0-255"""
  return "#{}".format("".join(["{:02x}".format(a) for a in rgb]))

class xasyColorDlg(Toplevel):
  """A dialog for choosing an asymptote color. It displays the usual asy presets and allows custom rgb colors"""
  def __init__(self,master=None,color=(0,0,0)):
    Toplevel.__init__(self,master,width=500,height=500)
    self.resizable(False,False)
    self.parent = master
    self.title("Color Picker")
    self.transient(master)
    self.focus_set()
    self.wait_visibility()
    self.grab_set()
    self.color = self.oldColor = color
    cwidth = 120
    rheight = 20
    self.pframe=Frame(self,bd=0)
    self.pframe.rowconfigure(0,weight=1)
    self.pframe.columnconfigure(0,weight=1)
    Label(self.pframe,text="Color Presets").grid(row=0,column=0)
    self.colScroll = Scrollbar(self.pframe,orient=VERTICAL)
    self.colorList = Canvas(self.pframe, width=cwidth*len(colorLayout), scrollregion=(0,0,20+cwidth*len(colorLayout),20+rheight*max([len(i) for i in colorLayout])),yscrollcommand=self.colScroll.set,relief=FLAT)
    self.colScroll.config(command=self.colorList.yview)
    self.colScroll.grid(row=1,column=1,sticky=N+S)
    self.colorList.grid(row=1,column=0,sticky=W)
    ccount = 0
    for column in colorLayout:
      rcount = 0
      for name in column:
        self.colorList.create_rectangle(10+cwidth*ccount,10+rheight*rcount,cwidth*ccount+25,rheight*rcount+25,tags=(name,"preset"),fill=RGB255hex(RGBreal255(asyColors[name])))
        self.colorList.create_text(cwidth*ccount+30,10+rheight*rcount,text=name,anchor=NW,tags=(name,"preset"),fill="black",activefill=RGB255hex(RGBreal255(asyColors[name])))
        rcount += 1
      ccount += 1
    self.colorList.tag_bind("preset","<Button-1>",self.setColorEvt)
    Button(self,text="Custom color...",command=self.getCustom).grid(row=2,column=0,sticky=W,padx=5,pady=5)
    self.colDisp = Canvas(self,width=200,height=20,background=RGB255hex(RGBreal255(self.color)),relief=SUNKEN, bd=3)
    self.colDisp.grid(row=2,column=1,columnspan=2)
    self.rowconfigure(3,minsize=10)
    self.columnconfigure(0,weight=1)
    self.columnconfigure(1,weight=1)
    self.columnconfigure(2,weight=1)
    Button(self,text="OK",default=ACTIVE,command=self.destroy).grid(row=4,column=1,sticky=E+W,padx=5,pady=5)
    Button(self,text="Cancel",command=self.cancel).grid(row=4,column=2,sticky=E+W,padx=5,pady=5)
    self.pframe.grid(row=1,column=0,columnspan=3,padx=10,pady=10)
    self.bind("<Return>",self.closeUp)
    self.setColor(color)
  def closeUp(self,event):
    """Close the dialog forcibly"""
    self.destroy()
  def getCustom(self):
    """Request a custom RGB color using a tkColorChooser"""
    result=tkColorChooser.askcolor(initialcolor=RGB255hex(RGBreal255(self.color)),title="Custom Color",parent=self)
    if result != (None,None):
      self.setColor((result[0][0]/255.0,result[0][1]/255.0,result[0][2]/255.0))
  def cancel(self):
    """Respond to the user pressing cancel"""
    self.color = self.oldColor
    self.destroy()
  def setColor(self,color):
    """Save the color and update the color display"""
    self.color = color
    self.colDisp.configure(background=RGB255hex(RGBreal255(self.color)))
  def setColorEvt(self,event):
    """Respond to the user clicking a color from the palette"""
    self.setColor(asyColors[self.colorList.gettags(CURRENT)[0]])
  def getColor(self,initialColor=(0,0,0)):
    """Use this method to prompt for a color. It returns the new color or the old color if the user cancelled the operation.

      e.g:
        print (xasyColorDlg(Tk()).getColor((1,1,0)))
    """
    self.setColor(initialColor)
    self.oldColor = initialColor
    self.wait_window(self)
    return self.color

if __name__ == '__main__':
  root = Tk()
  Button(root,text="Pick Color",command=lambda:xasyColorDlg(root).getColor()).pack()
  root.mainloop()
