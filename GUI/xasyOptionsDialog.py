#!/usr/bin/env python
#
# Author: Orest Shardt
# Created: June 29, 2007
#

from Tkinter import *
from xasyOptions import *

class xasyOptionsDlg:
  """A dialog to interact with users about their preferred settings"""
  def __init__(self,master):
    self.parent = master
    Label(self.parent,text="Xasy Options").grid(row=0,column=0)
    Label(self.parent,text="asy path:").grid(row=1,column=0)
    Entry(self.parent).grid(row=1,column=1,columnspan=3)
    Button(self.parent,text="OK").grid(row=2,column=1)
    Button(self.parent,text="Cancel").grid(row=2,column=2)
    Button(self.parent,text="Apply").grid(row=2,column=3) 

if __name__ == '__main__':
 root = Tk()
 opts = xasyOptionsDlg(root)
 root.mainloop()
