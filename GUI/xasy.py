#!/usr/bin/env python
#
# Author: Orest Shardt
# Created: June 29, 2007
#
import sys
from Tkinter import *
import xasyMainWin

root = Tk()
if len(sys.argv)>1:
  app = xasyMainWin.xasyMainWin(root,sys.argv[1])
else:
  app = xasyMainWin.xasyMainWin(root)
root.mainloop()
