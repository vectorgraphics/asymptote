#!/usr/bin/env python3
###########################################################################
#
# xasy implements a graphical interface for Asymptote.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
############################################################################

import getopt,sys,signal
import xasyMainWin
if sys.version_info >= (3, 0):
  from tkinter import *
else:
  from Tkinter import *
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QMainWindow
import Window1

signal.signal(signal.SIGINT,signal.SIG_IGN)

program=QApplication(args)
main_window=Window1.MainWindow1()
main_window.show()

root = Tk()
mag = 1.0
try:
  opts,args = getopt.getopt(sys.argv[1:],"x:")
  if(len(opts)>=1):
    mag = float(opts[0][1])
except:
  print ("Invalid arguments.")
  print ("Usage: xasy.py [-x magnification] [filename]")
  sys.exit(1)
if(mag <= 0.0):
  print ("Magnification must be positive.")
  sys.exit(1)
if(len(args)>=1):
  app = xasyMainWin.xasyMainWin(root,args[0],mag)
else:
  app = xasyMainWin.xasyMainWin(root,magnification=mag)
root.mainloop()
