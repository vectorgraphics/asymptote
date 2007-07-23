#!/usr/bin/env python
#
# Author: Orest Shardt
# Created: June 29, 2007
#

import pickle
import sys,os
import errno

defaultOptions = {
    'asyPath':'asy',
    'showDebug':False,
    'showGrid':False,
    'gridX':10,
    'gridY':10,
    'gridColor':'#eeeeee',
    'showAxes':True,
    'axisX':10,
    'axisY':10,
    'axesColor':'#cccccc',
    'tickColor':'#eeeeee',
    'defPenOptions':'',
    'defPenColor':'#000000',
    'defPenWidth':1.0
  }

options = defaultOptions.copy()

def settingsFileLocation():
  folder = ""
  if sys.platform[:3] == 'win': #for windows, wince, win32, etc
    try:
      folder = os.path.join(os.environ["APPDATA"],"xasy/")
    except:
      try:
        folder = os.path.expanduser("~/xasy/")
      except:
        pass
  else:
    try:
      folder = os.path.expanduser("~/.xasy/")
    except:
      pass
  return os.path.normcase(os.path.join(folder,"xasy.conf"))

def setDefaults():
  global options
  options = defaultOptions.copy()

def load():
  global options
  fileName = settingsFileLocation()
  if not os.path.exists(fileName):
    setDefaults()
    #make folder
    thedir = os.path.dirname(fileName)
    if not os.path.exists(thedir):
      try:
        os.makedirs(thedir)
      except:
        raise Exception,"Could not create configuration folder"
    if not os.path.isdir(thedir):
      raise Exception,"Configuration folder path does not point to a folder"
    save()
  try:
    f = open(fileName,"rb")
    newOptions = pickle.load(f)
    for key in options.keys():
      if type(newOptions[key]) != type(options[key]):
        raise Exception,"Bad type for entry in xasy settings"
    options = newOptions
  except:
    setDefaults()
    print "Error loading configuration: using defaults."

def save():
  global options
  fileName = settingsFileLocation()
  try:
    f = open(fileName,"wb")
    pickle.dump(options,f)
    f.close()
  except:
    raise Exception,"Error saving preferences"

if __name__=='__main__':
  print settingsFileLocation()
  print "Current content"
  load()
  print "Setting defaults"
  setDefaults()
  save()
  load()
  options['showAxes'] = options['showGrid'] = False
  save()
  print "Set to False"
  load()
  options['showAxes'] = options['showGrid'] = True
  save()
  print "Set to True"
  load()
  print options