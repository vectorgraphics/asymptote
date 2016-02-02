#!/usr/bin/env python
###########################################################################
#
# xasyOptions provides a mechanism for storing and restoring a user's
# preferences.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
###########################################################################

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
    'defPenWidth':1.0,
    'externalEditor':''
  }

if sys.platform[:3] == "win":
  defaultOptions['externalEditor'] = "%PROGRAMFILES%\Windows NT\Accessories\wordpad.exe"
else:
  defaultOptions['externalEditor'] = "emacs"


options = defaultOptions.copy()

def settingsFileLocation():
  folder = ""
  try:
    folder = os.path.expanduser("~/.asy/")
  except:
    pass
  return os.path.normcase(os.path.join(folder,"xasy.conf"))

def setAsyPathFromWindowsRegistry():
  try:
    import _winreg as registry
    #test both registry locations
    try:
      key = registry.OpenKey(registry.HKEY_LOCAL_MACHINE,"Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\Asymptote")
      options['asyPath'] = registry.QueryValueEx(key,"Path")[0]+"\\asy.exe"
      registry.CloseKey(key)
    except:
      key = registry.OpenKey(registry.HKEY_LOCAL_MACHINE,"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Asymptote")
      options['asyPath'] = registry.QueryValueEx(key,"InstallLocation")[0]+"\\asy.exe"
      registry.CloseKey(key)
  except:
    #looks like asy is not installed or this isn't Windows
    pass

def setDefaults():
  global options
  options = defaultOptions.copy()
  if sys.platform[:3] == 'win': #for windows, wince, win32, etc
    setAsyPathFromWindowsRegistry()
  save()

def load():
  global options
  fileName = settingsFileLocation()
  if not os.path.exists(fileName):
    #make folder
    thedir = os.path.dirname(fileName)
    if not os.path.exists(thedir):
      try:
        os.makedirs(thedir)
      except:
        raise Exception("Could not create configuration folder")
    if not os.path.isdir(thedir):
      raise Exception("Configuration folder path does not point to a folder")
    setDefaults()
  try:
    f = open(fileName,"rb")
    newOptions = pickle.load(f)
    for key in options.keys():
      if type(newOptions[key]) != type(options[key]):
        raise Exception("Bad type for entry in xasy settings")
    options = newOptions
  except:
    setDefaults()

def save():
  global options
  fileName = settingsFileLocation()
  try:
    f = open(fileName,"wb")
    pickle.dump(options,f)
    f.close()
  except:
    raise Exception("Error saving preferences")

load()

if __name__=='__main__':
  print (settingsFileLocation())
  print ("Current content")
  load()
  print ("Setting defaults")
  setDefaults()
  save()
  load()
  options['showAxes'] = options['showGrid'] = False
  save()
  print ("Set to False")
  load()
  options['showAxes'] = options['showGrid'] = True
  save()
  print ("Set to True")
  load()
  print (options)
