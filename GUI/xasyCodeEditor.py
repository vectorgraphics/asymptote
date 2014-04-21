#!/usr/bin/env python
###########################################################################
#
# xasyCodeEditor implements a simple text editor for Asymptote scripts in
# xasy.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
############################################################################

from subprocess import call
from tempfile import mkstemp
from os import remove
from os import fdopen
from os import path
from string import split
import xasyOptions

def getText(text=""):
  """Launch the external editor"""
  temp = mkstemp()
  tempf = fdopen(temp[0],"w")
  tempf.write(text)
  tempf.close()
  try:
    cmdpath,cmd = path.split(path.expandvars(xasyOptions.options['externalEditor']))
    split_cmd = split(cmd)
    cmdpart = [path.join(cmdpath,split_cmd[0])]
    argpart = split_cmd[1:]+[temp[1]]
    arglist = cmdpart+argpart
    call(arglist)
  except Exception as e:
    raise Exception('Error launching external editor.')
  
  try:
    tempf = open(temp[1],"r")
    text = tempf.read()
    tempf.close()
    remove(temp[1])
  except Exception as e:
    raise Exception('Error reading from external editor.')
  return text

if __name__ == '__main__':
  #run a test
  print getText("Here is some text to edit")
