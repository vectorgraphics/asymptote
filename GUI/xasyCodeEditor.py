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
from string import split
import xasyOptions

def getText(text=""):
  """Launch the external editor"""
  temp = mkstemp()
  tempf = fdopen(temp[0],"r+w")
  tempf.write(text)
  tempf.flush()
  try:
    call(split(xasyOptions.options['externalEditor'])+[temp[1]])
  except:
    raise Exception('Error launching external editor.')
  tempf.seek(0)
  text = tempf.read()
  remove(temp[1])
  return text

if __name__ == '__main__':
  #run a test
  print getText("Here is some text to edit")
