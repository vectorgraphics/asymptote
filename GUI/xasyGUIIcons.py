#!/usr/bin/env python
##################################################################
# This file stores the icons used by the xasy GUI
#
# About images and base64
#
# Suppose you have image.gif and want to create a base64
# string. This can be accomplished using:
#
# import base64
# base64.encodestring(open("image.gif","rb").read())
#
# The resulting output, including the enclosing single quotes,
# is the base64 encoding of the image and can be used in the
# dictionary below.
#
#
# Suppose you have a base64 string, b64str, and want to create
# an image. This can be accomplished using:
#
# import base64
# open("image.gif","w").write(base64.decodestring(b64str))
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
##################################################################
import base64
import os
#toolbar icon image data in base64 eliminates need to worry about files
#these are the base64 encodings of the content of the directory xasy3Imgs
iconB64 = {
'lower': 'R0lGODlhGAAYAPEBAAAAAP///8zMzAAAACH5BAEAAAIALAAAAAAYABgAAAItlI+py+0Po5yUgosz\nrrybK2giqADed6LHKCZm+p7xx2Zuqsqr95KcJpv9cJUCADs=\n',
'rotate': 'R0lGODlhGAAYAKEBAAAAAP///////////yH5BAEKAAEALAAAAAAYABgAAAI9jI8JkN0LoVMyxDrX\nu8x1yn2HOJWWiVpkqrCl5npxNss1At9GrvN1pFOdbjAfK7dC+ZKZpHHzwoyk0ZCjAAA7\n',
'raise': 'R0lGODlhGAAYAPEBAAAAAP///8zMzAAAACH5BAEAAAIALAAAAAAYABgAAAIwlI+pywgND3ixzVvZ\nNDSn3nlKKH7fhaZmObKtk8Yh6dKlLcfC5vZ1jvIJh8SikVUAADs=\n',
'fillPoly': 'R0lGODlhGAAYAPECAAAAAIOBg////6usrSH5BAEAAAMALAAAAAAYABgAAAJLnI+py+0PDZhxgRBo\nPRdjHU3eCDbiiJZKh6YAw7apJdfvWst3Eucf7SMBU7qh59XLGJWS2A7hxP2kmSfnYpV8slBuhxti\nbrrj8qMAADs=\n',
'move': 'R0lGODlhGAAYAIABAAAAAP///yH5BAEAAAEALAAAAAAYABgAAAI4jI+py+0I3gNUNhqtwlVD7m3h\nkoVdUJ4MaKTYysVymbDoYcM4Tmv9eAO2cp6YEKUavY5BpvMZKgAAOw==\n',
'drawBezi': 'R0lGODlhGAAYAPEBAAAAAP///6usrQAAACH5BAEAAAIALAAAAAAYABgAAAI6lI+py+0AnYRUKhox\nsFvUFDXdM4LWUaKnEaorhqSX1noPmMquWJukzpr0YitRcfE5oobFpPIJjUoZBQA7\n',
'vertiMove': 'R0lGODlhGAAYAIABAAAAAP///yH5BAEAAAEALAAAAAAYABgAAAIsjI+py+0I3gNUNhqtwlVD7m3h\nko2QmZRooKKt+Y5xOFtc7dwrtrLd3gsKTQUAOw==\n',
'horizMove': 'R0lGODlhGAAYAIABAAAAAP///yH5BAEAAAEALAAAAAAYABgAAAIljI+py+0Po5y02oshAGu/7Skg\n143mSYpgGTYt8mbyTNf2jedWAQA7\n',
'fillEllip': 'R0lGODlhGAAYAPECAAAAAIOBg////6usrSH5BAEAAAMALAAAAAAYABgAAAJAnI+py+0PowS0gkmD\n3qE6wIXctYDi2SkmepLGyrYHHIcuXW93Lr+86BrgakHfrzjjIRGVFgVjWUqm1Kr1ijUUAAA7\n',
'text': 'R0lGODlhGAAYAIABAAAAAP///yH5BAEAAAEALAAAAAAYABgAAAI+jI+py+0Po5x0AgSu1SZvHnhS\nBnpio5Ukt2Idm3bysYrnddLwy+czH0rhFDkbTigj6UzKl68CjUqn1Ko1UAAAOw==\n',
'drawPoly': 'R0lGODlhGAAYAPEBAAAAAP///6usrQAAACH5BAEAAAIALAAAAAAYABgAAAI6lI+py+0PDZhxAXnr\nyZtDqoCOaHnhZ25VKrAM61ryOkdwjdzJleoqlsP1frtOqTUjhXSxB0+zhEofBQA7\n',
'drawLines': 'R0lGODlhGAAYAPEBAAAAAP///6usrQAAACH5BAEAAAIALAAAAAAYABgAAAI3lI+py+0AnYRAPmoZ\njvlwX3Vh8j2XUIIWNXoZS3ZoO8soSK+4fRuYnQPyFEHhcHecFV+ppDNRAAA7\n',
'drawShape': 'R0lGODlhGAAYAPEBAAAAAP///6usrQAAACH5BAEAAAIALAAAAAAYABgAAAI+lI+pK+CAHou0HTdt\nhAkrnl3gFXZfSZ7oWIVsWr6NzHgwqhm0e+P5ngEBa63VayhaIHUToCT4iNp81KoVVwAAOw==\n',
'drawEllip': 'R0lGODlhGAAYAPEBAAAAAP///6usrQAAACH5BAEAAAIALAAAAAAYABgAAAIylI+py+0PowS0gklX\ndRd29XmgdIQh+Z1TSSJpyxpqZMLqzOB4sgsbmKFZgrCi8YhMNgoAOw==\n',
'select': 'R0lGODlhGAAYAPIDAAAAAICAgMDAwP///6usrQAAAAAAAAAAACH5BAEAAAQALAAAAAAYABgAAANH\nSLrc/mvA6YCkGIiLIQhb54Gh2HwkZxKo4KoiSpam7L6rfdNZ4M+C3I+0Ush8wSLKCFIyPsnisyld\nAD7VabR6DWSt37BYmgAAOw==\n',
'fillShape': 'R0lGODlhGAAYAPECAAAAAIOBg////6usrSH5BAEAAAMALAAAAAAYABgAAAJLnI+pO+CAHouhBtis\nm9le4FWbAoamhlHnmh7lGrYuHDMvLc+0tji7N9L9eMIhsWM83pKp5bDlxOWip6DqlyuaJJyHl8sJ\ni8fksqIAADs=\n',
'asy': 'R0lGODlhGAAYAIABAP8AAAAAACH5BAEKAAEALAIAAwAUABIAAAImjI+py+0AHINy0ZouNjBurmGd\nt40fFT4j2aydGqaBq8jvxH46UwAAOw==\n'
}

def createGIF(key):
  """Create a gif file from the data in the iconB64 list of icons"""
  if key not in iconB64.keys():
    print "Error: %s not found in icon list."%key
    print "Available icons:",iconB64.keys()
  else:
    print "Generating %s.gif"%key
    open("%s.gif"%key,"w").write(base64.decodestring(iconB64[key]))

def createGIFs():
  """Create the files for all the icons in iconB64"""
  for name in iconB64.keys():
    createGIF(name)

def createStrFromGif(gifFile):
  """Create the base64 representation of a file"""
  return base64.encodestring(gifFile.read())

if __name__=='__main__':
  print "Testing the xasyGUIIcons module."
  print "Generating all the GIFs:"
  createGIFs()
  print "Checking consistency of all icons in iconB64"
  allpassed = True
  for icon in iconB64.keys():
    print ("Checking %s"%icon),
    if createStrFromGif(open("%s.gif"%icon,"rb")) == iconB64[icon]:
      print "\tPassed."
    else:
      print "\tFailed."
      allpassed= False
  if allpassed:
    print "All files succeeded."
  s = raw_input("Delete files generated? (y/n)")
  if s == "y":
    for name in iconB64.keys():
      print "Deleting %s.gif"%name,
      os.unlink(name+".gif")
      print "\tdone"
  print "Done"
