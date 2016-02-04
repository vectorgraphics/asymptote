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
'rotate': 'R0lGODlhGAAYAPAAAAAAAAAAACH5BAEAAAEALAAAAAAYABgAAAI7jI8JkO231mux1mkistL1zX0Q\ng2Fi6aGmurKp+8KKrJB0Zt+nzOQw6XPZgqjczuQ7eohKEDKoUYWIgQIAOw==\n',
'raise': 'R0lGODlhGAAYAPEBAAAAAP///8zMzAAAACH5BAEAAAIALAAAAAAYABgAAAIwlI+pywgND3ixzVvZ\nNDSn3nlKKH7fhaZmObKtk8Yh6dKlLcfC5vZ1jvIJh8SikVUAADs=\n',
'fillPoly': 'R0lGODlhGAAYAPEAAAAAAIOBgwAAAAAAACH5BAEAAAIALAAAAAAYABgAAAJGlI+py+0PEYgNBDCp\nDPxqY3UcRoViRzrmKWbLyqIMHI9vHbsbfuoHjfOBcrlbT0ATIo+gldKpMD1lL8vUo5oqS9vS5wsp\nAAA7\n',
'move': 'R0lGODlhGAAYAIABAAAAAP///yH5BAEAAAEALAAAAAAYABgAAAI4jI+py+0I3gNUNhqtwlVD7m3h\nkoVdUJ4MaKTYysVymbDoYcM4Tmv9eAO2cp6YEKUavY5BpvMZKgAAOw==\n',
'drawBezi': 'R0lGODlhGAAYAPEBAAAAAP///6usrQAAACH5BAEAAAIALAAAAAAYABgAAAI6lI+py+0AnYRUKhox\nsFvUFDXdM4LWUaKnEaorhqSX1noPmMquWJukzpr0YitRcfE5oobFpPIJjUoZBQA7\n',
'vertiMove': 'R0lGODlhGAAYAIABAAAAAP///yH5BAEAAAEALAAAAAAYABgAAAIsjI+py+0I3gNUNhqtwlVD7m3h\nko2QmZRooKKt+Y5xOFtc7dwrtrLd3gsKTQUAOw==\n',
'horizMove': 'R0lGODlhGAAYAIABAAAAAP///yH5BAEAAAEALAAAAAAYABgAAAIljI+py+0Po5y02oshAGu/7Skg\n143mSYpgGTYt8mbyTNf2jedWAQA7\n',
'fillEllip': 'R0lGODlhGAAYAPECAAAAAIOBg////6usrSH5BAEAAAMALAAAAAAYABgAAAJAnI+py+0PowS0gkmD\n3qE6wIXctYDi2SkmepLGyrYHHIcuXW93Lr+86BrgakHfrzjjIRGVFgVjWUqm1Kr1ijUUAAA7\n',
'text': 'R0lGODlhGAAYAIABAAAAAP///yH5BAEAAAEALAAAAAAYABgAAAI+jI+py+0Po5x0AgSu1SZvHnhS\nBnpio5Ukt2Idm3bysYrnddLwy+czH0rhFDkbTigj6UzKl68CjUqn1Ko1UAAAOw==\n',
'drawPoly': 'R0lGODlhGAAYAPAAAAAAAAAAACH5BAEAAAEALAAAAAAYABgAAAI4jI+py+0PEYhtgkmlzgFL/4DJ\nFULiVi4ns66smrUxrMj1fdqHR+60kfPdgCwLzbWTIU1LE+cJKQAAOw==\n',
'drawLines': 'R0lGODlhGAAYAPEBAAAAAP///6usrQAAACH5BAEAAAIALAAAAAAYABgAAAI3lI+py+0AnYRAPmoZ\njvlwX3Vh8j2XUIIWNXoZS3ZoO8soSK+4fRuYnQPyFEHhcHecFV+ppDNRAAA7\n',
'drawShape': 'R0lGODlhGAAYAPAAAAAAAAAAACH5BAEAAAEALAAAAAAYABgAAAI5jI+pywffIjQzIrCwdXnTplmh\nMoKmKIHVeZXp5cFcPH+0HbjbqKN17OoxgrTeKiOkPHjH3fIGjS4KADs=\n',
'drawEllip': 'R0lGODlhGAAYAPEBAAAAAP///6usrQAAACH5BAEAAAIALAAAAAAYABgAAAIylI+py+0PowS0gklX\ndRd29XmgdIQh+Z1TSSJpyxpqZMLqzOB4sgsbmKFZgrCi8YhMNgoAOw==\n',
'select': 'R0lGODlhGAAYAPIDAAAAAICAgMDAwP///6usrQAAAAAAAAAAACH5BAEAAAQALAAAAAAYABgAAANH\nSLrc/mvA6YCkGIiLIQhb54Gh2HwkZxKo4KoiSpam7L6rfdNZ4M+C3I+0Ush8wSLKCFIyPsnisyld\nAD7VabR6DWSt37BYmgAAOw==\n',
'fillShape': 'R0lGODlhGAAYAPEAAAAAAIOBgwAAAAAAACH5BAEAAAIALAAAAAAYABgAAAJHlI+pywff4gsUxgSo\nrhflzXXCB4YXWQIiCqpnubnLw8KyU8Omket77wvcgD4ZUTcMIj3KlOLYejY1N8/R0qChaCIrtgsO\nRwoAOw==\n',
'asy': 'R0lGODlhGAAYAIABAP8AAAAAACH5BAEKAAEALAIAAwAUABIAAAImjI+py+0AHINy0ZouNjBurmGd\nt40fFT4j2aydGqaBq8jvxH46UwAAOw==\n'
}

def createGIF(key):
  """Create a gif file from the data in the iconB64 list of icons"""
  if key not in iconB64.keys():
    print ("Error: {:s} not found in icon list.".format(key))
    print ("Available icons:",iconB64.keys())
  else:
    print ("Generating {:s}.gif".format(key))
    open("{:s}.gif".format(key),"w").write(base64.decodestring(iconB64[key]))

def createGIFs():
  """Create the files for all the icons in iconB64"""
  for name in iconB64.keys():
    createGIF(name)

def createStrFromGif(gifFile):
  """Create the base64 representation of a file"""
  return base64.encodestring(gifFile.read())

if __name__=='__main__':
  print ("Testing the xasyGUIIcons module.")
  print ("Generating all the GIFs:")
  createGIFs()
  print ("Checking consistency of all icons in iconB64")
  allpassed = True
  for icon in iconB64.keys():
    print ("Checking {:s}".format(icon))
    if createStrFromGif(open("{:s}.gif".format(icon),"rb")) == iconB64[icon]:
      print ("\tPassed.")
    else:
      print ("\tFailed.")
      allpassed= False
  if allpassed:
    print ("All files succeeded.")
  s = raw_input("Delete generated files? (y/n)")
  if s == "y":
    for name in iconB64.keys():
      print ("Deleting {:s}.gif".format(name))
      os.unlink(name+".gif")
      print ("\tdone")
  print ("Done")
