#!/usr/bin/env python
###########################################################################
#
# xasyFile implements the loading, parsing, and saving of an xasy file.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
############################################################################

from string import *
from xasy2asy import *
import re

class xasyParseError(Exception):
  """A parsing error"""
  pass

class xasyFileError(Exception):
  """An i/o error or other error not related to parsing"""
  pass

def parseFile(inFile):
  """Parse a file returning a list of xasyItems"""
  lines = inFile.read()
  lines = lines.splitlines()
  #lines = [line for line in lines.splitlines() if not line.startswith("//")]
  result = []
  if lines[0] != "initXasyMode();":
    raise xasyFileError("Invalid file format: First line must be \"initXasyMode();\"")
  lines.pop(0)
  lineCount = 2
  lineNum = len(lines)
  while lineNum > 0:
    line = lines[0]
    lines.pop(0)
    if not line.isspace() and len(line)>0:
      try:
        #print ("Line {:d}: {:s}".format(lineCount,line)),
        lineResult = parseLine(line.strip(),lines)
      except:
        raise xasyParseError("Parsing error: line {:d} in {:s}\n{:s}".format(lineCount,inFile.name,line))

      if lineResult != None:
        result.append(lineResult)
        #print ("\tproduced: {:s}".format(str(lineResult)))
    lineCount += lineNum-len(lines)
    lineNum = len(lines)
  return result

transformPrefix = "xformStack"
scriptPrefix = "startScript(); {"
scriptSuffix = "} endScript();"
def extractScript(lines):
  """Find the code belonging to a script item"""
  theScript = ""
  line = lines.pop(0)
  level = 1
  while level > 0:
    check = line.lstrip()
    while check.endswith(scriptSuffix):
      level -= 1
      line = line[:len(line)-len(scriptSuffix)]
      check = line.lstrip()
    if check.startswith(scriptPrefix):
      level += 1
    theScript += line + "\n"
    if level > 0:
      line = lines.pop(0)

  global pendingTransformsD
  ts = pendingTransformsD[:]
  pendingTransformsD = []
  return xasyScript(None,script=theScript,transforms=ts[:])

pendingTransforms = []
pendingTransformsD = []
def addTransform(index,t,active=1):
  """Place a transform in the list of transforms, expanding the list as needed"""
  while len(pendingTransformsD) < index+1:
    pendingTransformsD.append(identity())
  deleted = int(active==0)
  pendingTransformsD[index]=asyTransform(t,deleted)

def parseIndexedTransforms(args):
  """Parse a list of indexedTransforms, adding them to the current list of transforms"""
  global pendingTransformsD
  pendingTransformsD = []
  args = args.replace("indexedTransform","")
  false = 0
  tList = [eval(a) for a in ")?(".join(args.split("),(")).split("?")]
  for a in tList:
    addTransform(*a)

def parseTransformExpression(line):
  """Parse statements related to the xformStack
  
  Syntax:
    xformStack.push(transform)
      e.g.: xformStack.push((0,0,1,0,0,1)); //the identity
    xformStack.add(indexedTransform(index,transform)[,...])
      e.g.: xformStack.add(indexedTransform(1,(0,0,1,0,0,1));
  """
  global pendingTransforms
  stackCmd = line[len(transformPrefix)+1:line.find("(")]
  if line[-2:] != ");":
    raise xasyParseError("Invalid syntax")
  args = line[line.find("(")+1:-2]
  if stackCmd == "push":
    t = asyTransform(eval(args))
    pendingTransforms.append(t)
  elif stackCmd == "add":
    parseIndexedTransforms(args)
  else:
    raise xasyParseError("Invalid transform stack command.")
  return None

def parseLabel(line):
  """Parse an asy Label statement, returning an xasyText item"""
  if not (line.startswith("Label(") and line.endswith(",align=SE)")):
    raise xasyParseError("Invalid syntax")
  args = line[6:-1]
  loc2 = args.rfind(",align=SE")
  loc1 = args.rfind(",",0,loc2-1)
  loc = args.rfind(",(",0,loc1-1)
  if loc < 2:
    raise xasyParseError("Invalid syntax")
  text = args[1:loc-1]
  location = eval(args[loc+1:args.find("),",loc)+1])
  pen = args[loc:loc2]
  pen = pen[pen.find(",")+1:]
  pen = pen[pen.find(",")+1:]
  pen = pen[pen.find(",")+1:]
  global pendingTransforms
  return xasyText(text,location,parsePen(pen),pendingTransforms.pop())

def parseLabelCommand(line):
  """Parse a label command returning an xasyText object
  
  Syntax:
    label(Label(text,location,pen,align=SE));
      e.g.: label(Label("Hello world!",(0,0),rgb(0,0,0)+0.5,align=SE));
  """
  if line[-2:] != ");":
    raise xasyParseError("Invalid syntax")
  arguments = line[6:-2]
  return parseLabel(arguments)

def parseDrawCommand(line):
  """Parse a draw command returning an xasyShape object
  
  Syntax:
    draw(path,pen);
      e.g.: draw((0,0)..controls(0.33,0.33)and(0.66,0.66)..(1,1),rgb(1,0,1)+1.5);
  """
  if line[-2:] != ");":
    raise xasyParseError("Invalid syntax")
  args = line[5:-2]
  loc = args.rfind(",rgb")
  path = args[:loc]
  pen = args[loc+1:]
  global pendingTransforms
  return xasyShape(parsePathExpression(path),parsePen(pen),pendingTransforms.pop())

def parseFillCommand(line):
  """Parse a fill command returning an xasyFilledShape object
  
  Syntax:
    fill(cyclic path,pen);
      e.g.: fill((0,0)..controls(0.33,0.33)and(0.66,0.66)..(1,1)..controls(0.66,0)and(0.33,0)..cycle,rgb(1,0,1)+1.5);
  """
  if line[-2:] != ");":
    raise xasyParseError("Invalid syntax")
  args = line[5:-2]
  loc = args.rfind(",rgb")
  path = args[:loc]
  pen = args[loc+1:]
  global pendingTransforms
  return xasyFilledShape(parsePathExpression(path),parsePen(pen),pendingTransforms.pop())

def parsePen(pen):
  """Parse a pen expression returning an asyPen
  
  Syntax:
    color+width[+options]
      e.g.: rgb(0,0,0)+1.5+evenodd
      e.g.: rgb(0,1,0)+1.23
  """
  try:
    tokens = pen.split("+")
    color = eval(tokens[0][3:])
    width = float(tokens[1])
    if len(tokens)>2:
      options = "+".join(tokens[2:])
    else:
      options = ""
    return asyPen(color,width,options)
  except:
    raise xasyParseError("Invalid pen")

def parsePathExpression(expr):
  """Parse an asy path returning an asyPath()"""
  result = asyPath()
  expr = "".join(expr.split())
  #print (expr)
  if expr.find("controls") != -1:
    #parse a path with control points
    tokens = expr.split("..")
    nodes = [a for a in tokens if not a.startswith("controls")]
    for a in range(len(nodes)):
      if nodes[a] != "cycle":
        nodes[a] = eval(nodes[a])
    controls = [map(eval,a.replace("controls","").split("and")) for a in tokens if a.startswith("controls")]
    result.initFromControls(nodes, controls)
  else:
    #parse a path without control points
    tokens = re.split(r"(::|--|\.\.)",expr)
    linkSet = re.findall("::|--|\.\.",expr)
    nodeSet = [a for a in tokens if not re.match(r"::|--|\.\.",a)]
    #print (nodeSet)
    for a in range(len(nodeSet)):
      if nodeSet[a] != "cycle":
        nodeSet[a] = eval(nodeSet[a])
    #print (nodeSet)
    result.initFromNodeList(nodeSet, linkSet)
  return result

def takeUntilSemicolon(line,lines):
  """Read and concatenate lines until the collected lines end with a semicolon"""
  data = line
  while not data.endswith(";"):
    newline = lines.pop(0)
    data += newline
  return data

def parseLine(line,lines):
  """Parse a line of the file"""
  if len(line)==0 or line.isspace() or line.startswith("//"):
    return None
  elif line.startswith(scriptPrefix):
    return extractScript(lines)
  elif line.startswith(transformPrefix):
    return parseTransformExpression(takeUntilSemicolon(line,lines))
  elif line.startswith("label("):
    return parseLabelCommand(takeUntilSemicolon(line,lines))
  elif line.startswith("draw("):
    return parseDrawCommand(takeUntilSemicolon(line,lines))
  elif line.startswith("fill("):
    return parseFillCommand(takeUntilSemicolon(line,lines))
  elif line.startswith("exitXasyMode();"):
    return None
  raise Exception("Could not parse the line")

fileHeader = """initXasyMode();
// This file was generated by xasy. It may be edited manually, however, a strict
// syntax must be followed. It is advised that manually scripted items be added
// in the form of a script either by using xasy or by mimicking the format of an
// xasy-generated script item.
// Please consult the documentation or the examples provided for details.
"""

fileFooter = """// This is the end of the file
exitXasyMode();

"""

def saveFile(file,xasyItems):
  """Write a list of xasyItems to a file"""
  file.write(fileHeader)
  for item in xasyItems:
    file.write(item.getCode()+"\n\n")
  file.write(fileFooter)

if __name__ == '__main__':
  root = Tk()
  try:
    name = raw_input("enter file name (\"../../xasyTest.asy\"):")
    if name == '':
      name = "../../xasyTest.asy"
    f = open(name,"rt")
  except:
    print ("Could not open file.")
    asy.quit()
    sys.exit(1)

  fileItems = [] 
  try:
    fileItems = parseFile(f)
    res = map(str,fileItems)
    print ("----------------------------------")
    print ("Objects in {:s}".format(f.name))
    print ("----------------------------------")
    for a in res:
      print (a)
    print ("----------------------------------")
    print ("successful parse")
    f.close()
  except:
    f.close()
    print ("parse failed")
    raise

  print ("making a file")
  f = open("testfile.asy","wt")
  saveFile(f,fileItems)
  f.close()
  root.configure(width=500,height=500)
  root.title("Results")
  canv = Canvas(root,width=500,height=500)
  canv.pack()
  for i in fileItems[1].imageList:
    canv.create_image(250+i.bbox[0],250-i.bbox[3],anchor = NW, image=i.image)
    Button(root,image=i.image).pack(side=LEFT)
  root.mainloop()
