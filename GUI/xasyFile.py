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
import xasy2asy as x2a
import io
import re


class xasyParseError(Exception):
    """A parsing error"""
    pass


class xasyFileError(Exception):
    """An i/o error or other error not related to parsing"""
    pass


def extractTransform(line):
    """Returns key and the new transform."""
    # see https://regex101.com/r/6DqkRJ/4 for info
    mapString = x2a.xasyItem.mapString
    testMatch = re.match(r'^{0:s}\s*\(\s*\"([^\"]+)\"\s*,\s*\(([\d, ]+)\)\s*\)'.format(mapString), line.strip())
    if testMatch is None:
        mapOnlyMatch = re.match(r'^{0:s}\s*\(\s *\"([^\"]+)\"\s*\)'.format(mapString), line.strip())
        if mapOnlyMatch is None:
            return None
        else:
            key = mapOnlyMatch.group(1)
            return key, x2a.identity()
    else:
        key = testMatch.group(1)
        rawStr = testMatch.group(2)
        rawStrArray = rawStr.split(',')

        if len(rawStrArray) != 6:
            return None
        transf = [int(val.strip()) for val in rawStrArray]
        return key, x2a.asyTransform(transf)


def extractTransformsFromFile(fileStr):
    transfDict = {}
    maxItemCount = 0
    with io.StringIO() as rawCode:
        for line in fileStr.splitlines():
            test_transf = extractTransform(line)
            if test_transf is None:
                rawCode.write(line + '\n')
            else:
                key, transf = test_transf
                if key not in transfDict.keys():
                    transfDict[key] = []
                transfDict[key].append(transf)

                # see https://regex101.com/r/RgeBVc/1 for regex

                testNum = re.match(r'^x(\d+)$', key)
                if testNum is not None:
                    maxItemCount = max(maxItemCount, int(testNum.group(1)))
        final_str = rawCode.getvalue()
    return final_str, transfDict, maxItemCount


def parseFile(inFile):
    """Parse a file returning a list of xasyItems"""
    lines = inFile.read()
    lines = lines.splitlines()

    # lines = [line for line in lines.splitlines() if not line.startswith("//")]
    result = []
    lineCount = 1
    lineNum = len(lines)
    while lineNum > 0:
        line = lines[0]
        lines.pop(0)
        if not line.isspace() and len(line) > 0:
            try:
                # print ("Line {:d}: {:s}".format(lineCount,line))
                lineResult = parseLine(line.strip(), lines)
            except:
                raise xasyParseError("Parsing error: line {:d} in {:s}\n{:s}".format(lineCount, inFile.name, line))

            if lineResult is not None:
                result.append(lineResult)
                # print ("\tproduced: {:s}".format(str(lineResult)))
        lineCount += lineNum - len(lines)
        lineNum = len(lines)
    return result


transformPrefix = "xformStack"
scriptPrefix = "startScript(); {"
scriptSuffix = "} endScript();"


pendingTransforms = []
pendingTransformsD = []

def extractScript(lines):
    """Find the code belonging to a script item"""
    theScript = ""
    line = lines.pop(0)
    level = 1
    while level > 0:
        check = line.lstrip()
        while check.endswith(scriptSuffix):
            level -= 1
            line = line[:len(line) - len(scriptSuffix)]
            check = line.lstrip()
        if check.startswith(scriptPrefix):
            level += 1
        theScript += line + "\n"
        if level > 0:
            line = lines.pop(0)

    global pendingTransformsD
    ts = pendingTransformsD[:]
    pendingTransformsD = []
    return x2a.xasyScript(None, script=theScript, transforms=ts[:], engine=None)


def addTransform(index, t, active=1):
    """Place a transform in the list of transforms, expanding the list as needed"""
    while len(pendingTransformsD) < index + 1:
        pendingTransformsD.append(x2a.identity())
    deleted = int(active == 0)
    pendingTransformsD[index] = x2a.asyTransform(t, deleted)


def parseIndexedTransforms(args):
    """Parse a list of indexedTransforms, adding them to the current list of transforms"""
    global pendingTransformsD
    pendingTransformsD = []
    args = args.replace("indexedTransform", "")
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
    stackCmd = line[len(transformPrefix) + 1:line.find("(")]
    if line[-2:] != ");":
        raise xasyParseError("Invalid syntax")
    args = line[line.find("(") + 1:-2]
    if stackCmd == "push":
        t = x2a.asyTransform(eval(args))
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
    loc1 = args.rfind(",", 0, loc2 - 1)
    loc = args.rfind(",(", 0, loc1 - 1)
    if loc < 2:
        raise xasyParseError("Invalid syntax")
    text = args[1:loc - 1]
    location = eval(args[loc + 1:args.find("),", loc) + 1])
    pen = args[loc:loc2]
    pen = pen[pen.find(",") + 1:]
    pen = pen[pen.find(",") + 1:]
    pen = pen[pen.find(",") + 1:]
    global pendingTransforms
    return x2a.xasyText(text, location, parsePen(pen), pendingTransforms.pop())


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
    pen = args[loc + 1:]
    global pendingTransforms
    return x2a.xasyShape(parsePathExpression(path), parsePen(pen), pendingTransforms.pop())


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
    pen = args[loc + 1:]
    global pendingTransforms
    return x2a.xasyFilledShape(parsePathExpression(path), parsePen(pen), pendingTransforms.pop())


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
        if len(tokens) > 2:
            options = "+".join(tokens[2:])
        else:
            options = ""
        return x2a.asyPen(color, width, options)
    except:
        raise xasyParseError("Invalid pen")


def parsePathExpression(expr):
    """Parse an asy path returning an asyPath()"""
    result = x2a.asyPath()
    expr = "".join(expr.split())
    # print (expr)
    if expr.find("controls") != -1:
        # parse a path with control points
        tokens = expr.split("..")
        nodes = [a for a in tokens if not a.startswith("controls")]
        for a in range(len(nodes)):
            if nodes[a] != "cycle":
                nodes[a] = eval(nodes[a])
        controls = [[eval(b) for b in a.replace("controls", "").split("and")] for a in tokens if
                    a.startswith("controls")]
        result.initFromControls(nodes, controls)
    else:
        # parse a path without control points
        tokens = re.split(r"(::|--|\.\.)", expr)
        linkSet = re.findall("::|--|\.\.", expr)
        nodeSet = [a for a in tokens if not re.match(r"::|--|\.\.", a)]
        # print (nodeSet)
        for a in range(len(nodeSet)):
            if nodeSet[a] != "cycle":
                nodeSet[a] = eval(nodeSet[a])
        # print (nodeSet)
        result.initFromNodeList(nodeSet, linkSet)
    return result


def takeUntilSemicolon(line, lines):
    """Read and concatenate lines until the collected lines end with a semicolon"""
    data = line
    while not data.endswith(";"):
        newline = lines.pop(0)
        data += newline
    return data


def parseLine(line, lines):
    """Parse a line of the file"""
    if len(line) == 0 or line.isspace() or line.startswith("//"):
        return None
    elif line.startswith(scriptPrefix):
        return extractScript(lines)
    elif line.startswith(transformPrefix):
        return parseTransformExpression(takeUntilSemicolon(line, lines))
    elif line.startswith("label("):
        return parseLabelCommand(takeUntilSemicolon(line, lines))
    elif line.startswith("draw("):
        return parseDrawCommand(takeUntilSemicolon(line, lines))
    elif line.startswith("fill("):
        return parseFillCommand(takeUntilSemicolon(line, lines))
    elif line.startswith("exitXasyMode();"):
        return None
    raise Exception("Could not parse the line")


def saveFile(file, xasyItems):
    """Write a list of xasyItems to a file"""
    for item in xasyItems:
        file.write(item.getTransformCode() + '\n')

    for item in xasyItems:
        file.write(item.getObjectCode() + '\n\n')

