#!/usr/bin/env python3

###########################################################################
#
# xasy2asy provides a Python interface to Asymptote
#
#
# Authors: Orest Shardt, Supakorn Rassameemasmuang, and John C. Bowman
#
###########################################################################

import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import PyQt5.QtSvg as Qs

import numpy as np

import sys
import os
import signal
import threading
import string
import subprocess
import tempfile
import re
import shutil
import copy
import queue
import io
import atexit
import DebugFlags

import xasyUtils as xu
import xasyArgs as xa
import xasyOptions as xo
import xasySvg as xs

class AsymptoteEngine:
    xasy=chr(4)+"\n"

    def __init__(self, path=None, keepFiles=DebugFlags.keepFiles, keepDefaultArgs=True):
        if path is None:
            path = xa.getArgs().asypath
            if path is None:
                opt = xo.BasicConfigs.defaultOpt
                opt.load()
                path = opt['asyPath']

        if sys.platform[:3] == 'win':
            rx = 0  # stdin
            wa = 2  # stderr
        else:
            rx, wx = os.pipe()
            ra, wa = os.pipe()
            os.set_inheritable(rx, True)
            os.set_inheritable(wx, True)
            os.set_inheritable(ra, True)
            os.set_inheritable(wa, True)
            self.ostream = os.fdopen(wx, 'w')
            self.istream = os.fdopen(ra, 'r')
            
        self.keepFiles = keepFiles
        if sys.platform[:3] == 'win':
            self.tmpdir = tempfile.mkdtemp(prefix='xasyData_',dir='./')+'/'
        else:
            self.tmpdir = tempfile.mkdtemp(prefix='xasyData_')+os.sep

        self.args=['-xasy', '-noV', '-q', '-inpipe=' + str(rx), '-outpipe=' + str(wa), '-o', self.tmpdir]

        self.asyPath = path
        self.asyProcess = None

    def start(self):
        try:
            if sys.platform[:3] == 'win':
                self.asyProcess = subprocess.Popen([self.asyPath] + self.args,
                                                stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                                universal_newlines=True)
                self.ostream = self.asyProcess.stdin
                self.istream = self.asyProcess.stderr
            else:
                self.asyProcess = subprocess.Popen([self.asyPath] + self.args,close_fds=False)
        finally:
            atexit.register(self.cleanup)

    def wait(self):
        if self.asyProcess.returncode is not None:
            return
        else:
            return self.asyProcess.wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.wait()

    @property
    def tempDirName(self):
        return self.tmpdir

    def startThenStop(self):
        self.start()
        self.stop()
        self.wait()

    @property
    def active(self):
        if self.asyProcess is None:
            return False
        return self.asyProcess.returncode is None

    def stop(self):
        if self.active:
            self.asyProcess.kill()

    def cleanup(self):
        self.stop()
        if self.asyProcess is not None:
            self.asyProcess.wait()
        if not self.keepFiles:
            if os.path.isdir(self.tempDirName + os.sep):
                shutil.rmtree(self.tempDirName, ignore_errors=True)

class asyTransform(Qc.QObject):
    """A python implementation of an asy transform"""

    def __init__(self, initTuple, delete=False):
        """Initialize the transform with a 6 entry tuple"""
        super().__init__()
        if isinstance(initTuple, (tuple, list)) and len(initTuple) == 6:
            self.t = initTuple
            self.x, self.y, self.xx, self.xy, self.yx, self.yy = initTuple
            self._deleted = delete
        else:
            raise TypeError("Illegal initializer for asyTransform")

    @property
    def deleted(self):
        return self._deleted

    @deleted.setter
    def deleted(self, value):
        self._deleted = value

    @classmethod
    def zero(cls):
        return asyTransform((0, 0, 0, 0, 0, 0))

    @classmethod
    def fromQTransform(cls, transform: Qg.QTransform):
        tx, ty = transform.dx(), transform.dy()
        xx, xy, yx, yy = transform.m11(), transform.m21(), transform.m12(), transform.m22()

        return asyTransform((tx, ty, xx, xy, yx, yy))

    @classmethod
    def fromNumpyMatrix(cls, transform: np.ndarray):
        assert transform.shape == (3, 3)

        tx = transform[0, 2]
        ty = transform[1, 2]

        xx, xy, yx, yy = transform[0:2, 0:2].ravel().tolist()[0]

        return asyTransform((tx, ty, xx, xy, yx, yy))

    def getRawCode(self):
        return xu.tuple2StrWOspaces(self.t)

    def getCode(self, asy2psmap=None):
        """Obtain the asy code that represents this transform"""
        if asy2psmap is None:
            asy2psmap = asyTransform((0, 0, 1, 0, 0, 1))
        if self.deleted:
            return 'zeroTransform'
        else:
            return (asy2psmap.inverted() * self * asy2psmap).getRawCode()

    def scale(self, s):
        return asyTransform((0, 0, s, 0, 0, s)) * self

    def toQTransform(self):
        return Qg.QTransform(self.xx, self.yx, self.xy, self.yy, self.x, self.y)

    def __str__(self):
        """Equivalent functionality to getCode(). It allows the expression str(asyTransform) to be meaningful."""
        return self.getCode()

    def isIdentity(self):
        return self == identity()

    def inverted(self):
        return asyTransform.fromQTransform(self.toQTransform().inverted()[0])

    def __eq__(self, other):
        return list(self.t) == list(other.t)

    def __mul__(self, other):
        """Define multiplication of transforms as composition."""
        if isinstance(other, tuple):
            if len(other) == 6:
                return self * asyTransform(other)
            elif len(other) == 2:
                return ((self.t[0] + self.t[2] * other[0] + self.t[3] * other[1]),
                        (self.t[1] + self.t[4] * other[0] + self.t[5] * other[1]))
            else:
                raise Exception("Illegal multiplier of {:s}".format(str(type(other))))
        elif isinstance(other, asyTransform):
            result = asyTransform((0, 0, 0, 0, 0, 0))
            result.x = self.x + self.xx * other.x + self.xy * other.y
            result.y = self.y + self.yx * other.x + self.yy * other.y
            result.xx = self.xx * other.xx + self.xy * other.yx
            result.xy = self.xx * other.xy + self.xy * other.yy
            result.yx = self.yx * other.xx + self.yy * other.yx
            result.yy = self.yx * other.xy + self.yy * other.yy
            result.t = (result.x, result.y, result.xx, result.xy, result.yx, result.yy)
            return result
        elif isinstance(other, str):
            if other != 'cycle':
                raise TypeError
            else:
                return 'cycle'
        else:
            raise TypeError("Illegal multiplier of {:s}".format(str(type(other))))


def identity():
    return asyTransform((0, 0, 1, 0, 0, 1))

def yflip():
    return asyTransform((0, 0, 1, 0, 0, -1))

class asyObj(Qc.QObject):
    """A base class for asy objects: an item represented by asymptote code."""
    def __init__(self):
        """Initialize the object"""
        super().__init__()
        self.asyCode = ''

    def updateCode(self, ps2asymap=identity()):
        """Update the object's code: should be overriden."""
        raise NotImplementedError

    def getCode(self, ps2asymap=identity()):
        """Return the code describing the object"""
        self.updateCode(ps2asymap)
        return self.asyCode


class asyPen(asyObj):
    """A python wrapper for an asymptote pen"""
    @staticmethod
    def getColorFromQColor(color):
        return color.redF(), color.greenF(), color.blueF()

    @staticmethod
    def convertToQColor(color):
        r, g, b = color
        return Qg.QColor.fromRgbF(r, g, b)

    @classmethod
    def fromAsyPen(cls, pen):
        assert isinstance(pen, cls)
        return cls(asyengine=pen._asyengine, color=pen.color, width=pen.width, pen_options=pen.options)

    def __init__(self, asyengine=None, color=(0, 0, 0), width=0.5, pen_options=""):
        """Initialize the pen"""
        asyObj.__init__(self)
        self.color = (0, 0, 0)
        self.options = pen_options
        self.width = width
        self._asyengine = asyengine
        self._deferAsyfy = False
        if pen_options:
            self._deferAsyfy = True
        self.updateCode()
        self.setColor(color)

    @property
    def asyEngine(self):
        return self._asyengine

    @asyEngine.setter
    def asyEngine(self, value):
        self._asyengine = value

    def updateCode(self, asy2psmap=identity()):
        """Generate the pen's code"""
        if self._deferAsyfy:
            self.computeColor()
        self.asyCode = 'rgb({:g},{:g},{:g})+{:s}'.format(self.color[0], self.color[1], self.color[2], str(self.width))
        if len(self.options) > 0:
            self.asyCode = self.asyCode + '+' + self.options

    def setWidth(self, newWidth):
        """Set the pen's width"""
        self.width = newWidth
        self.updateCode()

    def setColor(self, color):
        """Set the pen's color"""
        if isinstance(color, tuple) and len(color) == 3:
            self.color = color
        else:
            self.color = (0, 0, 0)
        self.updateCode()

    def setColorFromQColor(self, color):
        self.setColor(asyPen.getColorFromQColor(color))

    def computeColor(self):
        """Find out the color of an arbitrary asymptote pen."""
        assert isinstance(self.asyEngine, AsymptoteEngine)
        assert self.asyEngine.active

        fout = self.asyEngine.ostream
        fin = self.asyEngine.istream

        fout.write("pen p=" + self.getCode() + ';\n')
        fout.write("write(_outpipe,colorspace(p),newl);\n")
        fout.write("write(_outpipe,colors(p));\n")
        fout.write("flush(_outpipe);\n")
        fout.write(self.asyEngine.xasy)
        fout.flush()

        colorspace = fin.readline()
        if colorspace.find("cmyk") != -1:
            lines = fin.readline() + fin.readline() + fin.readline() + fin.readline()
            parts = lines.split()
            c, m, y, k = eval(parts[0]), eval(parts[1]), eval(parts[2]), eval(parts[3])
            k = 1 - k
            r, g, b = ((1 - c) * k, (1 - m) * k, (1 - y) * k)
        elif colorspace.find("rgb") != -1:
            lines = fin.readline() + fin.readline() + fin.readline()
            parts = lines.split()
            r, g, b = eval(parts[0]), eval(parts[1]), eval(parts[2])
        elif colorspace.find("gray") != -1:
            lines = fin.readline()
            parts = lines.split()
            r = g = b = eval(parts[0])
        else:
            raise ChildProcessError('Asymptote error.')
        self.color = (r, g, b)
        self._deferAsyfy = False

    def tkColor(self):
        """Return the tk version of the pen's color"""
        self.computeColor()
        return '#{}'.format("".join(["{:02x}".format(min(int(256 * a), 255)) for a in self.color]))

    def toQPen(self):
        if self._deferAsyfy:
            self.computeColor()
        newPen = Qg.QPen()
        newPen.setColor(asyPen.convertToQColor(self.color))
        newPen.setWidthF(self.width)

        return newPen


class asyPath(asyObj):
    """A python wrapper for an asymptote path"""

    def __init__(self, asyengine: AsymptoteEngine=None, forceCurve=False):
        """Initialize the path to be an empty path: a path with no nodes, control points, or links."""
        super().__init__()
        self.nodeSet = []
        self.linkSet = []
        self.forceCurve = forceCurve
        self.controlSet = []
        self.computed = False
        self.asyengine = asyengine

    @classmethod
    def fromPath(cls, oldPath):
        newObj = asyPath(None)
        newObj.nodeSet = copy.copy(oldPath.nodeSet)
        newObj.linkSet = copy.copy(oldPath.linkSet)
        newObj.controlSet = copy.deepcopy(oldPath.controlSet)
        newObj.computed = oldPath.computed
        newObj.asyengine = oldPath.asyengine

        return newObj

    @classmethod
    def fromBezierPoints(cls, pointList: list, engine=None):
        if not pointList:
            return None
        assert isinstance(pointList[0], BezierCurveEditor.BezierPoint)
        nodeList = []
        controlList = []
        for point in pointList:
            nodeList.append(BezierCurveEditor.QPoint2Tuple(point.point))
            if point.rCtrlPoint is not None:  # first
                controlList.append([BezierCurveEditor.QPoint2Tuple(point.rCtrlPoint)])
            if point.lCtrlPoint is not None:  # last
                controlList[-1].append(BezierCurveEditor.QPoint2Tuple(point.lCtrlPoint))
        newPath = asyPath(asyengine=engine)
        newPath.initFromControls(nodeList, controlList)
        return newPath

    def setInfo(self, path):
        self.nodeSet = copy.copy(path.nodeSet)
        self.linkSet = copy.copy(path.linkSet)
        self.controlSet = copy.deepcopy(path.controlSet)
        self.computed = path.computed

    @property
    def isEmpty(self):
        return len(self.nodeSet) == 0

    @property
    def isDrawable(self):
        return len(self.nodeSet) >= 2

    def toQPainterPath(self) -> Qg.QPainterPath:
        return self.toQPainterPathCurve() if self.containsCurve else self.toQPainterPathLine()

    def toQPainterPathLine(self):
        baseX, baseY = self.nodeSet[0]
        painterPath = Qg.QPainterPath(Qc.QPointF(baseX, baseY))

        for pointIndex in range(1, len(self.nodeSet)):
            node = self.nodeSet[pointIndex]
            if self.nodeSet[pointIndex] == 'cycle':
                node = self.nodeSet[0]

            painterPath.lineTo(*node)

        return painterPath


    def toQPainterPathCurve(self):
        if not self.computed:
            self.computeControls()

        baseX, baseY = self.nodeSet[0]
        painterPath = Qg.QPainterPath(Qc.QPointF(baseX, baseY))

        for pointIndex in range(1, len(self.nodeSet)):
            node = self.nodeSet[pointIndex]
            if self.nodeSet[pointIndex] == 'cycle':
                node = self.nodeSet[0]
            endPoint = Qc.QPointF(node[0], node[1])
            ctrlPoint1 = Qc.QPointF(self.controlSet[pointIndex-1][0][0], self.controlSet[pointIndex-1][0][1])
            ctrlPoint2 = Qc.QPointF(self.controlSet[pointIndex-1][1][0], self.controlSet[pointIndex-1][1][1])

            painterPath.cubicTo(ctrlPoint1, ctrlPoint2, endPoint)
        return painterPath

    def initFromNodeList(self, nodeSet, linkSet):
        """Initialize the path from a set of nodes and link types, "--", "..", or "::" """
        if len(nodeSet) > 0:
            self.nodeSet = nodeSet[:]
            self.linkSet = linkSet[:]
            self.computed = False

    def initFromControls(self, nodeSet, controlSet):
        """Initialize the path from nodes and control points"""
        self.controlSet = controlSet[:]
        self.nodeSet = nodeSet[:]
        self.computed = True

    def makeNodeStr(self, node):
        """Represent a node as a string"""
        if node == 'cycle':
            return node
        else:
            # if really want to, disable this rounding
            # shouldn't be to much of a problem since 10e-6 is quite small... 
            return '({:.6g},{:.6g})'.format(node[0], node[1])

    def updateCode(self, ps2asymap=identity()):
        """Generate the code describing the path"""
        # currently at postscript. Convert to asy
        asy2psmap =  ps2asymap.inverted()
        with io.StringIO() as rawAsyCode:
            count = 0
            rawAsyCode.write(self.makeNodeStr(asy2psmap * self.nodeSet[0]))
            for node in self.nodeSet[1:]:
                if not self.computed or count >= len(self.controlSet):
                    rawAsyCode.write(self.linkSet[count])
                    rawAsyCode.write(self.makeNodeStr(asy2psmap * node))
                else:
                    rawAsyCode.write('..controls ')
                    rawAsyCode.write(self.makeNodeStr(asy2psmap *  self.controlSet[count][0]))
                    rawAsyCode.write(' and ')
                    rawAsyCode.write(self.makeNodeStr(asy2psmap * self.controlSet[count][1]))
                    rawAsyCode.write(".." + self.makeNodeStr(asy2psmap * node))
                count = count + 1
            self.asyCode = rawAsyCode.getvalue()

    @property
    def containsCurve(self):
        return '..' in self.linkSet or self.forceCurve

    def getNode(self, index):
        """Return the requested node"""
        return self.nodeSet[index]

    def getLink(self, index):
        """Return the requested link"""
        return self.linkSet[index]

    def setNode(self, index, newNode):
        """Set a node to a new position"""
        self.nodeSet[index] = newNode

    def moveNode(self, index, offset):
        """Translate a node"""
        if self.nodeSet[index] != "cycle":
            self.nodeSet[index] = (self.nodeSet[index][0] + offset[0], self.nodeSet[index][1] + offset[1])

    def setLink(self, index, ltype):
        """Change the specified link"""
        self.linkSet[index] = ltype

    def addNode(self, point, ltype):
        """Add a node to the end of a path"""
        self.nodeSet.append(point)
        if len(self.nodeSet) != 1:
            self.linkSet.append(ltype)
        if self.computed:
            self.computeControls()

    def insertNode(self, index, point, ltype=".."):
        """Insert a node, and its corresponding link, at the given index"""
        self.nodeSet.insert(index, point)
        self.linkSet.insert(index, ltype)
        if self.computed:
            self.computeControls()

    def setControl(self, index, position):
        """Set a control point to a new position"""
        self.controlSet[index] = position

    def popNode(self):
        if len(self.controlSet) == len(self.nodeSet):
            self.controlSet.pop()
        self.nodeSet.pop()
        self.linkSet.pop()

    def moveControl(self, index, offset):
        """Translate a control point"""
        self.controlSet[index] = (self.controlSet[index][0] + offset[0], self.controlSet[index][1] + offset[1])

    def computeControls(self):
        """Evaluate the code of the path to obtain its control points"""
        # For now, if no asymptote process is given spawns a new one.
        # Only happens if asyengine is None.
        if self.asyengine is not None:
            assert isinstance(self.asyengine, AsymptoteEngine)
            assert self.asyengine.active
            asy = self.asyengine
            startUp = False
        else:
            startUp = True
            asy = AsymptoteEngine()
            asy.start()

        fout = asy.ostream
        fin = asy.istream

        fout.write("path p=" + self.getCode() + ';\n')
        fout.write("write(_outpipe,length(p),newl);\n")
        fout.write("write(_outpipe,unstraighten(p),endl);\n")
        fout.write(asy.xasy)
        fout.flush()

        lengthStr = fin.readline()
        pathSegments = eval(lengthStr.split()[-1])
        pathStrLines = []
        for i in range(pathSegments + 1):
            line = fin.readline()
            line = line.replace("\n", "")
            pathStrLines.append(line)
        oneLiner = "".join(pathStrLines).replace(" ", "")
        splitList = oneLiner.split("..")
        nodes = [a for a in splitList if a.find("controls") == -1]
        self.nodeSet = []
        for a in nodes:
            if a == 'cycle':
                self.nodeSet.append(a)
            else:
                self.nodeSet.append(eval(a))
        controls = [a.replace("controls", "").split("and") for a in splitList if a.find("controls") != -1]
        self.controlSet = [[eval(a[0]), eval(a[1])] for a in controls]
        self.computed = True

        if startUp:
            asy.stop()

class asyLabel(asyObj):
    """A python wrapper for an asy label"""

    def __init__(self, text="", location=(0, 0), pen=None, align=None, fontSize:int=None):
        """Initialize the label with the given test, location, and pen"""
        asyObj.__init__(self)
        self.align = align
        self.pen = pen
        self.fontSize = fontSize
        if align is None:
            self.align = 'SE'
        if pen is None:
            self.pen = asyPen()
        self.text = text
        self.location = location

    def updateCode(self, asy2psmap=identity()):
        """Generate the code describing the label"""
        newLoc = asy2psmap.inverted() * self.location
        locStr = xu.tuple2StrWOspaces(newLoc)
        self.asyCode = 'Label("{0}",{1},p={2}{4},align={3})'.format(self.text, locStr, self.pen.getCode(), self.align,
        self.getFontSizeText())

    def getFontSizeText(self):
        if self.fontSize is not None:
            return '+fontsize({:.6g})'.format(self.fontSize)
        else:
            return ''

    def setText(self, text):
        """Set the label's text"""
        self.text = text
        self.updateCode()

    def setPen(self, pen):
        """Set the label's pen"""
        self.pen = pen
        self.updateCode()

    def moveTo(self, newl):
        """Translate the label's location"""
        self.location = newl


class asyImage:
    """A structure containing an image and its format, bbox, and IDTag"""
    def __init__(self, image, format, bbox, transfKey=None, keyIndex=0):
        self.image = image
        self.format = format
        self.bbox = bbox
        self.IDTag = None
        self.key = transfKey
        self.keyIndex = keyIndex

class xasyItem(Qc.QObject):
    """A base class for items in the xasy GUI"""
    mapString = 'xmap'
    setKeyFormatStr = string.Template('$map("{:s}",{:s});').substitute(map=mapString)
    setKeyAloneFormatStr = string.Template('$map("{:s}");').substitute(map=mapString)
    resizeComment="// Resize to initial xasy transform"
    asySize=""
    def __init__(self, canvas=None, asyengine=None):
        """Initialize the item to an empty item"""
        super().__init__()
        self.transfKeymap = {}              # the new keymap.
        # should be a dictionary to a list...
        self.asyCode = ''
        self.imageList = []
        self.IDTag = None
        self.asyfied = False
        self.onCanvas = canvas
        self.keyBuffer = None
        self._asyengine = asyengine
        self.drawObjects = []
        self.drawObjectsMap = {}
        self.setKeyed = True
        self.unsetKeys = set()
        self.userKeys = set()
        self.lineOffset = 0
        self.imageHandleQueue = queue.Queue()

    def updateCode(self, ps2asymap=identity()):
        """Update the item's code: to be overriden"""
        with io.StringIO() as rawCode:
            transfCode = self.getTransformCode()
            objCode = self.getObjectCode()

            rawCode.write(transfCode)
            rawCode.write(objCode)
            self.asyCode = rawCode.getvalue() 

        return len(transfCode.splitlines()), len(objCode.splitlines())

    @property
    def asyengine(self):
        return self._asyengine

    @asyengine.setter
    def asyengine(self, value):
        self._asyengine = value

    def getCode(self, ps2asymap=identity()):
        """Return the code describing the item"""
        self.updateCode(ps2asymap)
        return self.asyCode

    def getTransformCode(self, asy2psmap=identity()):
        raise NotImplementedError

    def getObjectCode(self, asy2psmap=identity()):
        raise NotImplementedError

    def generateDrawObjects(self):
        raise NotImplementedError

    def handleImageReception(self, file, fileformat, bbox, count, key=None, localCount=0, containsClip=False):
        """Receive an image from an asy deconstruction. It replaces the default n asyProcess."""
        # image = Image.open(file).transpose(Image.FLIP_TOP_BOTTOM)
        if fileformat == 'png':
            image = Qg.QImage(file)
        elif fileformat == 'svg':
            if containsClip:
                image = xs.SvgObject(file)
            else:
                image = Qs.QSvgRenderer(file)
                assert image.isValid()
        else:
            raise Exception('Format not supported!')
        self.imageList.append(asyImage(image, fileformat, bbox, transfKey=key, keyIndex=localCount))
        if self.onCanvas is not None:
            # self.imageList[-1].iqt = ImageTk.PhotoImage(image)
            currImage = self.imageList[-1]
            currImage.iqt = image
            currImage.originalImage = image
            currImage.originalImage.theta = 0.0
            currImage.originalImage.bbox = list(bbox)
            currImage.performCanvasTransform = False

            # handle this case if transform is not in the map yet.
            # if deleted - set transform to 0, 0, 0, 0, 0
            transfExists = key in self.transfKeymap.keys()
            if transfExists:
                transfExists = localCount <= len(self.transfKeymap[key]) - 1
                if transfExists:
                    validKey = not self.transfKeymap[key][localCount].deleted
            else:
                validKey = False

            if (not transfExists) or validKey:
                currImage.IDTag = str(file)
                newDrawObj = DrawObject(currImage.iqt, self.onCanvas['canvas'], transform=identity(),
                                        btmRightanchor=Qc.QPointF(bbox[0], bbox[2]), drawOrder=-1, key=key,
                                        parentObj=self, keyIndex=localCount)
                newDrawObj.setBoundingBoxPs(bbox)
                newDrawObj.setParent(self)

                self.drawObjects.append(newDrawObj)

                if key not in self.drawObjectsMap.keys():
                    self.drawObjectsMap[key] = [newDrawObj]
                else:
                    self.drawObjectsMap[key].append(newDrawObj)
        return containsClip
    def asyfy(self, force=False):
        if self.asyengine is None:
            return 1
        if self.asyfied and not force:
            return

        self.drawObjects = []
        self.drawObjectsMap.clear()
        assert isinstance(self.asyengine, AsymptoteEngine)
        self.imageList = []

        self.unsetKeys.clear()
        self.userKeys.clear()

        self.imageHandleQueue = queue.Queue()
        worker = threading.Thread(target=self.asyfyThread, args=[])
        worker.start()
        item = self.imageHandleQueue.get()
        cwd=os.getcwd();
        os.chdir(self.asyengine.tempDirName)
        while item != (None,) and item[0] != "ERROR":
            if item[0] == "OUTPUT":
                print(item[1])
            else:
                keepFile = self.handleImageReception(*item)
                if not DebugFlags.keepFiles and not keepFile:
                    try:
                        os.remove(item[0])
                        pass
                    except OSError:
                        pass
                    finally:
                        pass
            item = self.imageHandleQueue.get()
        # self.imageHandleQueue.task_done()
        os.chdir(cwd);

        worker.join()

    def asyfyThread(self):
        """Convert the item to a list of images by deconstructing this item's code"""
        assert self.asyengine.active

        fout = self.asyengine.ostream
        fin = self.asyengine.istream

        self.lineOffset = len(self.getTransformCode().splitlines())

        fout.write("reset\n")
        fout.flush();
        for line in self.getCode().splitlines():
            if DebugFlags.printDeconstTranscript:
                print('fout:', line)
            fout.write(line+"\n")
        fout.write(self.asySize)
        fout.write("deconstruct();\n")
        fout.write('write(_outpipe,yscale(-1)*currentpicture.calculateTransform(),endl);\n')
        fout.write(self.asyengine.xasy)
        fout.flush()

        imageInfos = []                                 # of (box, key)
        n = 0

        keyCounts = {}

        def render():
            for i in range(len(imageInfos)):
                box, key, localCount, useClip = imageInfos[i]
                l, b, r, t = [float(a) for a in box.split()]
                name = "{:s}_{:d}.{:s}".format(self.asyengine.tempDirName, i, fileformat)

                self.imageHandleQueue.put((name, fileformat, (l, -t, r, -b), i, key, localCount, useClip))

        # key first, box second.
        # if key is "Done"
        raw_text = fin.readline()
        text = ""
        if DebugFlags.printDeconstTranscript:
            print(raw_text.strip())

        # template=AsyTempDir+"%d_%d.%s"
        fileformat = 'svg'

        while raw_text != "Done\n" and raw_text != "Error\n":
#            print(raw_text)
            text = fin.readline()       # the actual bounding box.
            # print('TESTING:', text)
            keydata = raw_text.strip().replace('KEY=', '', 1)  # key

            clipflag = keydata[-1] == '1'
            userkey = keydata[-2] == '1'
            keydata = keydata[:-3]

            if not userkey:
                self.unsetKeys.add(keydata)     # the line and column to replace. 
            else:
                self.userKeys.add(keydata)
            
#                print(line, col)

            if keydata not in keyCounts.keys():
                keyCounts[keydata] = 0

            imageInfos.append((text, keydata, keyCounts[keydata], clipflag))      # key-data pair

            # for the next item
            keyCounts[keydata] += 1

            raw_text = fin.readline()

            if DebugFlags.printDeconstTranscript:
                print(text.rstrip())
                print(raw_text.rstrip())

            n += 1

        if text == "Error\n":
            self.imageHandleQueue.put(("ERROR", fin.readline()))
        else:
            render()

        self.asy2psmap = asyTransform(xu.listize(fin.readline().rstrip(),float))
        self.imageHandleQueue.put((None,))
        self.asyfied = True

class xasyDrawnItem(xasyItem):
    """A base class for GUI items was drawn by the user. It combines a path, a pen, and a transform."""

    def __init__(self, path, engine, pen=None, transform=identity(), key=None):
        """Initialize the item with a path, pen, and transform"""
        super().__init__(canvas=None, asyengine=engine)
        if pen is None:
            pen = asyPen()
        self.path = path
        self.path.asyengine = engine
        self.asyfied = True
        self.pen = pen
        self._asyengine = engine
        self.rawIdentifier = ''
        self.transfKey = key
        self.transfKeymap = {self.transfKey: [transform]}

    @property
    def asyengine(self):
        return self._asyengine

    @asyengine.setter
    def asyengine(self, value: AsymptoteEngine):
        self._asyengine = value
        self.path.asyengine = value

    def setKey(self, newKey=None):
        transform = self.transfKeymap[self.transfKey][0]

        self.transfKey = newKey
        self.transfKeymap = {self.transfKey: [transform]}

    def generateDrawObjects(self, forceUpdate=False):
        raise NotImplementedError

    def appendPoint(self, point, link=None):
        """Append a point to the path. If the path is cyclic, add this point before the 'cycle' node."""
        if self.path.nodeSet[-1] == 'cycle':
            self.path.nodeSet[-1] = point
            self.path.nodeSet.append('cycle')
        else:
            self.path.nodeSet.append(point)
        self.path.computed = False
        self.asyfied = False
        if len(self.path.nodeSet) > 1 and link is not None:
            self.path.linkSet.append(link)

    def clearTransform(self):
        """Reset the item's transform"""
        self.transform = [identity()]
        self.asyfied = False

    def removeLastPoint(self):
        """Remove the last point in the path. If the path is cyclic, remove the node before the 'cycle' node."""
        if self.path.nodeSet[-1] == 'cycle':
            del self.path.nodeSet[-2]
        else:
            del self.path.nodeSet[-1]
        del self.path.linkSet[-1]
        self.path.computed = False
        self.asyfied = False

    def setLastPoint(self, point):
        """Modify the last point in the path. If the path is cyclic, modify the node before the 'cycle' node."""
        if self.path.nodeSet[-1] == 'cycle':
            self.path.nodeSet[-2] = point
        else:
            self.path.nodeSet[-1] = point
        self.path.computed = False
        self.asyfied = False


class xasyShape(xasyDrawnItem):
    """An outlined shape drawn on the GUI"""
    def __init__(self, path, asyengine, pen=None, transform=identity()):
        """Initialize the shape with a path, pen, and transform"""
        super().__init__(path=path, engine=asyengine, pen=pen, transform=transform)

    def getObjectCode(self, asy2psmap=identity()):
        return 'draw(KEY="{0}",{1},{2});'.format(self.transfKey, self.path.getCode(asy2psmap), self.pen.getCode())+'\n\n'

    def getTransformCode(self, asy2psmap=identity()):
        transf = self.transfKeymap[self.transfKey][0]
        if transf == identity():
            return ''
        else:
            return xasyItem.setKeyFormatStr.format(self.transfKey, transf.getCode(asy2psmap))+'\n'

    def generateDrawObjects(self, forceUpdate=False):
        if self.path.containsCurve:
            self.path.computeControls()
        transf = self.transfKeymap[self.transfKey][0]

        newObj = DrawObject(self.path.toQPainterPath(), None, drawOrder=0, transform=transf, pen=self.pen,
                            key=self.transfKey)
        newObj.originalObj = self
        newObj.setParent(self)
        return [newObj]

    def __str__(self):
        """Create a string describing this shape"""
        return "xasyShape code:{:s}".format("\n\t".join(self.getCode().splitlines()))


class xasyFilledShape(xasyShape):
    """A filled shape drawn on the GUI"""

    def __init__(self, path, asyengine, pen=None, transform=identity()):
        """Initialize this shape with a path, pen, and transform"""
        if path.nodeSet[-1] != 'cycle':
            raise Exception("Filled paths must be cyclic")
        super().__init__(path, asyengine, pen, transform)

    def getObjectCode(self, asy2psmap=identity()):
        return 'fill(KEY="{0}",{1},{2});'.format(self.transfKey, self.path.getCode(asy2psmap), self.pen.getCode())+'\n\n'

    def generateDrawObjects(self, forceUpdate=False):
        if self.path.containsCurve:
            self.path.computeControls()
        newObj = DrawObject(self.path.toQPainterPath(), None, drawOrder=0, transform=self.transfKeymap[self.transfKey][0],
                            pen=self.pen, key=self.transfKey, fill=True)
        newObj.originalObj = self
        newObj.setParent(self)
        return [newObj]

    def __str__(self):
        """Return a string describing this shape"""
        return "xasyFilledShape code:{:s}".format("\n\t".join(self.getCode().splitlines()))


class xasyText(xasyItem):
    """Text created by the GUI"""

    def __init__(self, text, location, asyengine, pen=None, transform=yflip(), key=None, align=None, fontsize:int=None):
        """Initialize this item with text, a location, pen, and transform"""
        super().__init__(asyengine=asyengine)
        if pen is None:
            pen = asyPen(asyengine=asyengine)
        if pen.asyEngine is None:
            pen.asyEngine = asyengine
        self.label = asyLabel(text, location, pen, align, fontSize=fontsize)
        # self.transform = [transform]
        self.transfKey = key
        self.transfKeymap = {self.transfKey: [transform]}
        self.asyfied = False
        self.onCanvas = None
    
    def setKey(self, newKey=None):
        transform = self.transfKeymap[self.transfKey][0]

        self.transfKey = newKey
        self.transfKeymap = {self.transfKey: [transform]}

    def getTransformCode(self, asy2psmap=yflip()):
        transf = self.transfKeymap[self.transfKey][0]
        if transf == yflip():
            # return xasyItem.setKeyAloneFormatStr.format(self.transfKey)
            return ''
        else:
            return xasyItem.setKeyFormatStr.format(self.transfKey, transf.getCode(asy2psmap))+"\n"

    def getObjectCode(self, asy2psmap=yflip()):
        return 'label(KEY="{0}",{1});'.format(self.transfKey, self.label.getCode(asy2psmap))+'\n'

    def generateDrawObjects(self, forceUpdate=False):
        self.asyfy(forceUpdate)
        return self.drawObjects

    def getBoundingBox(self):
        self.asyfy()
        return self.imageList[0].bbox

    def __str__(self):
        return "xasyText code:{:s}".format("\n\t".join(self.getCode().splitlines()))


class xasyScript(xasyItem):
    """A set of images create from asymptote code. It is always deconstructed."""

    def __init__(self, canvas, engine, script="", transforms=None, transfKeyMap=None):
        """Initialize this script item"""
        super().__init__(canvas, asyengine=engine)
        if transfKeyMap is not None:
            self.transfKeymap = transfKeyMap
        else:
            self.transfKeymap = {}

        self.script = script
        self.key2imagemap = {}
        self.namedUnsetKeys = {}
        self.keyPrefix = ''
        self.scriptAsyfied = False
        self.updatedPrefix = True

    def clearTransform(self):
        """Reset the transforms for each of the deconstructed images"""
        # self.transform = [identity()] * len(self.imageList)
        keyCount = {}

        for im in self.imageList:
            if im.key not in keyCount.keys():
                keyCount[im.key] = 1
            else:
                keyCount[im.key] += 1

        for key in keyCount:
            self.transfKeymap[key] = [identity()] * keyCount[key]

    def getMaxKeyCounter(self):
        maxCounter = -1
        for key in self.transfKeymap:
            testNum = re.match(r'^x(\d+)$', key)
            if testNum is not None:
                maxCounter = max(maxCounter, int(testNum.group(1)))
        return maxCounter + 1

    def getTransformCode(self, asy2psmap=identity()):
        with io.StringIO() as rawAsyCode:
            if self.transfKeymap:
                for key in self.transfKeymap.keys():
                    val = self.transfKeymap[key]

                    writeval = list(reversed(val))
                    # need to map all transforms in a list if there is any non-identity
                    # unfortunately, have to check all transformations in the list. 
                    while not all(checktransf == identity() for checktransf in writeval) and writeval:
                        transf = writeval.pop()
                        if transf.deleted:
                            rawAsyCode.write(xasyItem.setKeyFormatStr.format(key, transf.getCode(asy2psmap)) + '\n//')
                        if transf == identity() and not transf.deleted:
                            rawAsyCode.write(xasyItem.setKeyAloneFormatStr.format(key))
                        else:
                            rawAsyCode.write(xasyItem.setKeyFormatStr.format(key, transf.getCode(asy2psmap)))
                        rawAsyCode.write('\n')
            result = rawAsyCode.getvalue()
        return result

    def findNonIdKeys(self):
        return {key for key in self.transfKeymap if not all(transf == identity() for transf in self.transfKeymap[key]) }

    def getObjectCode(self, asy2psmap=identity()):
        numeric=r'([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?)))'
        rSize=re.compile("size\(\("+numeric+","+numeric+","+numeric+","
                         +numeric+","+numeric+","+numeric+"\)\); "+
                         self.resizeComment)

        newScript = self.getReplacedKeysCode(self.findNonIdKeys())
        with io.StringIO() as rawAsyCode:
            for line in newScript.splitlines():
                if(rSize.match(line)):
                    self.asySize=line.rstrip()+'\n'
                else:
                    raw_line = line.rstrip().replace('\t', ' ' * 4)
                    rawAsyCode.write(raw_line + '\n')

            self.updatedCode = rawAsyCode.getvalue()
            return self.updatedCode

    def setScript(self, script):
        """Sets the content of the script item."""
        self.script = script
        self.updateCode()

    def setKeyPrefix(self, newPrefix=''):
        self.keyPrefix = newPrefix
        self.updatedPrefix = False

    def getReplacedKeysCode(self, key2replace: set=None) -> str:
        keylist = {}
        prefix = ''
        
        key2replaceSet = self.unsetKeys if key2replace is None else \
                        self.unsetKeys & key2replace

        linenum2key = {}

        if not self.updatedPrefix:
            prefix = self.keyPrefix

        for key in key2replaceSet:
            actualkey = key

            key = key.split(':')[0]
            raw_parsed = xu.tryParseKey(key)
            assert raw_parsed is not None
            line, col = [int(val) for val in raw_parsed.groups()]
            if line not in keylist:
                keylist[line] = set()
            keylist[line].add(col)
            linenum2key[(line, col)] = actualkey
            self.unsetKeys.discard(key)


        raw_code_lines = self.script.splitlines()
        with io.StringIO() as raw_str:
            for i_0 in range(len(raw_code_lines)):
                i = i_0 + self.lineOffset
                curr_str = raw_code_lines[i_0]
                if i + 1 in keylist.keys():
                    # this case, we have a key.
                    with io.StringIO() as raw_line:
                        for j in range(len(curr_str)):
                            raw_line.write(curr_str[j])
                            if j + 1 in keylist[i + 1]:
                                # at this point, replace keys with xkey
                                raw_line.write('KEY="{0:s}",'.format(linenum2key[(i + 1, j + 1)]))
                                self.userKeys.add(linenum2key[(i + 1, j + 1)])
                        curr_str = raw_line.getvalue()
                # else, skip and just write the line.
                raw_str.write(curr_str + '\n')
            return raw_str.getvalue()

    def getUnusedKey(self, oldkey) -> str:
        baseCounter = 0
        newKey = oldkey
        while newKey in self.userKeys:
            newKey = oldkey + ':' + str(baseCounter)
            baseCounter += 1
        return newKey

    def asyfy(self, keyOnly=False):
        """Generate the list of images described by this object and adjust the length of the transform list."""
        super().asyfy()

        # Id --> Transf --> asy-fied --> Transf
        # Transf should keep the original, raw transformation
        # but for all new drawn objects - assign Id as transform.

        if self.scriptAsyfied:
            return

        keyCount = {}
        settedKey = {}

        for im in self.imageList:
            if im.key in self.unsetKeys and im.key not in settedKey.keys():
                oldkey = im.key
                self.unsetKeys.remove(im.key)
                im.key = self.getUnusedKey(im.key)
                self.unsetKeys.add(im.key)

                for drawobj in self.drawObjectsMap[oldkey]:
                    drawobj.key = im.key

                self.drawObjectsMap[im.key] = self.drawObjectsMap[oldkey]
                self.drawObjectsMap.pop(oldkey)

                settedKey[oldkey] = im.key
            elif im.key in settedKey.keys():
                im.key = settedKey[im.key]

            if im.key not in keyCount.keys():
                keyCount[im.key] = 1
            else:
                keyCount[im.key] += 1

            if im.key not in self.key2imagemap.keys():
                self.key2imagemap[im.key] = [im]
            else:
                self.key2imagemap[im.key].append(im)

            

        for key in keyCount:
            if key not in self.transfKeymap.keys():
                self.transfKeymap[key] = [identity()] * keyCount[key]
            else:
                while len(self.transfKeymap[key]) < keyCount[key]:
                    self.transfKeymap[key].append(identity())

                # while len(self.transfKeymap[key]) > keyCount[key]:
                    # self.transfKeymap[key].pop()

        # change of basis 
        for keylist in self.transfKeymap.values():
            for i in range(len(keylist)):
                if keylist[i] != identity():
                    keylist[i] = self.asy2psmap * keylist[i] * self.asy2psmap.inverted()

        self.updateCode()
        self.scriptAsyfied = True

    def generateDrawObjects(self, forceUpdate=False):
        self.asyfy(forceUpdate)
        return self.drawObjects

    def __str__(self):
        """Return a string describing this script"""
        retVal = "xasyScript\n\tTransforms:\n"
        for xform in self.transform:
            retVal += "\t" + str(xform) + "\n"
        retVal += "\tCode Ommitted"
        return retVal


class DrawObject(Qc.QObject):
    def __init__(self, drawObject, mainCanvas=None, transform=identity(), btmRightanchor=Qc.QPointF(0, 0),
                 drawOrder=(-1, -1), pen=None, key=None, parentObj=None, fill=False, keyIndex=0):
        super().__init__()
        self.drawObject = drawObject
        self.mainCanvas = mainCanvas
        self.pTransform = transform
        self.baseTransform = transform
        self.drawOrder = drawOrder
        self.btmRightAnchor = btmRightanchor
        self.originalObj = parentObj
        self.explicitBoundingBox = None
        self.useCanvasTransformation = False
        self.key = key
        self.cachedSvgImg = None
        self.cachedDPI = None
        self.maxDPI=0
        self.keyIndex = keyIndex
        self.pen = pen
        self.fill = fill

    def getInteriorScrTransform(self, transform):
        """Generates the transform with Interior transform applied beforehand."""
        if isinstance(transform, Qg.QTransform):
            transform = asyTransform.fromQTransform(transform)
        return self.transform * transform * self.baseTransform.inverted()

    @property
    def transform(self):
        return self.pTransform

    @transform.setter
    def transform(self, value):
        self.pTransform = value

    def setBoundingBoxPs(self, bbox):
        l, b, r, t = bbox
        self.explicitBoundingBox = Qc.QRectF(Qc.QPointF(l, b), Qc.QPointF(r, t))
        # self.explicitBoundingBox = Qc.QRectF(0, 0, 100, 100)

    @property
    def boundingBox(self):
        if self.explicitBoundingBox is not None:
            testBbox = self.explicitBoundingBox
        else:
            if isinstance(self.drawObject, Qg.QImage):
                testBbox = self.drawObject.rect()
                testBbox.moveTo(self.btmRightAnchor.toPoint())
            elif isinstance(self.drawObject, Qg.QPainterPath):
                testBbox = self.baseTransform.toQTransform().mapRect(self.drawObject.boundingRect())
            else:
                raise TypeError('drawObject is not a valid type!')
        pointList = [self.getScreenTransform().toQTransform().map(point) for point in [
            testBbox.topLeft(), testBbox.topRight(), testBbox.bottomLeft(), testBbox.bottomRight()
        ]]
        return Qg.QPolygonF(pointList).boundingRect()

    @property
    def localBoundingBox(self):
        testBbox = self.drawObject.rect()
        testBbox.moveTo(self.btmRightAnchor.toPoint())
        return testBbox

    def getScreenTransform(self):
        scrTransf = self.baseTransform.toQTransform().inverted()[0] * self.pTransform.toQTransform()
        return asyTransform.fromQTransform(scrTransf)

    def draw(self, additionalTransformation=None, applyReverse=False, canvas: Qg.QPainter=None, dpi=300):
        if canvas is None:
            canvas = self.mainCanvas
        if additionalTransformation is None:
            additionalTransformation = Qg.QTransform()
            
        assert canvas.isActive()

        canvas.save()
        if self.pen:
            oldPen = Qg.QPen(canvas.pen())
            canvas.setPen(self.pen.toQPen())
        else:
            oldPen = Qg.QPen()

        if not applyReverse:
            canvas.setTransform(additionalTransformation, True)
            canvas.setTransform(self.transform.toQTransform(), True)
        else:
            canvas.setTransform(self.transform.toQTransform(), True)
            canvas.setTransform(additionalTransformation, True)

        canvas.setTransform(self.baseTransform.toQTransform().inverted()[0], True)

        if isinstance(self.drawObject, Qg.QImage):
            canvas.drawImage(self.explicitBoundingBox, self.drawObject)
        elif isinstance(self.drawObject, xs.SvgObject):
            threshold = 1.44
            
            if self.cachedDPI is None or self.cachedSvgImg is None \
               or dpi > self.maxDPI*threshold:
                self.cachedDPI = dpi
                self.maxDPI=max(self.maxDPI,dpi)
                self.cachedSvgImg = self.drawObject.render(dpi)

            canvas.drawImage(self.explicitBoundingBox, self.cachedSvgImg)
        elif isinstance(self.drawObject, Qs.QSvgRenderer):
            self.drawObject.render(canvas, self.explicitBoundingBox)
        elif isinstance(self.drawObject, Qg.QPainterPath):
            path = self.baseTransform.toQTransform().map(self.drawObject)
            if self.fill:
                if self.pen:
                    brush = self.pen.toQPen().brush()
                else:
                    brush = Qg.QBrush()
                canvas.fillPath(path, brush)
            else:
                canvas.drawPath(path)

        if self.pen:
            canvas.setPen(oldPen)
        canvas.restore()

    def collide(self, coords, canvasCoordinates=True):
        # modify these values to grow/shrink the fuzz. 
        fuzzTolerance = 1
        marginGrowth = 1
        leftMargin = marginGrowth if self.boundingBox.width() < fuzzTolerance else 0
        topMargin = marginGrowth if self.boundingBox.height() < fuzzTolerance else 0

        newMargin = Qc.QMarginsF(leftMargin, topMargin, leftMargin, topMargin)
        return self.boundingBox.marginsAdded(newMargin).contains(coords)

    def getID(self):
        return self.originalObj
