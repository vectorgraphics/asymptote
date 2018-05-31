###########################################################################
#
# xasy2asy provides a Python interface to Asymptote
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
###########################################################################

import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import PyQt5.QtSvg as Qs

import numpy as np
from tkinter import *

import sys
import os
import signal
import threading
import subprocess
import tempfile
import queue
import io

import CubicBezier
import xasyUtils as xu
import xasyArgs as xa
import xasyOptions as xo

import BezierCurveEditor
import uuid

console = None


class DebugFlags:
    keepFiles = False
    printFoutTranscript = True
    printDeconstTranscript = True


class AsymptoteEngine:
    def __init__(self, path=None, args=None, customOutdir=None, keepFiles=DebugFlags.keepFiles, keepDefaultArgs=True,
                 stdoutMode=None, stdinMode=None, stderrMode=None, endargs=None):
        if path is None:
            path = xa.getArgs().asypath
            if path is None:
                opt = xo.xasyOptions()
                opt.load()
                path = opt['asyPath']

        rx, wx = os.pipe()
        ra, wa = os.pipe()

        os.set_inheritable(rx, True)
        os.set_inheritable(wx, True)
        os.set_inheritable(ra, True)
        os.set_inheritable(wa, True)

        self._stdoutMode = stdoutMode
        self._stdinMode = stdinMode
        self._stderrMode = stderrMode

        self._ostream = os.fdopen(wx, 'w')
        self._istream = os.fdopen(ra, 'r')
        self.keepFiles = keepFiles
        self.useTmpDir = customOutdir is None
        if customOutdir is None:
            self.tmpdir = tempfile.TemporaryDirectory(prefix='xasyData_')
            oargs = self.tmpdir.name + os.sep
        else:
            self.tmpdir = customOutdir
            oargs = customOutdir

        if args is None:
            args = []

        if endargs is None:
            endargs = []

        assert isinstance(args, list)

        self.args = args

        if keepDefaultArgs:
            self.args = args + ['-noV', '-inpipe=' + str(rx), '-outpipe=' + str(wa), '-o', oargs] + endargs

        self.asyPath = path
        self.asyProcess = None

        self.rx = rx
        self.wa = wa

    def wait(self):
        if self.asyProcess.returncode is not None:
            return
        else:
            return self.asyProcess.wait()

    def start(self):
        self.asyProcess = subprocess.Popen([self.asyPath] + self.args, close_fds=False,
                                           stdin=self._stdinMode, stderr=self._stderrMode)
        self.istream.readline()
        if self.asyProcess.returncode is not None:
            raise ChildProcessError('Asymptote failed to open')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.wait()

    def hangup(self):
        if self.active:
            self.asyProcess.send_signal(signal.SIGHUP)

    @property
    def tempDirName(self):
        return self.tmpdir.name + os.sep

    def startThenStop(self):
        self.start()
        self.stop()
        self.wait()

    @property
    def stdout(self):
        if self.asyProcess is None:
            return None
        return self.asyProcess.stdout

    @property
    def stdin(self):
        if self.asyProcess is None:
            return None
        return self.asyProcess.stdin

    @property
    def stderr(self):
        if self.asyProcess is None:
            return None
        return self.asyProcess.stderr

    @property
    def active(self):
        if self.asyProcess is None:
            return False
        return self.asyProcess.returncode is None

    @property
    def ostream(self):
        return self._ostream

    @property
    def istream(self):
        return self._istream

    def stop(self):
        if self.active:
            self.asyProcess.terminate()
            if not self.keepFiles and not self.useTmpDir:
                try:
                    os.rmdir(self.tempDirName)
                finally:
                    pass


def closeConsole(event):
    global console
    console = None


def consoleOutput(line):
    global console
    global ctl
    if not console:
        ctl = Toplevel()
        ctl.title("Asymptote Console")
        ctl.bind("<Destroy>", closeConsole)
        yscrollbar = Scrollbar(ctl)
        yscrollbar.pack(side=RIGHT, fill=Y)
        console = Text(ctl, yscrollcommand=yscrollbar.set)
        console.pack()
        yscrollbar.config(command=console.yview)
    console.insert(END, line)
    ctl.lift()


class asyTransform(Qc.QObject):
    """A python implementation of an asy transform"""

    def __init__(self, initTuple, delete=False):
        """Initialize the transform with a 6 entry tuple"""
        super().__init__()
        if isinstance(initTuple, (tuple, list)) and len(initTuple) == 6:
            self.t = initTuple
            self.x, self.y, self.xx, self.xy, self.yx, self.yy = initTuple
            self.deleted = delete
        else:
            raise TypeError("Illegal initializer for asyTransform")

    @classmethod
    def zero(cls):
        return asyTransform((0, 0, 0, 0, 0, 0))

    @classmethod
    def fromQTransform(cls, transform):
        assert isinstance(transform, Qg.QTransform)
        tx, ty = transform.dx(), transform.dy()
        xx, xy, yx, yy = transform.m11(), transform.m21(), transform.m12(), transform.m22()

        return asyTransform((tx, ty, xx, xy, yx, yy))

    @classmethod
    def fromNumpyMatrix(cls, transform):
        assert isinstance(transform, np.ndarray)
        assert transform.shape == (3, 3)

        tx = transform[0, 2]
        ty = transform[1, 2]

        xx, xy, yx, yy = transform[0:2, 0:2].ravel().tolist()[0]

        return asyTransform((tx, ty, xx, xy, yx, yy))

    def getCode(self):
        """Obtain the asy code that represents this transform"""
        if self.deleted:
            return str(self.t) + ", false"
        else:
            return str(self.t)

    def scale(self, s):
        return asyTransform((0, 0, s, 0, 0, s)) * self

    def toQTransform(self):
        return Qg.QTransform(self.xx, self.yx, self.xy, self.yy, self.x, self.y)

    def __str__(self):
        """Equivalent functionality to getCode(). It allows the expression str(asyTransform) to be meaningful."""
        return self.getCode()

    def isIdentity(self):
        return (self.x == 0) and (self.y == 0) and \
               (self.xx == 1) and (self.xy == 0) and \
               (self.yx == 0) and (self.yy == 1)

    def inverted(self):
        return asyTransform.fromQTransform(self.toQTransform().inverted()[0])

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
        else:
            raise Exception("Illegal multiplier of {:s}".format(str(type(other))))


def identity():
    return asyTransform((0, 0, 1, 0, 0, 1))


class asyObj(Qc.QObject):
    """A base class for asy objects: an item represented by asymptote code."""
    def __init__(self):
        """Initialize the object"""
        super().__init__()
        self.asyCode = ''

    def updateCode(self, mag=1.0):
        """Update the object's code: should be overriden."""
        raise NotImplementedError

    def getCode(self):
        """Return the code describing the object"""
        self.updateCode()
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

    def updateCode(self, mag=1.0):
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
        fout.write("file fout=output(mode='pipe');\n")
        fout.write("write(fout,colorspace(p),newl);\n")
        fout.write("write(fout,colors(p));\n")
        fout.write("flush(fout);\n")
        fout.flush()

        self.asyEngine.hangup()

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

    def __init__(self, asyengine=None):
        """Initialize the path to be an empty path: a path with no nodes, control points, or links."""
        super().__init__()
        self.nodeSet = []
        self.linkSet = []
        self.controlSet = []
        self.computed = False
        self.asyengine = asyengine

    @classmethod
    def fromBezierPoints(cls, pointList, engine=None):
        assert isinstance(pointList, list)
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

    @property
    def isEmpty(self):
        return len(self.nodeSet) == 0

    @property
    def isDrawable(self):
        return len(self.nodeSet) >= 2

    def toQPainterPath(self):
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
            return '({0}, {1})'.format(str(node[0]), str(node[1]))

    def updateCode(self, mag=1.0):
        """Generate the code describing the path"""
        with io.StringIO() as rawAsyCode:
            count = 0
            rawAsyCode.write(self.makeNodeStr(self.nodeSet[0]))
            for node in self.nodeSet[1:]:
                if not self.computed or count >= len(self.controlSet):
                    rawAsyCode.write(self.linkSet[count])
                    rawAsyCode.write(self.makeNodeStr(node))
                else:
                    rawAsyCode.write('..controls')
                    rawAsyCode.write(self.makeNodeStr(self.controlSet[count][0]))
                    rawAsyCode.write('and')
                    rawAsyCode.write(self.makeNodeStr(self.controlSet[count][1]))
                    rawAsyCode.write(".." + self.makeNodeStr(node) + "\n")
                count = count + 1
            self.asyCode = rawAsyCode.getvalue()

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
            self.nodeSet[index] = (self.nodeSet[index][0] + offset[0], self.nodeSet[1] + offset[1])

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

        fout.write("file fout=output(mode='pipe');\n")
        fout.write("path p=" + self.getCode() + ';\n')
        fout.write("write(fout,length(p),newl);\n")
        fout.write("write(fout,unstraighten(p),endl);\n")
        fout.flush()

        asy.hangup()

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

    def __init__(self, text="", location=(0, 0), pen=None, align=None):
        """Initialize the label with the given test, location, and pen"""
        asyObj.__init__(self)
        self.align = align
        self.pen = pen
        if align is None:
            self.align = 'SE'
        if pen is None:
            self.pen = asyPen()
        self.text = text
        self.location = location

    def updateCode(self, mag=1.0):
        """Generate the code describing the label"""
        self.asyCode = 'Label("{0}", {1}, p={2}, align={3})'.format(self.text, tuple(self.location), self.pen.getCode(),
                                                                    self.align)

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
    def __init__(self, image, format, bbox, transfKey=None):
        self.image = image
        self.format = format
        self.bbox = bbox
        self.IDTag = None
        self.key = transfKey


class xasyItem(Qc.QObject):
    """A base class for items in the xasy GUI"""
    setKeyFormatStr = 'map("{:s}",{:s});'

    def __init__(self, canvas=None, asyengine=None):
        """Initialize the item to an empty item"""
        super().__init__()
        self.transform = [identity()]
        self.transfKeymap = {}              # the new keymap.
        self.asyCode = ''
        self.imageList = []
        self.IDTag = None
        self.asyfied = False
        self.onCanvas = canvas
        self.keyBuffer = None
        self.asyengine = asyengine
        self.drawObjects = []
        self.imageHandleQueue = queue.Queue()

    def updateCode(self, mag=1.0):
        """Update the item's code: to be overriden"""
        raise NotImplementedError

    def getCode(self):
        """Return the code describing the item"""
        self.updateCode()
        return self.asyCode

    def generateDrawObjects(self, mag=1.0):
        raise NotImplementedError

    def handleImageReception(self, file, format, bbox, count, key=None):
        """Receive an image from an asy deconstruction. It replaces the default in asyProcess."""
        # image = Image.open(file).transpose(Image.FLIP_TOP_BOTTOM)
        image = Qg.QImage(file).mirrored(False, True)
        self.imageList.append(asyImage(image, format, bbox, transfKey=key))
        if self.onCanvas is not None:
            # self.imageList[-1].iqt = ImageTk.PhotoImage(image)
            currImage = self.imageList[-1]
            currImage.iqt = image
            currImage.originalImage = image.copy()
            currImage.originalImage.theta = 0.0
            currImage.originalImage.bbox = list(bbox)
            currImage.performCanvasTransform = False

            # handle this case if transform is not in the map yet.
            # if deleted - set transform to 0, 0, 0, 0, 0
            if key not in self.transfKeymap.keys() or not self.transfKeymap[key].deleted:
                currImage.IDTag = str(file)
                newDrawObj = DrawObject(currImage.iqt, self.onCanvas['canvas'], transform=identity(),
                                        btmRightanchor=Qc.QPointF(bbox[0], bbox[2]), drawOrder=-1, key=key,
                                        parentObj=self)
                newDrawObj.setParent(self)
                self.drawObjects.append(newDrawObj)

    def asyfy(self, mag=1.0, force=False):
        if self.asyengine is None:
            return 1
        if self.asyfied and not force:
            return
        self.drawObjects = []
        assert isinstance(self.asyengine, AsymptoteEngine)
        self.imageList = []
        self.imageHandleQueue = queue.Queue()
        worker = threading.Thread(target=self.asyfyThread, args=[mag])
        worker.start()
        item = self.imageHandleQueue.get()
        if console is not None:
            console.delete(1.0, END)
        while item != (None,) and item[0] != "ERROR":
            if item[0] == "OUTPUT":
                consoleOutput(item[1])
            else:
                self.handleImageReception(*item)
                if not DebugFlags.keepFiles:
                    try:
                        os.remove(item[0])
                    finally:
                        pass
            item = self.imageHandleQueue.get()
        # self.imageHandleQueue.task_done()
        worker.join()

    def asyfyThread(self, mag=1.0):
        """Convert the item to a list of images by deconstructing this item's code"""
        assert self.asyengine.active

        fout = self.asyengine.ostream
        fin = self.asyengine.istream

        for line in self.getCode().splitlines():
            if DebugFlags.printDeconstTranscript:
                print('fout:', line)
            fout.write(line+"\n")
        fout.write("deconstruct({:f});\n".format(mag))
        fout.flush()

        self.asyengine.hangup()

        maxargs = int(fin.readline().split()[0])        # should be 256, for now.
        imageInfos = []                                 # of (box, key)
        batch = 0
        n = 0

        def render():
            for i in range(len(imageInfos)):
                box, key = imageInfos[i]
                l, b, r, t = [float(a) for a in box.split()]
                name = "{:s}{:d}_{:d}.{:s}".format(self.asyengine.tempDirName, batch, i + 1, fileformat)

                self.imageHandleQueue.put((name, fileformat, (l, b, r, t), i, key))

        # key first, box second.
        # if key is "Done"
        raw_text = fin.readline()
        text = ""
        if DebugFlags.printDeconstTranscript:
            print(raw_text.strip())

        # template=AsyTempDir+"%d_%d.%s"
        fileformat = 'png'

#        print(raw_text)
        while raw_text != "Done\n" and raw_text != "Error\n":
            text = fin.readline()       # the actual bounding box.
            # print('TESTING:', text)
            keydata = raw_text.strip().replace('KEY=', '', 1)  # key
#                print(line, col)
            imageInfos.append((text, keydata))      # key-data pair

            raw_text = fin.readline()

            if DebugFlags.printDeconstTranscript:
                print(text.rstrip())
                print(raw_text.rstrip())

            n += 1
            if n >= maxargs:
                render()
                imageInfos = []
                batch += 1
                n = 0

        if text == "Error\n":
            self.imageHandleQueue.put(("ERROR", fin.readline()))
        else:
            render()
        self.imageHandleQueue.put((None,))
        self.asyfied = True


class xasyDrawnItem(xasyItem):
    """A base class for GUI items was drawn by the user. It combines a path, a pen, and a transform."""

    def __init__(self, path, engine, pen=None, transform=identity(), key=None):
        """Initialize the item with a path, pen, and transform"""
        super().__init__()
        if pen is None:
            pen = asyPen()
        self.path = path
        self.path.asyengine = engine
        self.pen = pen
        self.rawIdentifier = 'x' + str(uuid.uuid4())
        self.transfKey = key
        if key is None:
            self.transfKey = self.rawIdentifier
        self.transfKeymap = {self.transfKey: transform}

    def generateDrawObjects(self, mag=1.0, forceUpdate=False):
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

    def updateCode(self, mag=1.0):
        raise NotImplementedError


class xasyShape(xasyDrawnItem):
    """An outlined shape drawn on the GUI"""
    def __init__(self, path, asyengine, pen=None, transform=identity()):
        """Initialize the shape with a path, pen, and transform"""
        super().__init__(path=path, engine=asyengine, pen=pen, transform=transform)

    def updateCode(self, mag=1.0):
        """Generate the code to describe this shape"""
        with io.StringIO() as rawAsyCode:
            rawAsyCode.write(xasyItem.setKeyFormatStr.format(self.transfKey, self.transfKeymap[self.transfKey].getCode()
                                                             ))
            rawAsyCode.write(
                '\ndraw(KEY="{0}", {1}, {2})'.format(self.transfKey, self.path.getCode(), self.pen.getCode()))
            self.asyCode = rawAsyCode.getvalue()

    def generateDrawObjects(self, mag=1.0, forceUpdate=False):
        self.path.computeControls()
        newObj = DrawObject(self.path.toQPainterPath(), None, drawOrder=0, transform=self.transfKeymap[self.transfKey],
                            pen=self.pen, key=self.transfKey)
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

    def updateCode(self, mag=1.0):
        """Generate the code describing this shape"""
        self.asyCode = "xformStack.push(" + self.transform[0].getCode() + ");\n"
        self.asyCode += "fill(" + self.path.getCode() + "," + self.pen.getCode() + ");"

    def generateDrawObjects(self, mag=1.0, forceUpdate=False):
        self.path.computeControls()
        newObj = DrawObject(self.path.toQPainterPath(), None, drawOrder=0, transform=self.transfKeymap[self.transfKey],
                            pen=self.pen, key=self.transfKey, fill=True)
        newObj.originalObj = self
        newObj.setParent(self)
        return [newObj]

    def __str__(self):
        """Return a string describing this shape"""
        return "xasyFilledShape code:{:s}".format("\n\t".join(self.getCode().splitlines()))


class xasyText(xasyItem):
    """Text created by the GUI"""

    def __init__(self, text, location, asyengine, pen=None, transform=identity(), key=None, align=None):
        """Initialize this item with text, a location, pen, and transform"""
        super().__init__(asyengine=asyengine)
        if pen is None:
            pen = asyPen(asyengine=asyengine)
        if pen.asyEngine is None:
            pen.asyEngine = asyengine
        self.label = asyLabel(text, location, pen, align)
        # self.transform = [transform]
        if key is None:
            self.key = 'x:' + str(uuid.uuid4())
        else:
            self.key = key
        self.transfKeymap = {self.key: transform}
        self.onCanvas = None

    def updateCode(self, mag=1.0):
        """Generate the code describing this object"""
        with io.StringIO() as rawAsyCode:
            rawAsyCode.write(xasyItem.setKeyFormatStr.format(self.key, self.transfKeymap[self.key].getCode()))
            rawAsyCode.write('\nlabel(KEY="{0}", {1});\n'.format(self.key, self.label.getCode()))
            self.asyCode = rawAsyCode.getvalue()

    def generateDrawObjects(self, mag=1.0, forceUpdate=False):
        self.asyfy(mag, forceUpdate)
        return self.drawObjects

    def getBoundingBox(self, mag=1.0):
        self.asyfy(mag)
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
        self.setKeyed = False

    def clearTransform(self):
        """Reset the transforms for each of the deconstructed images"""
        # self.transform = [identity()] * len(self.imageList)
        for im in self.imageList:
            self.transfKeymap[im.key] = identity()

    def updateCode(self, mag=1.0):
        """Generate the code describing this script"""
        with io.StringIO() as rawAsyCode:
            if self.transfKeymap:
                for key, val in self.transfKeymap.items():
                    if val.deleted:
                        rawAsyCode.write(xasyItem.setKeyFormatStr.format(key, str(asyTransform.zero())) + '\n')
                        rawAsyCode.write('// ')
                    rawAsyCode.write(xasyItem.setKeyFormatStr.format(key, str(val)) + '\n')

            for line in self.script.splitlines():
                raw_line = line.rstrip().replace('\t', ' ' * 4)
                rawAsyCode.write(raw_line + '\n')

            self.asyCode = rawAsyCode.getvalue()

    def setScript(self, script):
        """Sets the content of the script item."""
        self.script = script
        self.updateCode()
        self.setKeyed = False

    def setKey(self, prefix=''):
        fout = self.asyengine.ostream
        fin = self.asyengine.istream

        for line in self.script.splitlines():
            fout.write(line + '\n')
        fout.write('deconstruct();\n')
        fout.flush()

        self.asyengine.hangup()

        keylist = {}
        linebuf = fin.readline()
        while linebuf != 'Done\n':
            if linebuf.startswith('KEY='):
                key = linebuf.rstrip().replace('KEY=', '', 1)
                raw_parsed = xu.tryParseKey(key)
                if raw_parsed is not None:
                    line, col = [int(val) for val in raw_parsed.groups()]
                    if line not in keylist:
                        keylist[line] = set()

                    keylist[line].add(col)
            linebuf = fin.readline()

        raw_code_lines = self.script.splitlines()

        with io.StringIO() as raw_str:
            for i in range(len(raw_code_lines)):
                curr_str = raw_code_lines[i]
                if i + 1 in keylist.keys():
                    # this case, we have a key.
                    with io.StringIO() as raw_line:
                        for j in range(len(curr_str)):
                            raw_line.write(curr_str[j])
                            if j + 1 in keylist[i + 1]:
                                raw_line.write('KEY="x{2:s}{0:d}.{1:d}", '.format(i + 1, j + 1, prefix))
                        curr_str = raw_line.getvalue()
                # else, skip and just write the line.
                raw_str.write(curr_str + '\n')
                self.script = raw_str.getvalue()
        self.updateCode()
        self.setKeyed = True

    def asyfy(self, mag=1.0, keyOnly=False):
        """Generate the list of images described by this object and adjust the length of the transform list."""
        super().asyfy(mag, keyOnly)

        # remove any unnessecary keys
        # not anymore - transfKeymap is supposed to be storing the raw transform data
        # and the rest, to be interpreted case by case.

        # Id --> Transf --> asy-fied --> Transf
        # Transf should keep the original, raw transformation
        # but for all new drawn objects - assign Id as transform.

        # key_set = set([im.key for im in self.imageList])
        # keys_to_remove = []
        #
        # for key in self.transfKeymap.keys():
        #     if key not in key_set:
        #         keys_to_remove.append(key)
        #
        # for key in keys_to_remove:
        #     self.transfKeymap.pop(key)

        # add in any missng key:
        for im in self.imageList:
            if im.key not in self.transfKeymap.keys():
                self.transfKeymap[im.key] = identity()

        self.updateCode()

    def generateDrawObjects(self, mag=1.0, forceUpdate=False):
        self.asyfy(mag, forceUpdate)
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
                 drawOrder=(-1, -1), pen=None, key=None, parentObj=None, fill=False):
        super().__init__()
        self.drawObject = drawObject
        self.mainCanvas = mainCanvas
        self.pTransform = transform
        self.baseTransform = transform
        self.drawOrder = drawOrder
        self.btmRightAnchor = btmRightanchor
        self.originalObj = parentObj
        self.useCanvasTransformation = False
        self.key = key
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

    @property
    def boundingBox(self):
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

    def draw(self, additionalTransformation=None, applyReverse=False, canvas=None):
        if canvas is None:
            canvas = self.mainCanvas
        if additionalTransformation is None:
            additionalTransformation = Qg.QTransform()

        assert isinstance(canvas, Qg.QPainter)
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
            canvas.drawImage(self.btmRightAnchor, self.drawObject)
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
        return self.boundingBox.contains(coords)

    def getID(self):
        return self.originalObj
