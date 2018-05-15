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

import numpy as np
from tkinter import *

import sys
import os
import signal
import threading
from subprocess import *
import tempfile
import queue
import io

import xasyOptions as xo
import CubicBezier
import xasyUtils

import BezierCurveEditor
import uuid

quickAsyFailed = True
console = None
options = xo.xasyOptions()
options.load()


class DebugFlags:
    keepFiles = True
    printDeconstTranscript = True

# TODO: Eventually move this to a class of its own.


def startQuickAsy():
    global quickAsy
    global quickAsyFailed
    global asyTempRawDir
    global AsyTempDir
    global fout, fin

    if quickAsyRunning():
        return
    try:
        fout.close()
        quickAsy.wait()
    except:
        pass
    try:
        quickAsyFailed = False
        asyTempRawDir = tempfile.TemporaryDirectory(prefix='xasyData_')

        # TODO: For windows users, make sure tmp directory has correct folder seperator.
        AsyTempDir = asyTempRawDir.name + '/'
        if os.name == 'nt':
            quickAsy = Popen([options['asyPath'], "-noV", "-multiline", "-q",
                              "-o" + AsyTempDir, "-inpipe=0", "-outpipe=2"], stdin=PIPE,
                             stderr=PIPE, universal_newlines=True)
            fout = quickAsy.stdin
            fin = quickAsy.stderr
        else:
            (rx, wx) = os.pipe()
            (ra, wa) = os.pipe()
            if sys.version_info >= (3, 4):
                os.set_inheritable(rx, True)
                os.set_inheritable(wx, True)
                os.set_inheritable(ra, True)
                os.set_inheritable(wa, True)

            if options['debugMode']:
                # NOTE: Really ugly here, but eventaully move this to handling arguments
                asyPathBase = './asydev'
            else:
                asyPathBase = options['asyPath']
            quickAsy = Popen([asyPathBase, "-noV", "-multiline", "-q",
                              "-o" + AsyTempDir, "-inpipe=" + str(rx), "-outpipe=" + str(wa)],
                             close_fds=False)
            fout = os.fdopen(wx, 'w')
            fin = os.fdopen(ra, 'r')

        if quickAsy.returncode is not None:
            quickAsyFailed = True
    except:
        quickAsyFailed = True


def stopQuickAsy():
    if quickAsyRunning():
        fout.write("exit;\n")


def getAsyTempDir():
    return AsyTempDir


def quickAsyRunning():
    return (not quickAsyFailed) and quickAsy.returncode is None


def asyExecute(command):
    if not quickAsyRunning():
        startQuickAsy()
    fout.write(command)


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


class asyTransform:
    """A python implementation of an asy transform"""

    def __init__(self, initTuple, delete=False):
        """Initialize the transform with a 6 entry tuple"""
        if isinstance(initTuple, tuple) and len(initTuple) == 6:
            self.t = initTuple
            self.x, self.y, self.xx, self.xy, self.yx, self.yy = initTuple
            self.deleted = delete
        else:
            raise TypeError("Illegal initializer for asyTransform")

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


class asyObj:
    """A base class for asy objects: an item represented by asymptote code."""
    def __init__(self):
        """Initialize the object"""
        self.asyCode = ""

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

    def __init__(self, color=(0, 0, 0), width=0.5, pen_options=""):
        """Initialize the pen"""
        asyObj.__init__(self)
        self.color = (0, 0, 0)
        self.options = pen_options
        self.width = width
        self.setColor(color)
        self.updateCode()
        if pen_options:
            self.computeColor()

    def updateCode(self, mag=1.0):
        """Generate the pen's code"""
        self.asyCode = "rgb({:g},{:g},{:g})+{:s}".format(self.color[0], self.color[1], self.color[2], str(self.width))
        if len(self.options) > 0:
            self.asyCode += "+" + self.options

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
        fout.write("pen p=" + self.getCode() + ';\n')
        fout.write("file fout=output(mode='pipe');\n")
        fout.write("write(fout,colorspace(p),newl);\n")
        fout.write("write(fout,colors(p));\n")
        fout.write("flush(fout);\n")
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

    def tkColor(self):
        """Return the tk version of the pen's color"""
        self.computeColor()
        return '#{}'.format("".join(["{:02x}".format(min(int(256 * a), 255)) for a in self.color]))

    def toQPen(self):
        newPen = Qg.QPen()
        newPen.setColor(asyPen.convertToQColor(self.color))
        newPen.setWidthF(self.width)

        return newPen


class asyPath(asyObj):
    """A python wrapper for an asymptote path"""

    def __init__(self):
        """Initialize the path to be an empty path: a path with no nodes, control points, or links."""
        super().__init__()
        self.nodeSet = []
        self.linkSet = []
        self.controlSet = []
        self.computed = False

    @classmethod
    def fromBezierPoints(cls, pointList):
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
        newPath = asyPath()
        newPath.initFromControls(nodeList, controlList)
        return newPath

    def toQPainterPath(self):
        if not self.computed:
            self.computeControls()

        baseX, baseY = self.nodeSet[0]
        painterPath = Qg.QPainterPath(Qc.QPointF(baseX, baseY))

        for pointIndex in range(1, len(self.nodeSet)):
            endPoint = Qc.QPointF(self.nodeSet[pointIndex][0], self.nodeSet[pointIndex][1])
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
            return "(" + str(node[0]) + "," + str(node[1]) + ")"

    def updateCode(self, mag=1.0):
        """Generate the code describing the path"""

        # TODO: Change this to io.StringIO for better performance.
        if not self.computed:
            count = 0
            # this string concatenation could be optimised
            self.asyCode = self.makeNodeStr(self.nodeSet[0])
            for node in self.nodeSet[1:]:
                self.asyCode += self.linkSet[count] + self.makeNodeStr(node)
                count += 1
        else:
            count = 0
            # this string concatenation could be optimised
            self.asyCode = self.makeNodeStr(self.nodeSet[0])
            for node in self.nodeSet[1:]:
                self.asyCode += "..controls"
                self.asyCode += self.makeNodeStr(self.controlSet[count][0])
                self.asyCode += "and"
                self.asyCode += self.makeNodeStr(self.controlSet[count][1])
                self.asyCode += ".." + self.makeNodeStr(node) + "\n"
                count += 1

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

    def moveControl(self, index, offset):
        """Translate a control point"""
        self.controlSet[index] = (self.controlSet[index][0] + offset[0], self.controlSet[index][1] + offset[1])

    def computeControls(self):
        """Evaluate the code of the path to obtain its control points"""
        fout.write("file fout=output(mode='pipe');\n")
        fout.write("path p=" + self.getCode() + ';\n')
        fout.write("write(fout,length(p),newl);\n")
        fout.write("write(fout,unstraighten(p),endl);\n")
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


class asyLabel(asyObj):
    """A python wrapper for an asy label"""

    def __init__(self, text="", location=(0, 0), pen=asyPen()):
        """Initialize the label with the given test, location, and pen"""
        asyObj.__init__(self)
        self.text = text
        self.location = location
        self.pen = pen

    def updateCode(self, mag=1.0):
        """Generate the code describing the label"""
        self.asyCode = "Label(\"" + self.text + "\"," + str(
            (self.location[0], self.location[1])) + "," + self.pen.getCode() + ",align=SE)"

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


class xasyItem:
    """A base class for items in the xasy GUI"""
    setKeyFormatStr = 'xformMap.add("{:s}", {:s});'

    def __init__(self, canvas=None):
        """Initialize the item to an empty item"""
        self.transform = [identity()]
        self.transfKeymap = {}              # the new keymap.
        self.asyCode = ''
        self.imageList = []
        self.IDTag = None
        self.asyfied = False
        self.onCanvas = canvas
        self.keyBuffer = None
        self.imageHandleQueue = queue.Queue()

    def updateCode(self, mag=1.0):
        """Update the item's code: to be overriden"""
        raise NotImplementedError

    def getCode(self):
        """Return the code describing the item"""
        self.updateCode()
        return self.asyCode

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
            if key not in self.transfKeymap.keys() or not self.transfKeymap[key].deleted:
                # TODO: Look at Transform & translations
                #  we still want a unique ID tag for each file. The key is for transformation.
                currImage.IDTag = str(file)
                if key in self.transfKeymap.keys():
                    inputTransform = self.transfKeymap[key]
                else:
                    inputTransform = identity()

                # TODO: The problem now is to get asy to recognize the mapped transformation.
                # It is in there - saved, but asy doesn't recognize it.

                self.onCanvas['drawDict'][currImage.IDTag] = \
                    DrawObject(currImage.iqt, self.onCanvas['canvas'], transform=inputTransform,
                               btmRightanchor=Qc.QPointF(bbox[0], bbox[2]),
                               drawOrder=count, key=key)
                self.onCanvas['drawDict'][currImage.IDTag].originalObj = self, count
                self.onCanvas['drawDict'][currImage.IDTag].draw()

    def asyfy(self, mag=1.0, keyOnly=False):
        self.imageList = []
        self.imageHandleQueue = queue.Queue()
        worker = threading.Thread(target=self.asyfyThread, args=[mag, keyOnly])
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

    def asyfyThread(self, mag=1.0, keyOnly=False):
        """Convert the item to a list of images by deconstructing this item's code"""
        fout.write("reset;\n")
        # TODO: Figure out what's wrong. with initXasyMode();
        # fout.write("initXasyMode();\n")
        fout.write("atexit(null);\n")
        for line in self.getCode().splitlines():
            fout.write(line + "\n")
        fout.write("deconstruct({:f});\n".format(mag))
        fout.flush()

        maxargs = int(fin.readline().split()[0])        # should be 256, for now.
        imageInfos = []                                 # of (box, key)
        batch = 0
        n = 0

        def render():
            for i in range(len(imageInfos)):
                box, key = imageInfos[i]
                l, b, r, t = [float(a) for a in box.split()]
                name = "{:s}{:d}_{:d}.{:s}".format(AsyTempDir, batch, i + 1, fileformat)

                self.imageHandleQueue.put((name, fileformat, (l, b, r, t), i, key))

        # key first, box second.
        # if key is "Done"
        raw_text = fin.readline()
        text = ""
        if DebugFlags.printDeconstTranscript:
            print(raw_text.strip())

        # template=AsyTempDir+"%d_%d.%s"
        fileformat = "png"

        while raw_text != "Done\n" and raw_text != "Error\n":
            text = fin.readline()       # the actual bounding box.
            # print('TESTING:', text)
            keydata = raw_text.strip().replace('KEY=', '', 1)  # key
            if not keydata.startswith('x:'):
                line, col = keydata.split('.')
                if self.keyBuffer is None:
                    self.keyBuffer = {}
                if line not in self.keyBuffer:
                    self.keyBuffer[line] = set()
                self.keyBuffer[line].add(col)
                print(line, col)
            imageInfos.append((text, keydata))      # key-data pair

            raw_text = fin.readline()

            if DebugFlags.printDeconstTranscript:
                print(text.strip())
                print(raw_text.strip())

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

    def drawOnCanvas(self, canvas, mag, forceAddition=False):
        raise NotImplementedError


class xasyDrawnItem(xasyItem):
    """A base class for GUI items was drawn by the user. It combines a path, a pen, and a transform."""

    def __init__(self, path, pen=asyPen(), transform=identity(), key=None):
        """Initialize the item with a path, pen, and transform"""
        super().__init__()
        self.path = path
        self.pen = pen
        self.rawIdentifier = 'x:' + str(np.random.randint(0, 10000))
        self.transfKey = key
        if key is None:
            self.transfKey = 'x:' + str(uuid.uuid4())
        self.transfKeymap = {self.transfKey: transform}
        self.transform = [transform]

    def appendPoint(self, point, link=None):
        """Append a point to the path. If the path is cyclic, add this point before the 'cycle' node."""
        if self.path.nodeSet[-1] == 'cycle':
            self.path.nodeSet[-1] = point
            self.path.nodeSet.append('cycle')
        else:
            self.path.nodeSet.append(point)
        self.path.computed = False
        if len(self.path.nodeSet) > 1 and link is not None:
            self.path.linkSet.append(link)

    def clearTransform(self):
        """Reset the item's transform"""
        self.transform = [identity()]

    def removeLastPoint(self):
        """Remove the last point in the path. If the path is cyclic, remove the node before the 'cycle' node."""
        if self.path.nodeSet[-1] == 'cycle':
            del self.path.nodeSet[-2]
        else:
            del self.path.nodeSet[-1]
        del self.path.linkSet[-1]
        self.path.computed = False

    def setLastPoint(self, point):
        """Modify the last point in the path. If the path is cyclic, modify the node before the 'cycle' node."""
        if self.path.nodeSet[-1] == 'cycle':
            self.path.nodeSet[-2] = point
        else:
            self.path.nodeSet[-1] = point
        self.path.computed = False

    def updateCode(self, mag=1.0):
        raise NotImplementedError

    def drawOnCanvas(self, canvas, mag, forceAddition=False):
        raise NotImplementedError


class xasyShape(xasyDrawnItem):
    """An outlined shape drawn on the GUI"""

    def __init__(self, path, pen=asyPen(), transform=identity()):
        """Initialize the shape with a path, pen, and transform"""
        super().__init__(path, pen, transform)

    def updateCode(self, mag=1.0):
        """Generate the code to describe this shape"""
        with io.StringIO() as rawAsyCode:
            rawAsyCode.write(xasyItem.setKeyFormatStr.format(self.transfKey, self.transfKeymap[self.transfKey].getCode()))
            rawAsyCode.write('\ndraw(KEY={0}, {1}, {2})'.format(self.transfKey, self.path.getCode(), self.pen.getCode()))
            self.asyCode = rawAsyCode.getvalue()

    def drawOnCanvas(self, canvas, mag, asyFy=False, forceAddition=False):
        """Add this shape to a Qt (not TK anymore) canvas"""
        # for now. Drawing custom no longer needed (?) Use QPainterPath instead?
        if not asyFy:
            if self.IDTag is None or forceAddition:
                # add ourselves to the canvas
                self.path.computeControls()
                # self.IDTag = canvas.create_line(0, 0, 0, 0, tags=("drawn", "xasyShape"), fill=self.pen.tkColor(),
                #                                width=self.pen.width * mag)
                self.IDTag = Qc.QLine()
                self.drawOnCanvas(canvas, mag)
            else:
                self.path.computeControls()
                while self.rawIdentifier in canvas['drawDict'].keys():
                    self.rawIdentifier = np.random.randint(0, 1000000)

                drawPriority = len(canvas['drawDict']) + 1
                canvas['drawDict'][self.rawIdentifier] = DrawObject(self.path.toQPainterPath(), canvas['canvas'],
                                                       drawOrder=drawPriority, transform=self.transform[0], pen=self.pen, key=self.transfKey)
                canvas['drawDict'][self.rawIdentifier].originalObj = self, 0
                canvas['drawDict'][self.rawIdentifier].draw()
                self.IDTag = canvas['drawDict'][self.rawIdentifier].getID()

    def __str__(self):
        """Create a string describing this shape"""
        return "xasyShape code:{:s}".format("\n\t".join(self.getCode().splitlines()))


class xasyFilledShape(xasyShape):
    """A filled shape drawn on the GUI"""

    def __init__(self, path, pen=asyPen(), transform=identity()):
        """Initialize this shape with a path, pen, and transform"""
        if path.nodeSet[-1] != 'cycle':
            raise Exception("Filled paths must be cyclic")
        super().__init__(path, pen, transform)

    def updateCode(self, mag=1.0):
        """Generate the code describing this shape"""
        self.asyCode = "xformStack.push(" + self.transform[0].getCode() + ");\n"
        self.asyCode += "fill(" + self.path.getCode() + "," + self.pen.getCode() + ");"

    def drawOnCanvas(self, canvas, mag, asyFy=False, forceAddition=False):
        """Add this shape to a tk canvas"""
        if not asyFy:
            if self.IDTag is None or forceAddition:
                # add ourselves to the canvas
                self.path.computeControls()
                self.IDTag = canvas.create_polygon(0, 0, 0, 0, 0, 0, tags=("drawn", "xasyFilledShape"),
                                                   fill=self.pen.tkColor(), outline=self.pen.tkColor(), width=1 * mag)
                self.drawOnCanvas(canvas, mag)
            else:
                self.path.computeControls()
                pointSet = []
                previousNode = self.path.nodeSet[0]
                nodeCount = 0
                if len(self.path.nodeSet) == 0:
                    pointSet = [0, 0, 0, 0, 0, 0]
                elif len(self.path.nodeSet) == 1:
                    if self.path.nodeSet[-1] != 'cycle':
                        p = self.transform[0] * (self.path.nodeSet[0][0], self.path.nodeSet[0][1])
                        pointSet = [p[0], -p[1], p[0], -p[1], p[0], -p[1]]
                    else:
                        pointSet = [0, 0, 0, 0, 0, 0]
                elif len(self.path.nodeSet) == 2:
                    if self.path.nodeSet[-1] != 'cycle':
                        p = self.transform[0].scale(mag) * (self.path.nodeSet[0][0], self.path.nodeSet[0][1])
                        p2 = self.transform[0].scale(mag) * (self.path.nodeSet[1][0], self.path.nodeSet[1][1])
                        pointSet = [p[0], -p[1], p2[0], -p2[1], p[0], -p[1]]
                    else:
                        pointSet = [0, 0, 0, 0, 0, 0]
                else:
                    for node in self.path.nodeSet[1:]:
                        if node == 'cycle':
                            node = self.path.nodeSet[0]
                        transform = self.transform[0].scale(mag)
                        points = CubicBezier.makeBezier(transform * previousNode,
                                                        transform * self.path.controlSet[nodeCount][0],
                                                        transform * self.path.controlSet[nodeCount][1],
                                                        transform * node)
                        for point in points:
                            pointSet += [point[0], -point[1]]
                        nodeCount += 1
                        previousNode = node
                canvas.coords(self.IDTag, *pointSet)
                canvas.itemconfigure(self.IDTag, fill=self.pen.tkColor(), outline=self.pen.tkColor(), width=1 * mag)
        else:
            # first asyfy then add an image list
            pass

    def __str__(self):
        """Return a string describing this shape"""
        return "xasyFilledShape code:{:s}".format("\n\t".join(self.getCode().splitlines()))


class xasyText(xasyItem):
    """Text created by the GUI"""

    def __init__(self, text, location, pen=asyPen(), transform=identity(), key=None):
        """Initialize this item with text, a location, pen, and transform"""
        super().__init__()
        self.label = asyLabel(text, location, pen)
        # self.transform = [transform]
        if key is None:
            # TODO: Hopefully asy engine can store a list of used keys...
            self.key = str(uuid.uuid4())
        else:
            self.key = key
        self.transfKeymap = {key: transform}
        self.onCanvas = None

    def updateCode(self, mag=1.0):
        """Generate the code describing this object"""
        with io.StringIO() as rawAsyCode:
            rawAsyCode.write(xasyItem.setKeyFormatStr.format(self.key, self.transfKeymap[self.key]))
            rawAsyCode.write('\nlabel({0});\n'.format(self.label.getCode()))
            self.asyCode = rawAsyCode.getvalue()
        # self.asyCode = "xformStack.push(" + self.transform[0].getCode() + ");\n"

    def drawOnCanvas(self, canvas, mag, asyFy=True, forceAddition=False):
        """Adds the label's images to a tk canvas"""
        if not self.onCanvas:
            self.onCanvas = canvas
        elif self.onCanvas != canvas:
            raise Exception("Error: item cannot be added to more than one canvas")
        self.asyfy(mag)

    def __str__(self):
        return "xasyText code:{:s}".format("\n\t".join(self.getCode().splitlines()))


class xasyScript(xasyItem):
    """A set of images create from asymptote code. It is always deconstructed."""

    def __init__(self, canvas, script="", transforms=None, transfKeyMap=None):
        """Initialize this script item"""
        super().__init__(canvas)
        # if transforms is not None:
        #     self.transform = transforms[:]
        # else:
        #     self.transform = []

        if transfKeyMap is not None:
            self.transfKeymap = transfKeyMap
        else:
            self.transfKeymap = {}

        self.script = script

    def clearTransform(self):
        """Reset the transforms for each of the deconstructed images"""
        # self.transform = [identity()] * len(self.imageList)
        for im in self.imageList:
            self.transfKeymap[im.key] = identity()

    def updateCode(self, mag=1.0):
        """Generate the code describing this script"""
        # TODO: Find a way to directly write-in the key.
        with io.StringIO() as rawAsyCode:
            if self.transfKeymap:
                transfMapList = [xasyItem.setKeyFormatStr.format(key, str(value))
                                 for key, value in self.transfKeymap.items()]
                transfMapList.append('')
                rawAsyCode.write('\n'.join(transfMapList))

            # start/endScript messes up with keys.
            rawAsyCode.write("startScript(); {\n")

            for line in self.script.splitlines():
                raw_line = line.rstrip().replace('\t', ' ' * 4)
                rawAsyCode.write(raw_line + '\n')

            rawAsyCode.write("\n} endScript();\n")

            self.asyCode = rawAsyCode.getvalue()
            pass

    def setScript(self, script):
        """Sets the content of the script item."""
        self.script = script
        self.updateCode()

    def asyfy(self, mag=1.0, keyOnly=False):
        """Generate the list of images described by this object and adjust the length of the transform list."""
        super().asyfy(mag, keyOnly)
        while len(self.imageList) > len(self.transform):
            self.transform.append(identity())
        while len(self.imageList) < len(self.transform):
            self.transform.pop()

        # remove any unnessecary keys
        key_set = set([im.key for im in self.imageList])
        keys_to_remove = []

        for key in self.transfKeymap:
            if key not in key_set:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.transfKeymap.pop(key)

        # add in any missng key:

        for im in self.imageList:
            if im.key not in self.transfKeymap:
                self.transfKeymap[im.key] = identity()

        self.updateCode()

    def drawOnCanvas(self, canvas, mag, asyFy=True, forceAddition=False):
        """Adds the script's images to a tk canvas"""
        if not self.onCanvas:
            self.onCanvas = canvas
        elif self.onCanvas != canvas:
            raise Exception("Error: item cannot be added to more than one canvas")
        self.asyfy(mag)

    def __str__(self):
        """Return a string describing this script"""
        retVal = "xasyScript\n\tTransforms:\n"
        for xform in self.transform:
            retVal += "\t" + str(xform) + "\n"
        retVal += "\tCode Ommitted"
        return retVal


class DrawObject:
    def __init__(self, drawObject, mainCanvas=None, transform=identity(), btmRightanchor=Qc.QPointF(0, 0),
                 drawOrder=-1, pen=None, key=None):
        self.drawObject = drawObject
        self.mainCanvas = mainCanvas
        self.pTransform = transform
        self.baseTransform = transform
        self.drawOrder = drawOrder
        self.btmRightAnchor = btmRightanchor
        self.originalObj = None
        self.useCanvasTransformation = False
        self.key = key
        self.pen = pen

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

    def draw(self, additionalTransformation=Qg.QTransform(), applyReverse=False):
        assert isinstance(self.mainCanvas, Qg.QPainter)
        assert self.mainCanvas.isActive()

        self.mainCanvas.save()
        if self.pen:
            oldPen = Qg.QPen(self.mainCanvas.pen())
            self.mainCanvas.setPen(self.pen.toQPen())
        else:
            oldPen = Qg.QPen()
        if not applyReverse:
            self.mainCanvas.setTransform(additionalTransformation, True)
            self.mainCanvas.setTransform(self.transform.toQTransform(), True)
        else:
            self.mainCanvas.setTransform(self.transform.toQTransform(), True)
            self.mainCanvas.setTransform(additionalTransformation, True)

        self.mainCanvas.setTransform(self.baseTransform.toQTransform().inverted()[0], True)

        if isinstance(self.drawObject, Qg.QImage):
            self.mainCanvas.drawImage(self.btmRightAnchor, self.drawObject)
        elif isinstance(self.drawObject, Qg.QPainterPath):
            self.mainCanvas.drawPath(self.baseTransform.toQTransform().map(self.drawObject))

        if self.pen:
            self.mainCanvas.setPen(oldPen)
        self.mainCanvas.restore()

    def collide(self, coords, canvasCoordinates=True):
        return self.boundingBox.contains(coords)

    def getID(self):
        return self.originalObj


if __name__ == '__main__':
    root = Tk()
    t = xasyText("test", (0, 0))
    t.asyfy()
