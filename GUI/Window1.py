import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import os
import xasy2asy as x2a
import xasyFile as xf
from pyUIClass.window1 import Ui_MainWindow


class MainWindow1(Qw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.filename = None
        #self.ui.imgLabel.updateGeometry()
        # canvasSize = Qc.QSize(450, 450)
        canvasSize = self.rect().size()
        self.canvasPixmap = Qg.QPixmap(canvasSize)
        self.canvasPixmap.fill()
        self.mainCanvas = Qg.QPainter(self.canvasPixmap)
        centerPoint = (self.canvasPixmap.rect().bottomLeft() + self.canvasPixmap.rect().topRight()) / 2
        self.mainCanvas.translate(centerPoint)
        self.magnification = 1

    def loadFile(self, name):
        self.ui.statusbar.showMessage(name)
        self.filename = os.path.abspath(name)
        x2a.startQuickAsy()
        # self.retitle()
        try:
            try:
                f = open(self.filename, 'rt')
            except:
                if self.filename[-4:] == ".asy":
                    raise
                else:
                    f = open(self.filename + ".asy", 'rt')
                    self.filename += ".asy"
                    self.retitle()
            self.fileItems = xf.parseFile(f)
            f.close()
        except IOError:
            Qw.QMessageBox.critical(self, "File Opening Failed.", "File could not be opened.")
            # messagebox.showerror("File Opening Failed.", "File could not be opened.")
            self.fileItems = []
        except Exception:
            self.fileItems = []
            self.autoMakeScript = True
            if self.autoMakeScript or Qw.QMessageBox.question(self, "Error Opening File",
                                                              "File was not recognized as an xasy file.\nLoad as a script item?") == \
                    Qw.QMessageBox.Yes:
                # try:
                item = x2a.xasyScript(self.mainCanvas)
                f.seek(0)
                item.setScript(f.read())
                self.fileItems.append(item)
                # except:
                #     Qw.QMessageBox.critical(self, "File Opening Failed.", "File could not be opened.")
                #     # messagebox.showerror("File Opening Failed.", "Could not load as a script item.")
                #     self.fileItems = []
        self.populateCanvasWithItems()
        #self.populatePropertyList()
        #self.updateCanvasSize()
        self.ui.imgLabel.setPixmap(self.canvasPixmap)
        # self.update()


    def populateCanvasWithItems(self):
        # if (not self.testOrAcquireLock()):
        #     return
        self.itemCount = 0
        for item in self.fileItems:
            item.drawOnCanvas(self.mainCanvas, self.magnification, forceAddition=True)
            # self.bindItemEvents(item)
        # self.releaseLock()
