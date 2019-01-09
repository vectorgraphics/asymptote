#!/usr/bin/env python3

from pyUIClass.labelTextEditor import Ui_Dialog
import PyQt5.QtWidgets as Qw
import PyQt5.QtSvg as Qs
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
import xasyArgs as xa
import xasy2asy as x2a
import subprocess
import xasyOptions as xo
import xasyUtils as xu
import tempfile
import uuid
import os
import io


class labelEditor(Qw.QDialog):
    def __init__(self, text=''):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.btnAccept.clicked.connect(self.accept)
        self.ui.btnCancel.clicked.connect(self.reject)
        self.ui.chkMathMode.stateChanged.connect(self.chkMathModeChecked)
        self.ui.btnPreview.clicked.connect(self.btnPreviewOnClick)
        self.ui.btnGetText.clicked.connect(self.btnGetTextOnClick)

        self.svgPreview = None
        self.initializeText(text)

    def initializeText(self, text: str):
        if text[0] == '$' and text[-1] == '$':
            self.ui.chkMathMode.setChecked(True)
            text = text.strip('$')

            if text.startswith('\\displaystyle{'):
                self.ui.cmbMathStyle.setCurrentText('Display Style')
                text = text.rstrip('}')
                text = text.replace('\\displaystyle{', '', 1)
            elif text.startswith('\\scriptstyle{'):
                self.ui.cmbMathStyle.setCurrentText('Script Style')
                text = text.rstrip('}')
                text = text.replace('\\scriptstyle{', '', 1)

        self.ui.txtLabelEdit.setPlainText(text)

    def chkMathModeChecked(self, checked):
        self.ui.cmbMathStyle.setEnabled(checked)

    def getText(self):
        rawText = self.ui.txtLabelEdit.toPlainText()
        rawText.replace('\n', ' ')
        if self.ui.chkMathMode.isChecked():
            prefix = ''
            suffix = ''
            if self.ui.cmbMathStyle.currentText() == 'Display Style':
                prefix = '\\displaystyle{'
                suffix = '}'
            elif self.ui.cmbMathStyle.currentText() == 'Script Style':
                prefix = '\\scriptstyle{'
                suffix = '}'
            return '${0}{1}{2}$'.format(prefix, rawText, suffix)
        else:
            return rawText

    def btnPreviewOnClick(self):
        path = xa.getArgs().asypath
        if path is None:
            opt = xo.xasyOptions().load()
            path = opt['asyPath']

        asyInput = """
        frame f;
        label(f, "{0}");
        write(min(f), newl);
        write(max(f), newl);
        shipout(f);
        """

        self.svgPreview = Qs.QSvgRenderer()
        with tempfile.TemporaryDirectory(prefix='xasylbl_') as tmpdir:
            id = str(uuid.uuid4())
            tmpFile = os.path.join(tmpdir, 'lbl-{0}.svg'.format(id))

            with subprocess.Popen(args=[path, '-fsvg', '-o', tmpFile, '-'], encoding='utf-8', stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE) as asy:
                asy.stdin.write(asyInput.format(self.getText()))
                asy.stdin.close()
                out = asy.stdout.read()

            raw_array = out.splitlines()

            bounds_1, bounds_2 = [val.strip() for val in raw_array]

            min_bounds = xu.listize(bounds_1, (float, float))
            max_bounds = xu.listize(bounds_2, (float, float))

            new_rect = self.processBounds(min_bounds, max_bounds)
            self.svgPreview.load(tmpFile)



        self.drawPreview(new_rect)

    def drawPreview(self, naturalBounds):
        img = Qg.QPixmap(self.ui.lblLabelPreview.size())
        img.fill(Qg.QColor.fromRgbF(1, 1, 1, 1))
        if self.svgPreview is None:
            pass
        else:
            with Qg.QPainter(img) as pnt:
                scale_ratio = self.getIdealScaleRatio(naturalBounds, self.ui.lblLabelPreview.rect())

                pnt.translate(self.ui.lblLabelPreview.rect().center())
                pnt.scale(scale_ratio, scale_ratio)
                self.svgPreview.render(pnt, naturalBounds)
            self.ui.lblLabelPreview.setPixmap(img)


    def getIdealScaleRatio(self, rect, boundsRect):
        assert isinstance(rect, (Qc.QRect, Qc.QRectF))
        assert isinstance(rect, (Qc.QRect, Qc.QRectF))

        magic_ratio = 0.50
        idealRatioHeight = (magic_ratio * boundsRect.height()) / rect.height()
        magicRatioWidth = 0.50

        if idealRatioHeight * rect.width() > magicRatioWidth * boundsRect.width():
            idealRatioWidth = (magicRatioWidth * boundsRect.width()) / rect.width()
            idealRatio = min(idealRatioHeight, idealRatioWidth)
        else:
            idealRatio = idealRatioHeight
        return idealRatio

    def processBounds(self, minPt, maxPt):
        p1x, p1y = minPt
        p2x, p2y = maxPt

        minPt = Qc.QPointF(p1x, p1y)
        maxPt = Qc.QPointF(p2x, p2y)

        newRect = Qc.QRectF(minPt, maxPt)
        return newRect


    def btnGetTextOnClick(self):
        msgbox = Qw.QMessageBox()
        msgbox.setText('Text Preview:\n' + self.getText())
        msgbox.setWindowTitle('Text preview')
        msgbox.show()
        return msgbox.exec_()
