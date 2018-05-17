from pyUIClass.labelTextEditor import Ui_Dialog
import PyQt5.QtWidgets as Qw
import PyQt5.QtSvg as Qs
import PyQt5.QtGui as Qg
import xasyArgs as xa
import xasy2asy as x2a
import xasyOptions as xo
import uuid
import os

class labelEditor(Qw.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.btnAccept.clicked.connect(self.accept)
        self.ui.btnCancel.clicked.connect(self.reject)
        self.ui.chkMathMode.stateChanged.connect(self.chkMathModeChecked)
        self.ui.btnPreview.clicked.connect(self.btnPreviewOnClick)
        self.ui.btnGetText.clicked.connect(self.btnGetTextOnClick)

    def chkMathModeChecked(self, checked):
        self.ui.cmbMathStyle.setEnabled(checked)

    def getText(self):
        rawText = self.ui.txtLabelEdit.toPlainText()
        rawText.replace('\n', ' ')
        if self.ui.chkMathMode.isChecked():
            prefix = ''
            if self.ui.cmbMathStyle.currentText() == 'Display Style':
                prefix = '\\displaystyle'
            elif self.ui.cmbMathStyle.currentText() == 'Script Style':
                prefix = '\\scriptstyle'
            return '${0}{{{1}}}$'.format(prefix, rawText)
        else:
            return rawText

    def btnPreviewOnClick(self):
        path = xa.getArgs().asypath
        if path is None:
            opt = xo.xasyOptions().load()
            path = opt['asyPath']

        tmpFile = '/tmp/' + str(uuid.uuid4()) + '.svg'
        asy = x2a.AsymptoteEngine(path, customOutdir=tmpFile, args=['-f svg'])
        asy.start()
        asy.ostream.write('label("{0}");\n'.format(self.getText()))
        asy.ostream.flush()
        asy.stop()
        asy.wait()

        img = Qg.QPixmap(self.ui.lblLabelPreview.size())
        img.fill(Qg.QColor.fromRgbF(1, 1, 1, 1))
        self.ui.lblLabelPreview.setPixmap(img)
        pnt = Qg.QPainter(img)
        svgRender = Qs.QSvgRenderer()

        while not os.path.isfile(tmpFile):
            continue

        svgRender.load(tmpFile)
        svgRender.render(pnt)

        self.ui.lblLabelPreview.setPixmap(img)
        pnt.end()

        print('done')
        os.remove(tmpFile)




    def btnGetTextOnClick(self):
        msgbox = Qw.QMessageBox()
        msgbox.setText('Text Preview:\n' + self.getText())
        msgbox.setWindowTitle('Text preview')
        msgbox.show()
        return msgbox.exec_()