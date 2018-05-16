from pyUIClass.labelTextEditor import Ui_Dialog
import PyQt5.QtCore as Qc
import PyQt5.QtWidgets as Qw


class labelEditor(Qw.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.btnAccept.clicked.connect(self.accept)
        self.ui.btnCancel.clicked.connect(self.reject)
        self.ui.chkMathMode.stateChanged.connect(self.chkMathModeChecked)

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
        pass
    # TODO: Once asy engine is moved to an encapsulated class, work on this... (and when the interactive mode is
    # completely migrated... )

    def btnGetTextOnClick(self):
        msgbox = Qw.QMessageBox()
        msgbox.setText('Text Preview:\n' + self.getText())
        msgbox.setWindowTitle('Text preview')
        msgbox.show()
        return msgbox.exec_()