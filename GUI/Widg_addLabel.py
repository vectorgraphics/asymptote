from pyUIClass.widg_addLabel import Ui_Form
import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg

import labelEditor

class Widg_addLabel(Qw.QWidget):
    def __init__(self, info):
        super().__init__()
        self.ui = Ui_Form()
        self.info = info
        self.ui.setupUi(self)
        self.setFixedSize(self.size())

        if 'alignIndex' not in self.info.keys():
            self.info['alignIndex'] = 0

        if 'shift_x' not in self.info.keys():
            self.info['shift_x'] = None

        if 'shift_y' not in self.info.keys():
            self.info['shift_y'] = None

        if 'align' not in self.info.keys():
            self.info['align'] = (0, 0)

        if self.info['shift_x'] is not None:
            self.ui.txtShiftX.setText(str(self.info['shfit_x']))

        if self.info['shift_y'] is not None:
            self.ui.txtShiftY.setText(str(self.info['shfit_y']))

        self.ui.cmbAlign.setCurrentIndex(self.info['alignIndex'])

        validator = Qg.QDoubleValidator()

        self.ui.txtShiftX.setValidator(validator)
        self.ui.txtShiftY.setValidator(validator)

        self.ui.cmbAlign.currentTextChanged.connect(self.updateCheck)
        self.ui.cmbAlign.currentIndexChanged.connect(self.cmbIndexUpdate)
        self.ui.txtShiftX.textEdited.connect(self.shftXUpdate)
        self.ui.txtShiftY.textEdited.connect(self.shftYUpdate)
        self.ui.btnAdvancedEdit.clicked.connect(self.btnAdvancedEditClicked)

        self.updateCheck(self.ui.cmbAlign.currentText())

    def btnAdvancedEditClicked(self):
        advancedEditDialog = labelEditor.labelEditor()
        advancedEditDialog.show()
        result = advancedEditDialog.exec_()
        if result == Qw.QDialog.Accepted:
            self.ui.txtLabelText.setText(advancedEditDialog.getText())

    @property
    def labelText(self):
        return self.ui.txtLabelText.text()

    def updateCheck(self, a0):
        self.ui.txtShiftX.setEnabled(a0 == 'Custom')
        self.ui.txtShiftY.setEnabled(a0 == 'Custom')

    def shftXUpdate(self, text):
        if text:
            self.info['shfit_x'] = float(text)

    def shftYUpdate(self, text):
        if text:
            self.info['shift_x'] = float(text)

    def cmbIndexUpdate(self, index):
        self.info['alignIndex'] = index
        if self.ui.cmbAlign.currentText() == 'Custom':
            self.info['align'] = (self.info['shift_x'], self.info['shift_y'])
        elif self.ui.cmbAlign.currentText() == 'None':
            self.info['align'] = (0, 0)
        else:
            self.info['align'] = self.ui.cmbAlign.currentText()
