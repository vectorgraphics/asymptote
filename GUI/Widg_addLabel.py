#!/usr/bin/env python3

from pyUIClass.widg_addLabel import Ui_Form
import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg

import labelEditor
import xasyUtils as xu


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
            self.ui.txtShiftX.setText(str(self.info['shift_x']))

        if self.info['shift_y'] is not None:
            self.ui.txtShiftY.setText(str(self.info['shift_y']))

        
        self.ui.cmbFontSize.setCurrentText(str(self.info['fontSize']) if self.info['fontSize'] is not None else '-')
        self.ui.cmbAlign.setCurrentIndex(self.info['alignIndex'])

        validator = Qg.QDoubleValidator()

        self.ui.txtShiftX.setValidator(validator)
        self.ui.txtShiftY.setValidator(validator)
        self.ui.cmbFontSize.setValidator(validator)

        self.ui.cmbAlign.currentTextChanged.connect(self.updateCheck)
        self.ui.cmbAlign.currentIndexChanged.connect(self.cmbIndexUpdate)
        self.ui.txtShiftX.textEdited.connect(self.shftXUpdate)
        self.ui.txtShiftY.textEdited.connect(self.shftYUpdate)
        self.ui.btnAdvancedEdit.clicked.connect(self.btnAdvancedEditClicked)
        self.ui.cmbFontSize.currentTextChanged.connect(self.cmbFontSizeTextChanged)

        self.updateCheck(self.ui.cmbAlign.currentText())

    def cmbFontSizeTextChanged(self, text: str): 
        tryParseVal = xu.tryParse(text, float)
        self.info['fontSize'] = tryParseVal

    def btnAdvancedEditClicked(self):
        advancedEditDialog = labelEditor.labelEditor(self.ui.txtLabelText.text())
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
            self.info['shift_x'] = float(text)
            self.updateAlign()

    def shftYUpdate(self, text):
        if text:
            self.info['shift_y'] = float(text)
            self.updateAlign()

    def updateAlign(self):
        index = self.ui.cmbAlign.currentIndex()
        self.info['alignIndex'] = index
        if self.ui.cmbAlign.currentText() == 'Custom':
            self.info['align'] = (self.info['shift_x'], self.info['shift_y'])
        elif self.ui.cmbAlign.currentText() == 'None':
            self.info['align'] = (0, 0)
        else:
            self.info['align'] = self.ui.cmbAlign.currentText()

    def cmbIndexUpdate(self, index):
        self.updateAlign()
