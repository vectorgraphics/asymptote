#!/usr/bin/env python3

import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc
from pyUIClass.setCustomAnchor import Ui_Dialog


class CustomAnchorDialog(Qw.QDialog):

    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.buttonBox.button(Qw.QDialogButtonBox.Reset).clicked.connect(self.resetDialog)

        validator = Qg.QDoubleValidator()

        self.ui.lineEditX.setValidator(validator)
        self.ui.lineEditY.setValidator(validator)

        self.ui.lineEditX.textChanged.connect(self.checkTextChanged)
        self.ui.lineEditY.textChanged.connect(self.checkTextChanged)

    def checkTextChanged(self, text):
        if str(text) not in {'.', '-', '.-', '-.'} and str(text):
            self.ui.buttonBox.button(Qw.QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.ui.buttonBox.button(Qw.QDialogButtonBox.Ok).setEnabled(False)

    def getPoint(self):
        xPoint = float(self.ui.lineEditX.text())
        yPoint = float(self.ui.lineEditY.text())

        return Qc.QPointF(xPoint, yPoint)

    def handleBtnBoxClick(self, button):
        assert isinstance(button, Qw.QAbstractButton)
        if button.text() == 'Reset':
            self.resetDialog()

    def resetDialog(self):
        self.ui.lineEditX.setText('0')
        self.ui.lineEditY.setText('0')
