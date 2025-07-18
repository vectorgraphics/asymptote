#!/usr/bin/env python3

import PySide6.QtWidgets as QtWidgets
import PySide6.QtGui as QtGui
import PySide6.QtCore as QtCore
from xasyqtui.setCustomAnchor import Ui_Dialog


class CustomAnchorDialog(QtWidgets.QDialog):

    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Reset).clicked.connect(self.resetDialog)

        validator = QtGui.QDoubleValidator()

        self.ui.lineEditX.setValidator(validator)
        self.ui.lineEditY.setValidator(validator)

        self.ui.lineEditX.textChanged.connect(self.checkTextChanged)
        self.ui.lineEditY.textChanged.connect(self.checkTextChanged)

    def checkTextChanged(self, text):
        if str(text) not in {'.', '-', '.-', '-.'} and str(text):
            self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        else:
            self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def getPoint(self):
        xPoint = float(self.ui.lineEditX.text())
        yPoint = float(self.ui.lineEditY.text())

        return QtCore.QPointF(xPoint, yPoint)

    def handleBtnBoxClick(self, button):
        assert isinstance(button, QtWidgets.QAbstractButton)
        if button.text() == 'Reset':
            self.resetDialog()

    def resetDialog(self):
        self.ui.lineEditX.setText('0')
        self.ui.lineEditY.setText('0')
