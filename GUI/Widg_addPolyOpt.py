#!/usr/bin/env python3

from pyUIClass.widg_addPolyOpt import Ui_Form
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtGui as QtGui

import sys


class Widg_addPolyOpt(QtWidgets.QWidget):
    def __init__(self, info):
        super().__init__()
        self.ui = Ui_Form()
        self.info = info
        self.ui.setupUi(self)
        self.setFixedSize(self.size())

        self.ui.chkInscribed.setChecked(self.info['inscribed'])
        self.ui.txtSides.setText(str(self.info['sides']))
        self.ui.txtSides.setValidator(QtGui.QIntValidator())

        self.ui.chkInscribed.stateChanged.connect(self.chkInscribedUpdate)
        self.ui.txtSides.textChanged.connect(self.txtSidesUpdate)

    def chkInscribedUpdate(self, checked):
        self.info['inscribed'] = checked

    def txtSidesUpdate(self, text):
        if text:
            self.info['sides'] = int(text)
