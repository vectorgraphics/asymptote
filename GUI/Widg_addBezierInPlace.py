#!/usr/bin/env python3

from pyUIClass.widg_addBezierInplace import Ui_Form
import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg
import PyQt5.QtCore as Qc

import sys


class Widg_addBezierInplace(Qw.QWidget):
    def __init__(self, info):
        super().__init__()
        self.ui = Ui_Form()
        self.info = info
        self.ui.setupUi(self)
        self.setFixedSize(self.size())

        self.ui.chkBezierCurve.stateChanged.connect(self.chkBezierCurveUpdate)
        
    def chkBezierCurveUpdate(self, checked):
        self.info['useBezier'] = checked
    
