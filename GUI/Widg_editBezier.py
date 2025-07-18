#!/usr/bin/env python3

from xasyqtui.widg_editBezier import Ui_Form

import PySide6.QtWidgets as QtWidgets
import PySide6.QtCore as QtCore

class LockMode:
    noLock = 0
    angleLock = 1
    angleAndScaleLock = 2

class Widg_editBezier(QtWidgets.QWidget):
    def __init__(self, info: dict, enableCurveFeatures: bool=True):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.info = info

        self.ui.chkRecompute.setChecked(self.info['autoRecompute'])
        self.ui.cmbLockMode.setCurrentIndex(self.info['editBezierlockMode'])

        self.ui.cmbLockMode.currentIndexChanged[int].connect(self.cmbLockIndexChange)
        self.ui.chkRecompute.stateChanged.connect(self.chkRecomputeChanged)

        self.disableOnAutoRecompute = {self.ui.cmbLockMode, self.ui.btnForceRecompute}
        self.curveBtnsOnly = {self.ui.cmbLockMode, self.ui.btnForceRecompute, self.ui.chkRecompute}

        for elem in self.curveBtnsOnly:
            elem.setEnabled(enableCurveFeatures)

    @property
    def autoRecompute(self) -> bool:
        return self.ui.chkRecompute.isChecked()

    @property
    def lockMode(self) -> int:
        return self.ui.cmbLockMode.currentIndex()

    @QtCore.Slot(int)
    def cmbLockIndexChange(self, index: int):
        self.info['editBezierlockMode'] = index

    @QtCore.Slot(int)
    def chkRecomputeChanged(self, checked: int):
        isChecked = (checked == 2)
        for obj in self.disableOnAutoRecompute:
            obj.setEnabled(not checked)
        self.info['autoRecompute'] = checked

        if isChecked:
            self.ui.btnForceRecompute.clicked.emit()
