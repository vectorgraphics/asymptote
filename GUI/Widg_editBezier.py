from pyUIClass.widg_editBezier import Ui_Form

import PyQt5.QtCore as Qc
import PyQt5.QtWidgets as Qw
import PyQt5.QtGui as Qg

class LockMode:
    noLock = 0
    angleLock = 1
    angleAndScaleLock = 2

class Widg_editBezier(Qw.QWidget):
    def __init__(self, info: dict):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.info = info
        self.ui.cmbLockMode.currentIndexChanged[int].connect(self.cmbLockIndexChange)
        self.ui.chkRecompute.stateChanged.connect(self.chkRecomputeChanged)

        self.disableOnAutoRecompute = {self.ui.cmbLockMode, self.ui.btnForceRecompute}

    @property
    def autoRecompute(self) -> bool:
        return self.ui.chkRecompute.isChecked()

    @property
    def lockMode(self) -> int:
        return self.ui.cmbLockMode.currentIndex()

    def cmbLockIndexChange(self, index: int):
        self.info['editBezierlockMode'] = index

    def chkRecomputeChanged(self, checked: int):
        isChecked = (checked == 2)
        for obj in self.disableOnAutoRecompute:
            obj.setEnabled(not checked)
        self.info['autoRecompute'] = checked
