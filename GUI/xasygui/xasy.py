#!/usr/bin/env python3

import os
import signal
import sys

import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

from .Window1 import MainWindow1


def main(args):
    os.environ["QT_LOGGING_RULES"]="*.debug=false;qt.qpa.*=false"
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps,True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)
    qtApp = QtWidgets.QApplication(args)
    signal.signal(signal.SIGINT,signal.SIG_DFL)
    mainWin1 = MainWindow1()
    mainWin1.show()
    return qtApp.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
