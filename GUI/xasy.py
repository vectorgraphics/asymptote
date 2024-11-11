#!/usr/bin/env python3

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent))

import signal, os
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
from Window1 import MainWindow1

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
