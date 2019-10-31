#!/usr/bin/env python3

import sys,signal,os
import PyQt5.QtWidgets as Qw
import PyQt5.QtCore as Qc
from Window1 import MainWindow1

def main(args):
    Qw.QApplication.setAttribute(Qc.Qt.AA_UseHighDpiPixmaps,True)
    Qw.QApplication.setAttribute(Qc.Qt.AA_EnableHighDpiScaling,True)
    qtApp = Qw.QApplication(args)
    signal.signal(signal.SIGINT,signal.SIG_DFL)
    mainWin1 = MainWindow1()
    mainWin1.show()
    return qtApp.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
