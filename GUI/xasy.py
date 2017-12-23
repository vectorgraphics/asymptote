#!/usr/bin/env python3

import sys
import PyQt5.QtWidgets as Qw
import xasy2asy
from Window1 import MainWindow1


def main(args):
    qtApp = Qw.QApplication(args)
    mainWin1 = MainWindow1()
    mainWin1.show()

    if len(args) >= 2:
        mainWin1.loadFile(args[1])
    else:
        mainWin1.initializeEmptyFile()
    return qtApp.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
