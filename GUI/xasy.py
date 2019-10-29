#!/usr/bin/env python3

import sys,signal
import PyQt5.QtWidgets as Qw
from Window1 import MainWindow1


def main(args):
    qtApp = Qw.QApplication(args)
    signal.signal(signal.SIGINT,signal.SIG_DFL)
    mainWin1 = MainWindow1()
    mainWin1.show()
    return qtApp.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
