#!/usr/bin/env python3
import argparse
import xasyVersion
import PyQt5.QtCore as Qc
# Add arguments here.


def parseArgs(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('-p', '--asypath', help='Custom Asymptote asy executable')
    parser.add_argument('-v', '--version', help='Version number', action='version',
                        version='xasy v{0}'.format(xasyVersion.xasyVersion))
    parser.add_argument('-l', '--language', help='language')
    parser.add_argument('-x', '--mag', help='Magnification. Defaults to 1', default=1, type=float)

    parser.add_argument(
            'filename', help='Filename to load. If empty, initializes empty xasy canvas.', nargs='?', default=None)

    return parser.parse_args()


def getArgs():
    return parseArgs(Qc.QCoreApplication.arguments())
