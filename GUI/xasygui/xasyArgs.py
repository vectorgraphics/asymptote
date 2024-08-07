#!/usr/bin/env python3
import argparse
import PyQt5.QtCore as QtCore
from xasyversion.version import VERSION
# Add arguments here.


def parseArgs(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('-p', '-asypath', '--asypath',
                        help='Custom path to asy executable')
    parser.add_argument('-v', '-version', '--version',
                        help='Version number', action='version',
                        version='xasy v{0}'.format(VERSION))
    parser.add_argument('-l', '-language', '--language',
                        help='language')
    parser.add_argument('-x', '-mag', '--mag',
                        help='Initial magnification. Defaults to 1',
                        default=1, type=float)
    parser.add_argument('-render', '--render',
                        help='Number of pixels per bp in 3D rendered bitmaps',
                        default=None, type=float)
    parser.add_argument('-additional-asy-args', '--additional-asy-args',
                        help='Comma-separated values of additional arguments to pass'
                             'to Asymptote',
                        dest='additionalAsyArgs',
                        type=str, default=None)
    parser.add_argument(
        'filename',
        help='Filename to load (if omitted, initialize blank canvas)',
        nargs='?', default=None)

    return parser.parse_args()


def getArgs():
    return parseArgs(QtCore.QCoreApplication.arguments())
