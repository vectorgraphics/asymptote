import argparse
import xasyVersion
# Add arguments here.


def parseArgs(args):
    parser = argparse.ArgumentParser(args)
    parser.add_argument('-f', '--file', help='Filename to load. If empty, initializes empty xasy canvas.')
    parser.add_argument('-p', '--asypath', help='Custom Asymptote asy executable')
    parser.add_argument('-v', '--version', help='Version number', action='version',
                        version='xasy v{0}'.format(xasyVersion.xasyVersion))
    parser.add_argument('-l', '--language', help='language')

    return parser.parse_args()
