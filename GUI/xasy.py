#!/usr/bin/env python3

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent))

from xasygui import xasy  # noqa


if __name__ == '__main__':
    sys.exit(xasy.main(sys.argv) or 0)
