#!/usr/bin/env python3
###########################################################################
#
# xasyFile implements the loading, parsing, and saving of an xasy file.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
############################################################################

from string import *
import xasy2asy as x2a
import io
import re


class xasyParseError(Exception):
    """A parsing error"""
    pass


class xasyFileError(Exception):
    """An i/o error or other error not related to parsing"""
    pass


def extractTransform(line):
    """Returns key and the new transform."""
    # see https://regex101.com/r/6DqkRJ/4 for info
    mapString = x2a.xasyItem.mapString
    testMatch = re.match(
        r'^{0:s}\s*\(\s*\"([^\"]+)\"\s*,\s*\(([-\d, .]+)\)\s*\)'.format(mapString), line.strip())
    if testMatch is None:
        mapOnlyMatch = re.match(r'^{0:s}\s*\(\s *\"([^\"]+)\"\s*\)'.format(mapString), line.strip())
        if mapOnlyMatch is None:
            return None
        else:
            key = mapOnlyMatch.group(1)
            return key, x2a.identity()
    else:
        key = testMatch.group(1)
        rawStr = testMatch.group(2)
        rawStrArray = rawStr.split(',')

        if len(rawStrArray) != 6:
            return None
        transf = [float(val.strip()) for val in rawStrArray]
        return key, x2a.asyTransform(transf)


def extractTransformsFromFile(fileStr):
    transfDict = {}
    maxItemCount = 0
    with io.StringIO() as rawCode:
        for line in fileStr.splitlines():
            test_transf = extractTransform(line.rstrip())
            if test_transf is None:
                rawCode.write(line + '\n')
            else:
                key, transf = test_transf
                if key not in transfDict.keys():
                    transfDict[key] = []
                transfDict[key].append(transf)

                # see https://regex101.com/r/RgeBVc/2 for regex

                testNum = re.match(r'^x(\d+)($|:.*$)', key)
                if testNum is not None:
                    maxItemCount = max(maxItemCount, int(testNum.group(1)))
        final_str = rawCode.getvalue()
    return final_str, transfDict, maxItemCount


def saveFile(file, xasyItems, asy2psmap):
    """Write a list of xasyItems to a file"""
    for item in xasyItems:
        file.write(item.getTransformCode(asy2psmap))

    for item in xasyItems:
        file.write(item.getObjectCode(asy2psmap))

    file.write('size('+str(asy2psmap*x2a.yflip())+'); '+ x2a.xasyItem.resizeComment+'\n')
