#!/usr/bin/env python3
###########################################################################
#
# xasyFile implements the loading, parsing, and saving of an asy file.
#
#
# Author: Orest Shardt
# Created: June 29, 2007
#
############################################################################

from string import *
import xasy2asy as xasy2asy
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
    mapString = xasy2asy.xasyItem.mapString
    testMatch = re.match(
        r'^{0:s}\s*\(\s*\"([^\"]+)\"\s*,\s*\(([-\d, .]+)\)\s*\)'.format(mapString), line.strip())
    if testMatch is None:
        mapOnlyMatch = re.match(r'^{0:s}\s*\(\s *\"([^\"]+)\"\s*\)'.format(mapString), line.strip())
        if mapOnlyMatch is None:
            return None
        else:
            key = mapOnlyMatch.group(1)
            return key, xasy2asy.identity()
    else:
        key = testMatch.group(1)
        rawStr = testMatch.group(2)
        rawStrArray = rawStr.split(',')

        if len(rawStrArray) != 6:
            return None
        transf = [float(val.strip()) for val in rawStrArray]
        return key, xasy2asy.asyTransform(transf)


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
        final_str = rawCode.getvalue()
    return final_str, transfDict

def xasy2asyCode(xasyItems, asy2psmap):
    asyCode = ''
    for item in xasyItems:
        asyCode += item.getTransformCode(asy2psmap)
    for item in xasyItems:
        asyCode += item.getObjectCode(asy2psmap)

    asyCode += 'size('+str(asy2psmap*xasy2asy.yflip())+'); '+ xasy2asy.xasyItem.resizeComment+'\n'
    return asyCode

def saveFile(file, xasyItems, asy2psmap):
    """Write a list of xasyItems to a file"""
    file.write(xasy2asyCode(xasyItems, asy2psmap))

def xasyToDict(file, xasyItems, asy2psmap):
    fileItems = []
    asyItems = []
    for item in xasyItems:
        if isinstance(item, xasy2asy.xasyScript):
            # reusing xasyFile code for objects
            # imported from asy script.
            asyItems.append({'item':item, 'type': 'xasyScript'})

        elif isinstance(item, xasy2asy.xasyText):
            # At the moment xasyText cannot be edited
            # so we treat it the same as xasyScript
            penData = {'color': item.pen.color, 'width': item.pen.width, 'options': item.pen.options}
            fileItems.append({'type': 'xasyText',
                    'align': item.label.align,
                    'location': item.label.location,
                    'fontSize': item.label.fontSize,
                    'text': item.label.text,
                    'transform': item.transfKeymap[item.transfKey][0].t,
                    'transfKey': item.transfKey,
                    'pen': penData
                    })

        elif isinstance(item, xasy2asy.xasyShape):
            penData = {'color': item.pen.color, 'width': item.pen.width, 'dashPattern': item.pen.dashPattern, 'options': item.pen.options}
            fileItems.append({'type': 'xasyShape',
                    'nodes': item.path.nodeSet,
                    'links': item.path.linkSet,
                    'fill': item.path.fill,
                    'transform': item.transfKeymap[item.transfKey][0].t,
                    'transfKey': item.transfKey,
                    'pen': penData
                    })

        elif isinstance(item, xasy2asy.asyArrow): #Will this ever even be reached?
            penData = {'color': item.pen.color, 'width': item.pen.width, 'dashPattern': item.pen.dashPattern, 'options': item.pen.options}
            fileItems.append({'type': 'asyArrow',
                    'pen': penData,
                    'arrowSettings': item.arrowSettings,
                    'transform': item.transfKeymap[item.transfKey][0].t,
                    'transfKey': item.transfKey,
                    'settings': item.arrowSettings,
                    'code': item.code
                    })

        else:
            # DEBUGGING PURPOSES ONLY
            print(type(item))

    return {'objects': fileItems, 'asy2psmap': asy2psmap.t}, asyItems
