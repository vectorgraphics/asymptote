#!/usr/bin/env python3

import io
import json

header="""#compdef _asy asy

function _asy () {
    local -a asyCmds
    asycmds=(\\
"""

footer="""
    )
    _describe asy asycmds
}

compdef _asy asy
"""

def main():
    f = io.open('args.json')
    items = json.loads(f.read())
    f.close()

    itemList = []

    for item in items:
        if item['argname'] != 'NULL':
            if item['desc'] != 'NULL':
                itemList.append('    "-{0:s}:{1:s}"{2}'.format(item['argname'], item['desc'], '\\' ))

    itemList.sort()

    with io.StringIO() as rawCode:
        rawCode.write(header)
        rawCode.write('\n'.join(itemList))
        rawCode.write(footer)

        print(rawCode.getvalue())


if __name__ == '__main__':
    main()
