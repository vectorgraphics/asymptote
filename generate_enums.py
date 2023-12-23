#!/usr/bin/env python3

# A script to generate enums in different languages from a CSV file.
# A CSV File contains
# enum1, 0
# enum2, ..
# ...
# enumn, n
# where 0,...,n are numbers.
#
# Written by Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com>

from typing import List, Tuple, Any, Union
from datetime import datetime
import io
import argparse
import os
import sys
import time
import re

def cleanComment(s):
    return re.sub(r' *#',' ',s)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-language', '--language', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-name', '--name', type=str, required=True)
    parser.add_argument('-xopt', '--xopt', type=str, nargs='*')
    return parser.parse_args()


def create_enums(filename: str) -> List[Union[Tuple[str, int, str], Tuple[str, int]]]:
    final_list=[]
    with io.open(filename,newline='') as rawfile:
        for line in rawfile.readlines():
            if line.startswith('#') or line.strip() == '':
                continue
            raw_line=line.strip().split(',')
            raw_str, raw_number=raw_line[0:2]
            comment=None
            if len(raw_line)>=3:
                comment=raw_line[-1]
                final_list.append((raw_str.strip(), int(raw_number.strip()),comment))
            else:
                final_list.append((raw_str.strip(), int(raw_number.strip())))
    return final_list


def datetime_now():
    return datetime.utcfromtimestamp(
        int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))
    )


def generate_enum_cpp(outname, enums, name, comment=None, *args, **kwargs):
    with io.open(outname, 'w') as fil:
        fil.write('// Enum class for {0}\n'.format(name))
        if comment is not None:
            fil.write('// {0}\n'.format(comment))
        fil.write('// Generated at {0}\n\n'.format(datetime_now()))
        if 'namespace' in kwargs:
            fil.write('namespace {0}\n'.format(kwargs['namespace']))
            fil.write('{\n')

        fil.write('enum {0} : uint32_t\n'.format(name))
        fil.write('{\n')

        for enumTxt, enumNum, *ar in enums:
            fil.write('{0}={1},\n'.format(enumTxt, enumNum))
            if len(ar) > 0:
                comment=cleanComment(ar[-1])
                if comment is not None:
                    fil.write('// {0}\n\n'.format(comment.strip()))

        fil.write('};\n\n')

        if 'namespace' in kwargs:
            fil.write('}} // namespace {0}\n'.format(kwargs['namespace']))
        fil.write('// End of File\n')


def generate_enum_java(outname, enums, name, comment=None, *args, **kwargs):
    with io.open(outname, 'w') as fil:
        fil.write('// Enum class for {0}\n'.format(name))
        if comment is not None:
            fil.write('// {0}\n'.format(comment))
        fil.write('// Generated at {0}\n\n'.format(datetime_now()))

        if 'package' in kwargs:
            fil.write('package {0};\n'.format(kwargs['package']))
        fil.write('\n')

        fil.write('public enum {0} {{\n'.format(name))

        spaces = kwargs.get('spaces', 4)
        spaces_tab=' '*spaces

        for i in range(len(enums)):
            enumTxt, enumNum, *ar = enums[i]
            endsep=',' if i < len(enums) - 1 else ';'
            fil.write('{2}{0}({1}){3}\n'.format(enumTxt,enumNum,spaces_tab,endsep))
            if len(ar) > 0:
                comment=cleanComment(ar[-1])
                if comment is not None:
                    fil.write('// {0}\n\n'.format(comment.strip()))

        out_lines=[
            '',
            '{0}(int value) {{'.format(name),
            '{0}this.value=value;'.format(spaces_tab),
            '}',
            'public String toString() {',
            '{0}return Integer.toString(value);'.format(spaces_tab),
            '}',
            'private int value;']

        for line in out_lines:
            fil.write(spaces_tab)
            fil.write(line)
            fil.write('\n')
        fil.write('};\n\n')
        fil.write('// End of File\n')


def generate_enum_asy(outname, enums, name, comment=None, *args, **kwargs):
    with io.open(outname, 'w') as fil:
        fil.write('// Enum class for {0}\n'.format(name))
        if comment is not None:
            fil.write('// {0}\n'.format(comment))
        fil.write('// Generated at {0}\n\n'.format(datetime_now()))

        fil.write('struct {0}\n'.format(name))
        fil.write('{\n')

        for enumTxt, enumNum, *ar in enums:
            fil.write('  int {0}={1};\n'.format(enumTxt, enumNum))
            if len(ar) > 0:
                comment=cleanComment(ar[-1])
                if comment is not None:
                    fil.write('// {0}\n\n'.format(comment.strip()))
        fil.write('};\n\n')
        fil.write('{0} {0};'.format(name))

        fil.write('// End of File\n')


def generate_enum_py(outname, enums, name, comment=None, *args, **kwargs):
    with io.open(outname, 'w') as fil:
        fil.write('#!/usr/bin/env python3\n')
        fil.write('# Enum class for {0}\n'.format(name))
        if comment is not None:
            fil.write('""" {0} """\n'.format(comment))
        fil.write('# Generated at {0}\n\n'.format(datetime_now()))
        fil.write('class {0}:\n'.format(name))
        for enumTxt, enumNum, *ar in enums:
            fil.write('    {0}_{2}={1}\n'.format(name, enumNum, enumTxt))
            if len(ar) > 0:
                comment=cleanComment(ar[-1])
                if comment is not None:
                    fil.write('    # {0}\n\n'.format(comment.strip()))
        fil.write('# End of File\n')


def main():
    arg = parse_args()
    if arg.language in {'python', 'py'}:
        fn = generate_enum_py
    elif arg.language in {'cxx', 'c++', 'cpp'}:
        fn = generate_enum_cpp
    elif arg.language in {'asy', 'asymptote'}:
        fn = generate_enum_asy
    elif arg.language in {'java'}:
        fn = generate_enum_java
    else:
        return 1

    custom_args = dict()
    if arg.xopt is not None:
        for xopt in arg.xopt:
            key, val = xopt.split('=')
            custom_args[key] = val

    enums = create_enums(arg.input)
    fn(arg.output, enums, arg.name, 'AUTO-GENERATED from '+arg.input, **custom_args)


if __name__ == '__main__':
    sys.exit(main() or 0)
