#!/usr/bin/env python3
# pylint: disable=too-many-locals,unused-argument,keyword-arg-before-vararg

# A script to generate enums in different languages from a CSV file.
# A CSV File contains
# enum1, 0
# enum2, ..
# ...
# enumn, n
# where 0,...,n are numbers.
#
# Written by Supakorn "Jamie" Rassameemasmuang <jamievlin@outlook.com>

import argparse
import io
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple, Union


def cleanComment(s):
    return re.sub(r" *#", " ", s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-language", "--language", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-name", "--name", type=str, required=True)
    parser.add_argument("-xopt", "--xopt", type=str, nargs="*")
    return parser.parse_args()


def create_enums(filename: str) -> List[Union[Tuple[str, int, str], Tuple[str, int]]]:
    final_list = []
    with io.open(filename, newline="", encoding="utf-8") as rawfile:
        for line in rawfile.readlines():
            if line.startswith("#") or line.strip() == "":
                continue
            raw_line = line.strip().split(",")
            raw_str, raw_number = raw_line[0:2]
            comment = None
            if len(raw_line) >= 3:
                comment = raw_line[-1]
                final_list.append((raw_str.strip(), int(raw_number.strip()), comment))
            else:
                final_list.append((raw_str.strip(), int(raw_number.strip())))
    return final_list


def datetime_now():
    return datetime.fromtimestamp(
        int(os.environ.get("SOURCE_DATE_EPOCH", time.time())), tz=timezone.utc
    )


def generate_enum_cpp(outname, enums, name, comment=None, *args, **kwargs):
    with io.open(outname, "w", encoding="utf-8") as fil:
        fil.write(f"// Enum class for {name}\n")
        if comment is not None:
            fil.write(f"// {comment}\n")
        if "namespace" in kwargs:
            fil.write(f"namespace {kwargs['namespace']}\n")
            fil.write("{\n")

        fil.write(f"enum {name} : uint32_t\n")
        fil.write("{\n")

        for enumTxt, enumNum, *ar in enums:
            if len(ar) > 0:
                comment = cleanComment(ar[-1])
                if comment is not None:
                    fil.write(f"// {comment.strip()}\n")
            fil.write(f"{enumTxt}={enumNum},\n\n")

        fil.write("};\n\n")

        if "namespace" in kwargs:
            fil.write(f"}} // namespace {kwargs['namespace']}\n")
        fil.write("// End of File\n")


def generate_enum_java(outname, enums, name, comment=None, *args, **kwargs):
    with io.open(outname, "w", encoding="utf-8") as fil:
        fil.write(f"// Enum class for {name}\n")
        if comment is not None:
            fil.write(f"// {comment}\n")
        if "package" in kwargs:
            fil.write(f"package {kwargs['package']};\n")
        fil.write("\n")

        fil.write(f"public enum {name} {{\n")

        spaces = kwargs.get("spaces", 4)
        spaces_tab = " " * spaces

        for i, enum in enumerate(enums):
            enumTxt, enumNum, *ar = enum
            endsep = "," if i < len(enums) - 1 else ";"
            fil.write(f"{spaces_tab}{enumTxt}({enumNum}){endsep}\n")
            if len(ar) > 0:
                comment = cleanComment(ar[-1])
                if comment is not None:
                    fil.write(f"// {comment.strip()}\n\n")

        out_lines = [
            "",
            f"{name}(int value) {{",
            f"{spaces_tab}this.value=value;",
            "}",
            "public String toString() {",
            f"{spaces_tab}return Integer.toString(value);",
            "}",
            "private int value;",
        ]

        for line in out_lines:
            fil.write(spaces_tab)
            fil.write(line)
            fil.write("\n")
        fil.write("};\n\n")
        fil.write("// End of File\n")


def generate_enum_asy(outname, enums, name, comment=None, *args, **kwargs):
    with io.open(outname, "w", encoding="utf-8") as fil:
        fil.write(f"// Enum class for {name}\n")
        if comment is not None:
            fil.write(f"// {comment}\n")
        fil.write(f"struct {name}\n")
        fil.write("{\n")

        for enumTxt, enumNum, *ar in enums:
            fil.write(f"  int {enumTxt}={enumNum};\n")
            if len(ar) > 0:
                comment = cleanComment(ar[-1])
                if comment is not None:
                    fil.write(f"// {comment.strip()}\n\n")
        fil.write("};\n\n")
        fil.write(f"{name} {name};")

        fil.write("// End of File\n")


def generate_enum_py(outname, enums, name, comment=None, *args, **kwargs):
    with io.open(outname, "w", encoding="utf-8") as fil:
        fil.write("#!/usr/bin/env python3\n")
        fil.write(f"# Enum class for {name}\n")
        if comment is not None:
            fil.write(f'""" {comment} """\n')
        fil.write(f"class {name}:\n")
        for enumTxt, enumNum, *ar in enums:
            fil.write(f"    {name}_{enumTxt}={enumNum}\n")
            if len(ar) > 0:
                comment = cleanComment(ar[-1])
                if comment is not None:
                    fil.write(f"    # {comment.strip()}\n\n")
        fil.write("# End of File\n")


def main():
    arg = parse_args()
    if arg.language in {"python", "py"}:
        fn = generate_enum_py
    elif arg.language in {"cxx", "c++", "cpp"}:
        fn = generate_enum_cpp
    elif arg.language in {"asy", "asymptote"}:
        fn = generate_enum_asy
    elif arg.language in {"java"}:
        fn = generate_enum_java
    else:
        return 1

    custom_args = {}
    if arg.xopt is not None:
        for xopt in arg.xopt:
            key, val = xopt.split("=")
            custom_args[key] = val

    enums = create_enums(arg.input)
    fn(arg.output, enums, arg.name, "AUTO-GENERATED from " + arg.input, **custom_args)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
