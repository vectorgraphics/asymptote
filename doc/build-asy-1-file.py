#!/usr/bin/env python3
import io
import re
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--options-file", type=str, required=True)
    parser.add_argument("--asy-1-begin-file", type=str, required=True)
    parser.add_argument("--asy-1-end-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.options_file, "r", encoding="utf-8") as optfile:
        options = [
            line.strip() for line in optfile.readlines() if line.strip().startswith("-")
        ]

    args_description_extract_regex = re.compile(r"-(.*?) {2}\s*([a-zA-Z0-9].*)")
    arg_matches = [args_description_extract_regex.match(line) for line in options]
    escaped_args_with_descs = [
        (match.group(1).replace("-", r"\-"), match.group(2))
        for match in arg_matches
        if match is not None
    ]
    transformed_args = [
        rf""".TP
.B \-{arg}
{desc}."""
        for arg, desc in escaped_args_with_descs
    ]

    output = None
    try:
        output = io.StringIO()
        with open(args.asy_1_begin_file, "r", encoding="utf-8") as f:
            output.write(f.read())

        output.write("\n".join(transformed_args))

        with open(args.asy_1_end_file, "r", encoding="utf-8") as f:
            output.write(f.read())

        with open(args.out_file, "w", encoding="utf-8") as out_file:
            out_file.write(output.getvalue())
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
