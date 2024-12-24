#!/usr/bin/env python3
__doc__ = """
Script to generate asy.list file. Equivalent to Makefile's asy-list.el file's logic
to generate asy.list.
"""

import argparse
import pathlib
import subprocess as sp
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asy-executable", required=True, help="Asymptote executable")
    parser.add_argument("--asy-base-dir", required=True, help="Asymptote base dir")
    parser.add_argument("--output-file", required=True, help="Output file")
    return parser.parse_args()


def run_asy_list(asy_exec: str, base_dir: pathlib.Path, asy_file: Optional[str] = None):
    base_args = [asy_exec, "-dir", str(base_dir), "-config", '""', "-render", "0", "-l"]
    if asy_file is not None:
        base_args.append(asy_file)
    out_data = sp.run(
        base_args,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        universal_newlines=True,
        check=True,
    )
    return out_data.stdout


def base_file_to_be_included_in_list_file(base_file_name: str):
    if base_file_name.startswith("plain") or base_file_name.startswith("three_"):
        return False
    if "map" in base_file_name:
        return False
    return True


def main():
    args = parse_args()
    base_dir = pathlib.Path(args.asy_base_dir)
    base_asy_list = run_asy_list(args.asy_executable, base_dir)

    base_file: pathlib.Path
    base_files_to_generate_list = [
        base_file
        for base_file in base_dir.glob("*.asy")
        if base_file_to_be_included_in_list_file(base_file.name)
    ]
    base_file_asy_lists = [
        run_asy_list(args.asy_executable, base_dir, str(base_file))
        for base_file in base_files_to_generate_list
    ]
    with open(args.output_file, "w", encoding="utf-8") as fil:
        fil.write(base_asy_list)
        for asy_list_info in base_file_asy_lists:
            fil.write(asy_list_info)


if __name__ == "__main__":
    main()
