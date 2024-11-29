#!/usr/bin/env python3
import argparse
import subprocess as sp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asy-executable", type=str, default="asy")
    parser.add_argument("--output-file", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    asy_output = sp.run(
        [args.asy_executable, "-h"],
        check=True,
        stderr=sp.STDOUT,
        stdout=sp.PIPE,
        text=True,
    )

    with open(args.output_file, "w", encoding="utf-8") as f:
        for line in asy_output.stdout.splitlines():
            stripped_lines = line.strip()
            if (
                stripped_lines.startswith("Asymptote")
                or stripped_lines.startswith("http")
                or stripped_lines.startswith("Usage:")
            ):
                continue
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    main()
