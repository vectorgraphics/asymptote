#!/usr/bin/env python3
import argparse
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--texify-loc", required=True)
    parser.add_argument("--texindex-loc", required=True)
    parser.add_argument("--texi-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    env = os.environ.copy()
    env["TEXINDEX"] = args.texindex_loc
    subprocess.run(
        [args.texify_loc, "--pdf", args.texi_file],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
