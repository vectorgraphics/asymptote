#!/usr/bin/env python3

__doc__ = """
Determines asymptote version for release, or with git info for development
"""
__author__ = "Supakorn 'Jamie' Rassameemasmuang <jamievlin@outlook.com>"

import pathlib
import re
import subprocess
from argparse import ArgumentParser
from subprocess import CalledProcessError

from determine_pkg_info import determine_asy_pkg_info


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base-version",
        type=str,
        help="Base version string. If blank, this information is "
        + "fetched from configure.ac",
    )
    version_mode = parser.add_mutually_exclusive_group()
    version_mode.add_argument("--version-for-release", action="store_true")
    version_mode.add_argument(
        "--version-with-git-info",
        action="store_true",
        help="Includes number of commits since last tag, if directory is a git repo. "
        + "If commit information cannot be determined, reverts to baseline version",
    )

    return parser.parse_args()


def determine_version_for_release(version_base: str):
    git_string = "git"
    if version_base.endswith("git"):
        return version_base[: -len(git_string)]
    return version_base


def determine_version_with_git_info(version_base: str):
    try:
        long_description = subprocess.run(
            ["git", "describe", "--long"],
            cwd=pathlib.Path(__file__).parent,
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
        )
    except CalledProcessError:
        return version_base

    first_version_base = re.sub(r"git-([0-9]*)-g.*", r"-\1", long_description.stdout)
    return re.sub(r"-0-g.*", r"", first_version_base)


def main():
    args = parse_args()
    version_base = args.base_version or determine_asy_pkg_info()["version-base"]

    if args.version_for_release:
        version = determine_version_for_release(version_base)
    elif args.version_with_git_info:
        version = determine_version_with_git_info(version_base)
    else:
        version = version_base

    print(version, end="")


if __name__ == "__main__":
    main()
