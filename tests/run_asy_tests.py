#!/usr/bin/env python3

import argparse
import pathlib
import subprocess

# ---------- setting up criteria for running tests

# these tests will not be run, even in all-tests mode
DIR_TO_EXCLUDE_FROM_ALL_TESTS = {"bench"}

# ------------------------------------------------
TEST_ROOT_DIR = pathlib.Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asy-is-from-makefile",
        action="store_true",
        help=(
            "If true, will use asymptote in <asymptote-dir>/asy and "
            + "<asymptote-dir>/base directory as executable. "
            + "This option is mutually exclusive with --asy and --asy-base-dir."
        ),
    )
    parser.add_argument(
        "--asy",
        type=str,
        help=(
            "Asymptote executable. "
            + "This argument is required if not running on makefile mode"
        ),
    )
    parser.add_argument(
        "--asy-base-dir",
        type=str,
        help=(
            "asy base directory. "
            + "This argument is required if not running on makefile mode"
        ),
    )

    test_mode_parser = parser.add_mutually_exclusive_group(required=False)
    test_mode_parser.add_argument(
        "--tests-list",
        type=str,
        help=(
            "A semicolon-separated list of tests to run. "
            + "For example, array/fields points to array/fields.asy test"
        ),
    )
    return parser.parse_args()


class TestRunner:  # pylint: disable=too-few-public-methods
    def __init__(self, asy_exec: str, asy_base_dir: str):
        self._asy_base_args = [
            asy_exec,
            "-dir",
            asy_base_dir,
            "-o",
            "out",
            "-globalwrite",
            "-noV",
        ]

    def run_test(self, test_path: pathlib.Path):
        print(f"Runing {test_path}")
        try:
            out_process = subprocess.run(
                self._asy_base_args + [test_path],
                cwd=TEST_ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
                universal_newlines=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Test {test_path} failed")
            print(e.stdout)
            raise RuntimeError("Asymptote test failed") from e

        print(out_process.stdout)
        print()


def glob_tests():
    return sorted(
        entry
        for entry in TEST_ROOT_DIR.glob("*/*.asy")
        if entry.is_file()
        if entry.parent.name not in DIR_TO_EXCLUDE_FROM_ALL_TESTS
    )


def main():
    args = parse_args()
    if args.asy_is_from_makefile and (args.asy or args.asy_base_dir):
        raise ValueError(
            "Cannot specify both --asy/--asy-base-dir while using makefile format"
        )

    if args.asy_is_from_makefile:
        asy_exec = str(TEST_ROOT_DIR / "../asy")
        asy_base_dir = str(TEST_ROOT_DIR / "../base")
    else:
        asy_exec = args.asy
        asy_base_dir = args.asy_base_dir
        if (not asy_exec) or (not asy_base_dir):
            raise ValueError(
                "--asy and --asy-base-dir are required when not using Makefile mode"
            )

    runner = TestRunner(asy_exec, asy_base_dir)
    if args.tests_list:
        test_lists_base = args.tests_list.split(";")
        test_list = [
            TEST_ROOT_DIR / (test_name + ".asy") for test_name in test_lists_base
        ]
    else:
        test_list = glob_tests()

    print(f"Running {len(test_list)} test files")
    for test in test_list:
        runner.run_test(test)
    print("All tests passed.")


if __name__ == "__main__":
    main()
