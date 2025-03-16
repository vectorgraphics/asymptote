#!/usr/bin/env python3

__doc__ = """
This file scans tests directory and generates a cmake file containing all *.asy tests,
excluding wce.
The resulting file is supposed to be checked in and imported by the main cmake build
scripts.

This script has 2 modes - list generation and verification. Verification does not
write to any file, but raises an error if the file output does not match the contents
of the test list cmake file (that is, could be missing a test).

To run in verification mode, pass in "--verify-no-missing-tests" arguments.
"""

import io
import os
import pathlib
import sys
import textwrap

REPO_ROOT = pathlib.Path(__file__).parent
TESTS_DIR = REPO_ROOT / "tests"
GENERATED_TESTS_LIST_CMAKE_FILE_PATH = (
    REPO_ROOT / "cmake-scripts/generated/asy-tests-list.cmake"
)

TESTS_REQUIRING_CMAKE_FEATURES = {"gsl": "ENABLE_GSL", "gc": "ENABLE_GC"}

TESTS_NOT_PART_OF_CORE_CHECKS = {"gc", "gs"}

TESTS_WITH_ARTIFACTS = {"gc": [".eps"], "output": ["circle.eps", "line.eps"]}

EXCLUDED_TESTS = {"bench"}

GENERATED_CMAKE_FILE_COMMENT_HEADER = textwrap.dedent(
    f"""
    # This file is automatically generated. Do not modify manually.
    # This file is checked in as part of the repo. It is not meant to be ignored.
    #
    # If more tests are added, run {pathlib.Path(__file__).name} to re-generate
    # the test list.
    """
)


def generate_tests_list_per_directory(test_dir: os.DirEntry):
    test_name = test_dir.name
    if test_name in EXCLUDED_TESTS:
        return ""

    with os.scandir(test_dir) as scanner_it:
        tests = sorted(
            entry.name[:-4]  # removing .asy extension
            for entry in scanner_it
            if entry.is_file() and entry.name.endswith(".asy")
        )

    if test_name in TESTS_WITH_ARTIFACTS:
        artifacts_text = f"TEST_ARTIFACTS {' '.join(TESTS_WITH_ARTIFACTS[test_name])}"
    else:
        artifacts_text = ""

    test_not_check_str = (
        "TEST_NOT_PART_OF_CHECK_TEST true"
        if test_name in TESTS_NOT_PART_OF_CORE_CHECKS
        else ""
    )

    with io.StringIO() as text_writer:
        if test_name in TESTS_REQUIRING_CMAKE_FEATURES:
            text_writer.write(f"if ({TESTS_REQUIRING_CMAKE_FEATURES[test_name]})\n")

        text_writer.write(
            textwrap.dedent(
                f"""
                    add_asy_tests(
                        TEST_DIR {test_name}
                        TESTS {' '.join(tests)}
                        {artifacts_text}
                        {test_not_check_str}
                    )
                """
            )
        )

        if test_name in TESTS_REQUIRING_CMAKE_FEATURES:
            text_writer.write("endif()\n")
        return text_writer.getvalue()


def write_cmake_lists_data_to_file(test_dirs, out_file):
    out_file.write(GENERATED_CMAKE_FILE_COMMENT_HEADER)
    out_file.write("\n")
    for entry in test_dirs:
        cmake_text = generate_tests_list_per_directory(entry)
        out_file.write(cmake_text)
        out_file.write("\n")


def get_test_dirs():
    with os.scandir(TESTS_DIR) as scanner_it:
        return sorted(
            (entry for entry in scanner_it if entry.is_dir()),
            key=lambda entry: entry.name,
        )


def main():
    # sort to make output deterministic
    args = sys.argv[1:]
    test_dirs = get_test_dirs()
    if "--verify-no-missing-tests" in args:
        with io.StringIO() as text_writer:
            write_cmake_lists_data_to_file(test_dirs, text_writer)
            expected_file_contents = text_writer.getvalue().strip()
        with open(GENERATED_TESTS_LIST_CMAKE_FILE_PATH, "r", encoding="utf-8") as in_f:
            actual_file_contents = in_f.read().strip()

        if expected_file_contents != actual_file_contents:
            raise RuntimeError("File contents do not match")

        print("test file has no missing tests!")
    else:
        with open(GENERATED_TESTS_LIST_CMAKE_FILE_PATH, "w", encoding="utf-8") as out_f:
            write_cmake_lists_data_to_file(test_dirs, out_f)


if __name__ == "__main__":
    main()
