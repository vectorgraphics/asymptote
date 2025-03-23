#!/usr/bin/env python3
import pathlib

REPO_ROOT = pathlib.Path(__file__).parents[1]

EXCLUDED_ROOT_FOLDERS = [
    "cmake-build-*",
    "cmake-install-*",
    ".git",
    ".vs",
    ".fleet",
    ".idea",
    ".vscode",
    "__pycache__",
    "GUI",
    "asydoc",
    "extfiles",
    "tools-cache",
    "vcpkg_installed",
    "LspCpp",
    "tinyexr",
]

EXCLUDED_FILE_GLOB_PATTERNS = [
    "base/asymptote.py",
    "misc/aspy.py",
    "misc/profile.py",
    "base/asymptote.pdf",
]


def print_subdirectory(subdirectory: pathlib.Path):
    for py_file in subdirectory.rglob("*.py"):
        if any(py_file.match(pattern) for pattern in EXCLUDED_FILE_GLOB_PATTERNS):
            continue
        print(py_file)


def print_non_gui_py_files_for_linting():
    for py_file in REPO_ROOT.glob("*.py"):
        if any(py_file.match(pattern) for pattern in EXCLUDED_FILE_GLOB_PATTERNS):
            continue
        print(py_file)

    for path in REPO_ROOT.iterdir():
        if not path.is_dir():
            continue
        if any(path.match(pattern) for pattern in EXCLUDED_ROOT_FOLDERS):
            continue
        print_subdirectory(path)


if __name__ == "__main__":
    print_non_gui_py_files_for_linting()
