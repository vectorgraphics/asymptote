#!/usr/bin/env python3
"""Print the python files that should be linted, one absolute path per line.

The candidate set comes from git rather than a filesystem walk: tracked files,
plus untracked files that are not ignored (so a new file is linted before it is
committed).  A working repository accumulates virtual environments, build trees
and scratch scripts, and asking git keeps those out for free provided they are
ignored -- anything not covered by .gitignore is still swept up, which is
usually a hint that it belongs there.

The candidates are then filtered through the exclusion lists below, which drop
checked-in code linted elsewhere (GUI) or not ours to lint (third-party
subtrees, generated files).
"""

import fnmatch
import os.path
import pathlib
import subprocess
import sys

# abspath rather than resolve(): callers rely on absolute output, but python
# before 3.9 leaves a relative __file__ relative, and resolving would rewrite
# symlinks the caller may have used deliberately.
REPO_ROOT = pathlib.Path(os.path.abspath(__file__)).parents[1]

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
    "VulkanMemoryAllocator",
]

EXCLUDED_FILE_GLOB_PATTERNS = [
    "base/asymptote.py",
    "misc/aspy.py",
    "misc/profile.py",
]


def git_candidate_files():
    """Return the repo-relative python files git knows or would soon know about.

    The ".py" selection is done here rather than with a git pathspec, which
    would have to survive the argument handling of whichever git is on PATH --
    on Windows a MinGW program whose runtime may expand wildcards first.
    """
    tracked = run_git("ls-files", "-z")
    untracked = run_git("ls-files", "-z", "--others", "--exclude-standard")
    return sorted(
        path for path in set(tracked) | set(untracked) if path.suffix == ".py"
    )


def run_git(*args):
    """Run a NUL-separated git command in the repository, returning its entries."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT)] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except OSError as exc:
        sys.exit(f"cannot run git: {exc}")
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode(errors="replace").strip()
        sys.exit(f"git {' '.join(args)} failed in {REPO_ROOT}: {message}")
    return [
        pathlib.PurePosixPath(name)
        for name in completed.stdout.decode().split("\0")
        if name
    ]


def is_excluded(relative_path: pathlib.PurePosixPath):
    root_folder = relative_path.parts[0]
    if any(fnmatch.fnmatch(root_folder, pattern) for pattern in EXCLUDED_ROOT_FOLDERS):
        return True
    return any(relative_path.match(pattern) for pattern in EXCLUDED_FILE_GLOB_PATTERNS)


def print_non_gui_py_files_for_linting():
    for relative_path in git_candidate_files():
        if is_excluded(relative_path):
            continue
        # Join component-wise: git reports posix separators, and this keeps the
        # result native without mixing two pathlib flavours.
        print(REPO_ROOT.joinpath(*relative_path.parts))


if __name__ == "__main__":
    print_non_gui_py_files_for_linting()
