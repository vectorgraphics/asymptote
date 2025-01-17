#!/usr/bin/env python3
import argparse
import pathlib
import sys
import subprocess
import shutil
from typing import Optional

from PyQt5.uic import compileUiDir
import os

BUILD_ROOT_DIRECTORY = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(BUILD_ROOT_DIRECTORY.parent))

import determine_pkg_info

XASY_ICONS_MODULE_NAME = "xasyicons"

PY_UI_FILE_DIR = BUILD_ROOT_DIRECTORY / "xasyqtui"
PY_ICONS_FILE_DIR = BUILD_ROOT_DIRECTORY / XASY_ICONS_MODULE_NAME
PY_VERSION_MODULE_DIR = BUILD_ROOT_DIRECTORY / "xasyversion"


def _map_ui_file(_: str, fileName: str):
    return str(PY_UI_FILE_DIR), fileName


def make_init_py_at_dir(dir_name: pathlib.Path):
    (dir_name / "__init__.py").touch(exist_ok=True)


def build_ui():
    compileUiDir(
        "windows",
        map=_map_ui_file,
        from_imports=True,
        import_from=XASY_ICONS_MODULE_NAME,
    )
    make_init_py_at_dir(PY_UI_FILE_DIR)


def build_icons():
    PY_ICONS_FILE_DIR.mkdir(exist_ok=True)
    make_init_py_at_dir(PY_ICONS_FILE_DIR)
    subprocess.run(
        [
            "pyrcc5",
            str(BUILD_ROOT_DIRECTORY / "res" / "icons.qrc"),
            "-o",
            str(PY_ICONS_FILE_DIR / "icons_rc.py"),
        ]
    )


def determine_asy_version() -> str:
    version_base = determine_pkg_info.determine_asy_pkg_info(
        BUILD_ROOT_DIRECTORY.parent / "configure.ac"
    ).get("version-base")
    if not version_base:
        return "0.0.0-SNAPSHOT"
    return version_base


def build_verison_module(version_override: Optional[str] = None):
    PY_VERSION_MODULE_DIR.mkdir(exist_ok=True)
    make_init_py_at_dir(PY_VERSION_MODULE_DIR)
    if version_override is not None:
        version = version_override
    else:
        version = determine_asy_version()
    with open(PY_VERSION_MODULE_DIR / "version.py", "w", encoding="utf-8") as f:
        f.write(f'VERSION="{version}"\n')


def clean():
    if PY_UI_FILE_DIR.exists():
        shutil.rmtree(PY_UI_FILE_DIR)

    if PY_ICONS_FILE_DIR.exists():
        shutil.rmtree(PY_ICONS_FILE_DIR)

    if PY_VERSION_MODULE_DIR.exists():
        shutil.rmtree(PY_VERSION_MODULE_DIR)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="subcommands", dest="subcommand")
    version_parser = subparsers.add_parser(
        "buildversionmodule", help="build version module"
    )
    build_parser = subparsers.add_parser("build", help="build command")
    for subparser in [build_parser, version_parser]:
        subparser.add_argument("--version-override", required=False, type=str)

    subparsers.add_parser("clean", help="clean command")
    subparsers.add_parser("buildicons", help="build icons")
    subparsers.add_parser("buildui", help="build ui files")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.subcommand == "buildui":
        build_ui()
    elif args.subcommand == "buildicons":
        build_icons()
    elif args.subcommand == "buildversionmodule":
        build_verison_module(args.version_override)
    elif args.subcommand == "build":
        build_ui()
        build_icons()
        build_verison_module(args.version_override)
    elif args.subcommand == "clean":
        clean()
    else:
        raise RuntimeError("Unknown subcommand")


if __name__ == "__main__":
    main()
