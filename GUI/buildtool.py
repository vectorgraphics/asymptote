#!/usr/bin/env python3
import pathlib
import sys
import subprocess
import shutil
from typing import Optional

import click
from PyQt5.uic import compileUiDir
import os

BUILD_ROOT_DIRECTORY = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(BUILD_ROOT_DIRECTORY.parent))

import determine_pkg_info

XASY_ICONS_MODULE_NAME = "xasyicons"

PY_UI_FILE_DIR = BUILD_ROOT_DIRECTORY / "xasyqtui"
PY_ICONS_FILE_DIR = BUILD_ROOT_DIRECTORY / XASY_ICONS_MODULE_NAME
PY_VERSION_MODULE_DIR = BUILD_ROOT_DIRECTORY / "xasyversion"


def add_version_override_arg(cmd_fn):
    return click.option(
        "--version-override",
        default=None,
        type=str,
        help="Version to use. If not given, uses information from configure.ac.",
    )(cmd_fn)


def _mapUiFile(_: str, fileName: str):
    return str(PY_UI_FILE_DIR), fileName


def make_init_py_at_dir(dir_name: pathlib.Path):
    (dir_name / "__init__.py").touch(exist_ok=True)


@click.command()
def buildUi():
    compileUiDir(
        "windows", map=_mapUiFile, from_imports=True, import_from=XASY_ICONS_MODULE_NAME
    )
    make_init_py_at_dir(PY_UI_FILE_DIR)


@click.command()
def buildIcons():
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


def determineAsyVersion() -> str:
    version_base = determine_pkg_info.determine_asy_pkg_info(
        BUILD_ROOT_DIRECTORY.parent / "configure.ac"
    ).get("version-base")
    if not version_base:
        return "0.0.0-SNAPSHOT"
    return version_base


def buildVersionModuleInternal(version_override: Optional[str] = None):
    PY_VERSION_MODULE_DIR.mkdir(exist_ok=True)
    make_init_py_at_dir(PY_VERSION_MODULE_DIR)
    if version_override is not None:
        version = version_override
    else:
        version = determineAsyVersion()
    with open(PY_VERSION_MODULE_DIR / "version.py", "w", encoding="utf-8") as f:
        f.write(f'VERSION="{version}"\n')


@click.command()
@add_version_override_arg
def buildVersionModule(version_override: Optional[str]):
    buildVersionModuleInternal(version_override)


@click.command()
def clean():
    if PY_UI_FILE_DIR.exists():
        shutil.rmtree(PY_UI_FILE_DIR)

    if PY_ICONS_FILE_DIR.exists():
        shutil.rmtree(PY_ICONS_FILE_DIR)

    if PY_VERSION_MODULE_DIR.exists():
        shutil.rmtree(PY_VERSION_MODULE_DIR)


@click.command()
@click.pass_context
@add_version_override_arg
def build(ctx: click.Context, version_override: Optional[str] = None):
    ctx.invoke(buildUi)
    ctx.invoke(buildIcons)
    buildVersionModuleInternal(version_override)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        ctx.invoke(build)


cli.add_command(buildUi)
cli.add_command(buildIcons)
cli.add_command(buildVersionModule)
cli.add_command(build)
cli.add_command(clean)


if __name__ == "__main__":
    cli()
