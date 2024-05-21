#!/usr/bin/env python3
import pathlib
import re
import subprocess
import shutil

import click
from PyQt5.uic import compileUiDir

BUILD_ROOT_DIRECTORY = pathlib.Path(__file__).parent

XASY_ICONS_MODULE_NAME = "xasyicons"

PY_UI_FILE_DIR = BUILD_ROOT_DIRECTORY / "xasyqtui"
PY_ICONS_FILE_DIR = BUILD_ROOT_DIRECTORY / XASY_ICONS_MODULE_NAME
PY_VERSION_MODULE_DIR = BUILD_ROOT_DIRECTORY / "xasyversion"


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


CMAKE_VERSION_REGEX = re.compile(r"(?:set|SET)\(\s*ASY_VERSION_BASE\s+([0-9.]+)\s*\)")


def determineAsyVersion() -> str:
    with open(
        BUILD_ROOT_DIRECTORY.parent / "pkg-info.cmake", "r", encoding="utf-8"
    ) as f:
        for line in f:
            if match_obj := CMAKE_VERSION_REGEX.match(line):
                return match_obj.group(1) + ".0"
    return "0.0.0-SNAPSHOT"


@click.command()
def buildVersionModule():
    PY_VERSION_MODULE_DIR.mkdir(exist_ok=True)
    make_init_py_at_dir(PY_VERSION_MODULE_DIR)
    version = determineAsyVersion()
    with open(PY_VERSION_MODULE_DIR / "version.py", "w", encoding="utf-8") as f:
        f.write(f'VERSION="{version}"\n')


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
def build(ctx: click.Context):
    ctx.invoke(buildUi)
    ctx.invoke(buildIcons)
    ctx.invoke(buildVersionModule)


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
