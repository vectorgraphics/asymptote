#!/usr/bin/env python3

import argparse
import pathlib
import shutil
import subprocess

ASYMPTOTE_SOURCE_ROOT = pathlib.Path(__file__).parent.parent
GUI_DIR = gui_dir = ASYMPTOTE_SOURCE_ROOT / "GUI"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--makensis-exec",
        type=str,
        required=True,
        help="Executable file to makensis.exe",
    )
    parser.add_argument(
        "--cmake-install-root",
        type=str,
        required=True,
        help="CMake Pre-NSIS install root",
    )
    parser.add_argument(
        "--asy-install-build-dir",
        type=str,
        required=True,
        help=(
            "Name of the install directory for asymptote build contained in the "
            + "cmake-install-root directory"
        ),
    )
    return parser.parse_args()


def check_gui_built():
    dirs_to_check = {"xasyicons", "xasyqtui", "xasyversion"}

    message = (
        "GUI is not fully built. "
        + "Please ensure GUI is built before running this build script."
    )
    if not all((GUI_DIR / dir_to_check).exists() for dir_to_check in dirs_to_check):
        raise RuntimeError(message)


def copy_gui_files(asy_install_root: pathlib.Path):
    gui_install_dir = asy_install_root / "GUI"
    if gui_install_dir.is_file():
        gui_install_dir.unlink(missing_ok=True)
    elif gui_install_dir.is_dir():
        shutil.rmtree(gui_install_dir)

    gui_install_dir.mkdir(exist_ok=True)

    exclude_prefixes = {
        ".vscode",
        ".fleet",
        ".idea",
        "__pycache__",
        ".python-version",
        ".gitignore",
        "buildtool.py",
        "requirements.",
        "setup.py",
        "xasy-launcher",
    }

    for file in GUI_DIR.iterdir():
        if any(
            file.name.lower().startswith(exclude_prefix.lower())
            for exclude_prefix in exclude_prefixes
        ):
            continue
        if file.is_dir():
            shutil.copytree(file, gui_install_dir / file.name)
        else:
            shutil.copy2(file, gui_install_dir / file.name)


def main():
    # check GUI built
    args = parse_args()
    check_gui_built()

    makensis_exec = pathlib.Path(args.makensis_exec)
    if not makensis_exec.is_file():
        raise RuntimeError("makensis executable cannot be found")

    # copy GUI to
    cmake_install_root = pathlib.Path(args.cmake_install_root)
    asy_install_root = cmake_install_root / args.asy_install_build_dir
    copy_gui_files(asy_install_root)

    # generate uninstall file
    with open(
        cmake_install_root / "AsymptoteUninstallList.nsi", "w", encoding="utf-8"
    ) as f:
        for file in asy_install_root.iterdir():
            if file.is_dir():
                f.write("RMDir /r $INSTDIR\\" + file.name)
            else:
                f.write("Delete $INSTDIR\\" + file.name)
            f.write("\n")

    # call nsis builder
    subprocess.run(
        [args.makensis_exec, str(cmake_install_root / "asymptote.nsi")], check=True
    )
    print("Build succeeded")


if __name__ == "__main__":
    main()
