#!/usr/bin/env python3

import argparse
import contextlib
import os
import pathlib
import sys
import shutil
import subprocess as sp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=str, default=".")
    parser.add_argument("--latexusage-name", type=str, default="latexusage")
    parser.add_argument("--latexusage-source-dir", type=str, required=True)
    parser.add_argument("--pdflatex-executable", type=str, default="pdflatex")
    parser.add_argument("--asy-executable", type=str, default="asy")
    parser.add_argument("--asy-base-dir", type=str, required=True)
    return parser.parse_args()


def print_called_process_error(e: sp.CalledProcessError):
    if e.stderr is not None:
        sys.stderr.write("Process stderr:\n")
        sys.stderr.write(e.stderr)

    if e.stdout is not None:
        sys.stderr.write("Process output:\n")
        sys.stderr.write(e.stdout)
    sys.stderr.flush()


def clean_artifacts(buildroot_path: pathlib.Path, latexusage_file_prefix: str):
    if (buildroot_path / (latexusage_file_prefix + ".tex")).is_file():
        os.remove((buildroot_path / (latexusage_file_prefix + ".tex")))
    for asyartifacts in buildroot_path.glob("latexusage-*"):
        os.remove(asyartifacts)
    for exts in ["pre", "aux", "out"]:
        with contextlib.suppress(FileNotFoundError):
            os.remove(buildroot_path / (latexusage_file_prefix + "." + exts))


def run_pdflatex(
    pdflatex_exec: str, buildroot_path: pathlib.Path, latexusage_name: str
):
    sp.run(
        [pdflatex_exec, latexusage_name + ".tex"],
        text=True,
        cwd=buildroot_path,
        check=True,
    )


def main():
    args = parse_args()
    buildroot_path = pathlib.Path(args.build_dir)
    clean_artifacts(buildroot_path, args.latexusage_name)

    # copy latexusage.pdf to build root, since TeX Live has some issues with
    # out of source builds

    shutil.copy2(
        pathlib.Path(args.latexusage_source_dir) / (args.latexusage_name + ".tex"),
        buildroot_path / (args.latexusage_name + ".tex"),
    )

    asy_base_dir = pathlib.Path(args.asy_base_dir)
    asy_base_args = [
        args.asy_executable,
        "-dir",
        str(asy_base_dir),
        "-noprc",
        "-config",
        '""',
        "-render=0",
        "-noV",
    ]

    try:
        # first pdflatex run
        run_pdflatex(args.pdflatex_executable, buildroot_path, args.latexusage_name)
        # asy run
        for asyfile in buildroot_path.glob("latexusage-*.asy"):
            sp.run(
                asy_base_args + [str(asyfile.name)],
                check=True,
                text=True,
                cwd=str(buildroot_path),
            )

        # second pdflatex run
        run_pdflatex(args.pdflatex_executable, buildroot_path, args.latexusage_name)
    except sp.CalledProcessError as e:
        print_called_process_error(e)
        raise
    finally:
        # clean up any latexusage-* files
        clean_artifacts(buildroot_path, args.latexusage_name)


if __name__ == "__main__":
    main()
