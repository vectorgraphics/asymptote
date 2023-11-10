#!/usr/bin/env python3
import io
from argparse import ArgumentParser
from typing import List, Optional
import subprocess as sp
import sys
import tempfile
import json


def execute_and_report_err(args: List[str], error_heading="Error"):
    try:
        return sp.run(
            args, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True, check=True
        )
    except sp.CalledProcessError as e:
        sys.stderr.write(f"{error_heading}\n")
        sys.stderr.write(e.stderr)
        sys.stderr.write(f"stdout:\n{e.stdout}")
        sys.stderr.write("\n")
        sys.stderr.flush()
        raise


def parse_args():
    args_parser = ArgumentParser()
    args_parser.add_argument(
        "--out-i-file", type=str, required=True, help="Output for preprocessed file"
    )

    args_parser.add_argument(
        "--out-dep-file", type=str, required=True, help="Location of output depfile"
    )

    args_parser.add_argument(
        "--in-src-file",
        type=str,
        required=True,
        help="Location of source file to process",
    )

    args_parser.add_argument(
        "--cxx-compiler", type=str, required=True, help="C++ compiler to use"
    )

    args_parser.add_argument("--msvc", action="store_true")

    args_parser.add_argument(
        "--include-dirs", type=str, help="Include directories separated by semicolon"
    )

    args_parser.add_argument(
        "--macro-defs",
        type=str,
        help="Macro definitions in the form VALA=CONTENTA or VALB",
    )

    args_parser.add_argument(
        "--cxx-standard",
        type=str,
    )
    return args_parser.parse_args()


class CompileOptions:
    def __init__(
        self,
        compiler: str,
        include_dirs: Optional[List[str]] = None,
        macros: Optional[List[str]] = None,
        extra_flags: Optional[List[str]] = None,
        standard: Optional[str] = None,
    ):
        self._compiler = compiler
        self._include_dirs = include_dirs or []
        self._macros = macros or []
        self._extra_flags = extra_flags or []
        self._standard = standard or "17"

    @property
    def compiler(self):
        return self._compiler

    def build_args_for_gcc(
        self,
        src_file: str,
        out_file: Optional[str],
        addr_flags: Optional[List[str]] = None,
    ):
        base_args = (
            [f"-I{include_dir}" for include_dir in self._include_dirs]
            + [f"-D{macro}" for macro in self._macros]
            + [f"-std=c++{self._standard}"]
            + self._extra_flags
        )

        if addr_flags:
            base_args.extend(addr_flags)

        if out_file:
            base_args.extend(["-o", out_file])

        base_args.append(src_file)
        return base_args

    def build_args_for_msvc(
        self,
        src_file: str,
        out_file: Optional[str],
        addr_flags: Optional[List[str]] = None,
    ):
        base_args = (
            [f"/I{include_dir}" for include_dir in self._include_dirs]
            + [f"/D{macro}" for macro in self._macros]
            + [f"/std:c++{self._standard}", "/Zc:__cplusplus"]
            + self._extra_flags
        )
        if addr_flags:
            base_args.extend(addr_flags)

        if out_file:
            base_args.extend(["/F", out_file])
        base_args.append(src_file)
        return base_args


def compile_for_depfile_gcc(
    compile_opt: CompileOptions, src_in: str, src_out: str, depfile_out: str
):
    args = [compile_opt.compiler] + compile_opt.build_args_for_gcc(
        src_in,
        None,
        ["-DDEPEND", "-DNOSYM", "-M", "-MG", "-O0", "-MT", src_out, "-MF", depfile_out],
    )
    try:
        sp.run(args, check=True, stdout=sp.DEVNULL, stderr=sp.STDOUT, text=True)
    except sp.CalledProcessError as e:
        sys.stderr.write('Process stderr:\n')
        sys.stderr.write(e.stderr)
        sys.stderr.write('Process stderr:\n')
        sys.stderr.write(e.stdout)
        raise


def compile_for_preproc_gcc(compile_opt: CompileOptions, src_in: str, preproc_out: str):
    args = [compile_opt.compiler] + compile_opt.build_args_for_gcc(
        src_in, preproc_out, ["-E", "-DNOSYM"]
    )
    try:
        sp.run(args, check=True, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    except sp.CalledProcessError as e:
        sys.stderr.write('Process stderr:\n')
        sys.stderr.write(e.stderr)
        sys.stderr.write('Process stderr:\n')
        sys.stderr.write(e.stdout)
        raise


def escape_windows_path(raw_path: str) -> str:
    escape_chars = {" ", "$", "#"}
    with io.StringIO() as ret_str_io:
        for char in raw_path:
            if char in escape_chars:
                ret_str_io.write("\\")
            ret_str_io.write(char)
        return ret_str_io.getvalue()


def compile_for_preproc_and_depfile_msvc(
    compile_opt: CompileOptions, src_in: str, preproc_out: str, depfile_out: str
):
    with tempfile.TemporaryDirectory() as td:
        args = [compile_opt.compiler] + compile_opt.build_args_for_msvc(
            src_in,
            None,
            [
                "/DNOSYM",
                "/DDEPEND",
                "/P",
                f"/Fi{preproc_out}",
                "/sourceDependencies",
                f"{td}/srcdep.json",
            ],
        )
        execute_and_report_err(args, "MSVC Error")
        with open(f"{td}/srcdep.json", "r", encoding="utf-8") as fread:
            dep_data = json.load(fread)

    include_fil_str = " ".join(
        escape_windows_path(include_fil)
        for include_fil in dep_data["Data"].get("Includes", [])
    )

    with open(depfile_out, "w", encoding="utf-8") as depfile_writer:
        depfile_writer.write(escape_windows_path(preproc_out))
        depfile_writer.write(": ")
        depfile_writer.write(include_fil_str)


def main():
    args = parse_args()
    opt = CompileOptions(
        args.cxx_compiler,
        args.include_dirs.split(";") if args.include_dirs else None,
        args.macro_defs.split(";") if args.macro_defs else None,
        standard=args.cxx_standard,
    )

    if args.msvc:
        compile_for_preproc_and_depfile_msvc(
            opt, args.in_src_file, args.out_i_file, args.out_dep_file
        )
    else:
        compile_for_depfile_gcc(
            opt, args.in_src_file, args.out_i_file, args.out_dep_file
        )
        compile_for_preproc_gcc(opt, args.in_src_file, args.out_i_file)


if __name__ == "__main__":
    main()
