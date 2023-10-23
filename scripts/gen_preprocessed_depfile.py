#!/usr/bin/env python3
from argparse import ArgumentParser
from typing import List, Optional
import subprocess as sp


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
    return args_parser.parse_args()


class CompileOptions:
    def __init__(
        self,
        compiler: str,
        include_dirs: Optional[List[str]] = None,
        macros: Optional[List[str]] = None,
        extra_flags: Optional[List[str]] = None,
    ):
        self._compiler = compiler
        self._include_dirs = include_dirs or []
        self._macros = macros or []
        self._extra_flags = extra_flags or []

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
            + self._extra_flags
        )

        if addr_flags:
            base_args.extend(addr_flags)

        if out_file:
            base_args.extend(["-o", out_file])

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
    sp.run(args, check=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)


def compile_for_preproc_gcc(compile_opt: CompileOptions, src_in: str, preproc_out: str):
    args = [compile_opt.compiler] + compile_opt.build_args_for_gcc(
        src_in, preproc_out, ["-E", "-DNOSYM"]
    )
    sp.run(args, check=True, stdout=sp.DEVNULL, stderr=sp.STDOUT)


def main():
    args = parse_args()
    opt = CompileOptions(
        args.cxx_compiler,
        args.include_dirs.split(";") if args.include_dirs else None,
        args.macro_defs.split(";") if args.macro_defs else None,
    )

    if args.msvc:
        raise NotImplementedError("Implement depfile + preprocess for msvc")
    else:
        compile_for_depfile_gcc(
            opt, args.in_src_file, args.out_i_file, args.out_dep_file
        )
        compile_for_preproc_gcc(opt, args.in_src_file, args.out_i_file)


if __name__ == "__main__":
    main()
