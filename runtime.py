#!/usr/bin/env python3

import argparse
import dataclasses
import io
import os
import re
import sys
from typing import List, Optional, TextIO, Dict, Tuple

STACK = "Stack"


class Er:
    errors = 0

    @classmethod
    def report_error(cls, filename, line, error):
        sys.stderr.write(f"{filename}:{line}: {error}\n")
        cls.errors += 1
        if cls.errors > 10:
            raise SystemExit("Too many errors, aborting")

    @classmethod
    def assoc_error(cls, filename, line, t):
        cls.report_error(filename, line, f"no asy type associated to '{t}'")


# Convert parameters into stack pop code in correct order
def c_params(params_list):
    global STACK
    result = []
    # Reverse order for correct push/pop
    for p in reversed(params_list):
        match = re.search(
            r"^\s*"  # optional whitespace
            r"(explicit)*"  # optional explicit
            r"\s*"
            r"(\w*(?:\s*\*)?)"  # type
            r"\s*"
            r"(\w*)"  # parameter name
            r"(=*)([\w.+\-]*)",  # optional default value
            p,
        )
        if not match:
            continue
        _, type_str, name, eqsign, val = match.groups()
        template = "" if type_str == "item" else f"<{type_str}>"
        def_val = f",{val}" if eqsign else ""
        r = f"  {type_str} {name}=vm::pop{template}({STACK}{def_val});\n"
        result.append(r)
    return result


def clean_type(v):
    return re.sub(r"\s+", "", v)


def clean_params(v: str) -> str:
    return v.replace("\n", "")


def symbolize(name, opsymbols):
    # If name is alphanumeric/underscore, just SYM(...) it
    if re.search(r"^[A-Za-z0-9_]+$", name):
        return f"SYM({name})"
    # If the name is recognized in opsymbols, substitute
    if name in opsymbols:
        return opsymbols[name]
    # Check if it is "operator smth" form
    opm = re.search(r"operator\s+(\w+)", name)
    if opm and opm.group(1) in opsymbols:
        return opsymbols[opm.group(1)]
    # Otherwise, generate a trans(...) symbol
    return f'symbol::trans("{name}")'


def asy_params(params_string, filename, line, type_map, opsymbols):
    # Split parameters by comma
    params_list = re.split(r",\s*", params_string) if params_string else []
    result = []
    for p in params_list:
        # (explicit)*\s*(\w*(?:\s*\*)?) => optional 'explicit', type, pointer
        # (\w*) => parameter name, possibly something like "x"
        # (=*) => optional equals sign for default parameter
        match = re.search(
            r"^\s*(explicit)*\s*"  # optional explicit
            r"(\w*(?:\s*\*)?)\s*"  # type or pointer type
            r"(\w*)"  # name
            r"(=*)",  # if '=' is present, there is a default value
            p,
        )
        if not match:
            continue
        explicit, t, n, d = match.groups()
        t = clean_type(t)
        if t not in type_map:
            Er.assoc_error(filename, line, t)
        has_def = "true" if d else "false"
        is_ex = "true" if explicit else "false"
        result.append(
            "formal("
            f"{type_map[t]}, {symbolize(n.lower(), opsymbols)}, {has_def}, {is_ex})"
        )
    return result


@dataclasses.dataclass()
class RunData:
    runtimeBaseFile: str
    srcTemplateDir: str
    prefix: str
    headerOutDir: str
    srcOutDir: str
    type_map: Dict = dataclasses.field(default_factory=dict)  # type: Dict[str, str]
    op_symbols: Dict = dataclasses.field(default_factory=dict)  # type: Dict[str, str]


def read_opsymbols(opsymbols_file: str) -> Dict:
    op_symbols: Dict = {}
    with open(opsymbols_file, "r", encoding="utf-8") as opf:
        for line in opf:
            # match OPSYMBOL("symname", token);
            m = re.search(r"^OPSYMBOL\(\"(.*)\", ([A-Za-z_]+)\);", line)
            if m:
                symname, token = m.groups()
                op_symbols[symname] = token
    return op_symbols


def parse_args() -> RunData:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opsym-file", dest="opsymbolsFile", required=True)
    parser.add_argument("--runtime-base-file", dest="runtimeBaseFile", required=True)
    parser.add_argument("--src-template-dir", dest="srcTemplateDir", required=True)
    parser.add_argument("--prefix", dest="prefix", required=True)
    parser.add_argument("--header-out-dir", dest="headerOutDir", required=True)
    parser.add_argument("--src-out-dir", dest="srcOutDir", required=True)
    args = parser.parse_args()
    return RunData(
        runtimeBaseFile=args.runtimeBaseFile,
        srcTemplateDir=args.srcTemplateDir,
        prefix=args.prefix,
        headerOutDir=args.headerOutDir,
        srcOutDir=args.srcOutDir,
        op_symbols=read_opsymbols(args.opsymbolsFile),
    )


def parse_types(
    data: str, input_filename: str, input_start_line: int
) -> Tuple[Dict, int]:
    """Return (new_entries, lines_consumed)."""
    entries: Dict = {}
    lines = data.split("\n")
    for line_no, line in enumerate(lines, start=input_start_line):
        # Remove // comments in each line
        trimmed = re.sub(r"//.*", "", line)

        # Skip empty lines
        if re.fullmatch(r"\s*", trimmed):
            continue
        # Regex for matching something like "item => ITEMVAL"
        # (\w*(?:\s*\*)?) matches a type or pointer (e.g. "int", "item*", etc.)
        # \s*=>\s* captures the '=>' then
        # (.*) grabs the rest of the line as the code
        match = re.search(
            r"(\w*(?:\s*\*)?)"  # Type or pointer type
            r"\s*=>\s*"  # =>
            r"(.*)",  # Everything else
            trimmed,
        )
        if not match:
            Er.report_error(input_filename, line_no, "bad type declaration")
            continue
        t, code = match.groups()
        entries[clean_type(t)] = code
    return entries, data.count("\n")


class MatchNotFoundError(Exception):
    pass


class FunctionData:
    # This regex attempts to parse a function definition in the form:
    # optional comments, return type, asy function name,
    # optional C++ function name, parameters, code
    _pat = re.compile(
        # pylint: disable=line-too-long
        r"^((?:\s*//[^\n]*\n)*)"  # $1: capture comment lines starting with //
        r"\s*"
        r"(\w*(?:\s*\*)?)"  # $2: return type (e.g., 'int', 'item*', etc.)
        r"\s*"
        r"([^(:]*)"  # $3: read to the first colon or open parenthesis (asy function name)
        r"\:*"
        r"([^(]*)"  # $4: read to the first open parenthesis (optional c++ function name)
        r"\s*"
        r"\(([\w\s*,=.+\-]*)\)"  # $5: parameters list inside parentheses
        r"\s*"
        r"\{(.*)\}",  # $6: function body up to last closing brace in section
        re.DOTALL,  # allow . to match newlines (for $6)
    )

    def __init__(
        self,
        section: str,
        prefix: str,
        function_count: int,
        header_lines: List,
    ):
        global STACK
        md = self._pat.search(section)
        if not md:
            raise MatchNotFoundError
        comments, return_type, name, cname, params, code = md.groups()
        # Insert a fallback cName if needed
        if cname:
            header_lines.append(f"void {cname}(vm::stack *);\n")
        else:
            cname = f"gen_{prefix}{function_count}"
            # Added newlines here would mess up the line count
            assert cname.count("\n") == 0

        # Clean up types
        return_type = clean_type(return_type)
        # If there's "Operator", remove it:
        name = re.sub(r"Operator\s*", "", name)
        # Replace 'return X;' with push
        qualifier = "" if return_type == "item" else f"<{return_type}>"
        code = re.sub(
            r"\breturn\s+([^;]*);",  # read until the next semicolon
            rf"{{{STACK}->push{qualifier}(\1); return;}}",
            code,
        )

        self.comments: str = comments
        self.return_type: str = return_type
        self.name: str = name
        self.cname: str = cname
        self.params_string: str = params
        self.code: str = code

    def generate_addFunc(
        self,
        *,
        in_line_counter: int,
        d: RunData,
    ) -> str:
        assert self.cname
        if self.name:
            if self.return_type not in d.type_map:
                Er.assoc_error(f"{d.prefix}.in", in_line_counter, self.return_type)
            asy_param_list = asy_params(
                self.params_string,
                f"{d.prefix}.in",
                in_line_counter,
                d.type_map,
                d.op_symbols,
            )
            asy_params_comma = ""
            if asy_param_list:
                joined = ", ".join(asy_param_list)
                asy_params_comma = f", {joined}"
            return (
                f'#line {in_line_counter} "{d.srcTemplateDir}/{d.prefix}.in"\n'
                f"  addFunc(ve, run::{self.cname}, {d.type_map[self.return_type]}, "
                f"{symbolize(self.name,d.op_symbols)}{asy_params_comma});\n"
            )
        # builtin with no name => REGISTER_BLTIN
        return (
            f'#line {in_line_counter} "{d.srcTemplateDir}/{d.prefix}.in"\n'
            f'  REGISTER_BLTIN(run::{self.cname},"{self.cname}");\n'
        )

    def write_cc(self, f_out: TextIO, d: RunData, in_line_counter: int) -> None:
        global STACK
        # Write out any preceding comments
        f_out.write(self.comments)
        in_line_counter += self.comments.count("\n")
        f_out.write(f'#line {in_line_counter} "{d.srcTemplateDir}/{d.prefix}.in"\n')

        # Split the parameter list by commas (ignoring whitespace after each comma)
        param_list = re.split(r",\s*", self.params_string) if self.params_string else []
        # Build param popping lines
        param_code = "".join(c_params(param_list))

        # Write out the function prototype as a comment
        prototype = f"{self.return_type} {self.name}({self.params_string});"
        in_line_counter += prototype.count("\n") + 1
        if self.name:
            prototype = clean_params(prototype)
            assert prototype.count("\n") == 0
            f_out.write(f"// {prototype}\n")

        # Actual function definition in prefix.cc
        param_name = STACK if self.return_type != "void" or param_list else ""
        f_out.write(f"void {self.cname}(stack *{param_name})\n")
        f_out.write("{\n")
        assert not param_code or param_code[-1] == "\n"
        f_out.write(param_code)
        f_out.write(f'#line {in_line_counter} "{d.srcTemplateDir}/{d.prefix}.in"')
        assert self.code[0] == "\n"
        f_out.write(self.code)
        f_out.write("}\n\n")


@dataclasses.dataclass
class SectionWriter:
    """Write sections while keeping track of input line numbers."""

    f_out: TextIO
    srcdir: str
    prefix: str
    base_line_counter: int = 1
    in_line_counter: int = 1

    def write_base_chunk(self, chunk: str) -> None:
        directive = f'#line {self.base_line_counter} "{self.srcdir}/runtimebase.in"\n'
        self.f_out.write(directive)
        self.f_out.write(chunk)
        self.base_line_counter += chunk.count("\n")

    def write_in_chunk(self, chunk: str, directive: Optional[str] = None) -> None:
        if directive is None:
            directive = (
                f'#line {self.in_line_counter} "{self.srcdir}/{self.prefix}.in"\n'
            )
        self.f_out.write(directive)
        self.f_out.write(chunk)
        self.in_line_counter += chunk.count("\n")


def overwrite_if_changed(filename: str, new_contents: str) -> None:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            old_contents = f.read()
            if old_contents == new_contents:
                return
    except FileNotFoundError:
        pass
    with open(filename, "w", encoding="utf-8") as f:
        f.write(new_contents)


def write_trans_namespace(f_out: TextIO, d: RunData, builtin: List) -> None:
    f_out.write("namespace trans {\n\n")
    f_out.write(f"void gen_{d.prefix}_venv(venv &ve)\n")
    f_out.write("{\n")
    f_out.write("".join(builtin))
    f_out.write("}\n\n")
    f_out.write("} // namespace trans\n")


def read_sections(filename: str) -> List:
    # Read *.in files split by form-feed + newline
    with open(filename, "rb") as f:
        # Convert to text with universal newline
        data = io.TextIOWrapper(f, encoding="utf-8", newline=None).read()
        # We split and merge corresponding sections delineated by form-feed
        sections = data.split("\f\n")
        sections[:-1] = [section + "\f\n" for section in sections[:-1]]
        # For now, imitate the behavior of the original perl code: if the last section
        # is empty, remove it.
        if sections and not sections[-1]:
            sections.pop()
    return sections


def main(d: RunData) -> None:
    outHeaderFile = os.path.join(d.headerOutDir, f"{d.prefix}.h")
    outSrcFile = os.path.join(d.srcOutDir, f"{d.prefix}.cc")
    sections_in, sections_base = read_input_sections(d)
    header_lines = process_sections(d, sections_in, sections_base, outSrcFile)
    finalize_output(outHeaderFile, outSrcFile, header_lines)


def read_input_sections(d: RunData) -> Tuple[List, List]:
    # Read base, prefix.in, etc., all separated by form-feed + newline
    sections_in = read_sections(os.path.join(d.srcTemplateDir, f"{d.prefix}.in"))
    sections_base = read_sections(d.runtimeBaseFile)
    return sections_in, sections_base


def process_sections(
    d: RunData, sections_in: List, sections_base: List, outSrcFile: str
) -> List:
    with open(outSrcFile, "w", newline="", encoding="utf-8") as f_out:
        out_autogen = (
            f"/***** Autogenerated from {d.prefix}.in; "
            "changes will be overwritten *****/\n\n"
        )
        # Output an autogenerated banner
        f_out.write(out_autogen)

        # Track how many lines have passed in each file
        w = SectionWriter(f_out, d.srcTemplateDir, d.prefix)

        # 1) runtimebase.in chunk #1
        w.write_base_chunk(sections_base[0])

        # 2) prefix.in chunk #1
        w.write_in_chunk(
            sections_in[0],
        )

        # 3) read next chunk from runtimebase.in for type mapping
        current_chunk = sections_base[1] if len(sections_base) > 1 else ""
        new_types, lines_consumed = parse_types(
            current_chunk, "runtimebase.in", w.base_line_counter
        )
        d.type_map.update(new_types)
        w.base_line_counter += lines_consumed

        # ... and similarly the next chunk from prefix.in
        current_chunk = sections_in[1] if len(sections_in) > 1 else ""
        new_types, lines_consumed = parse_types(
            current_chunk, f"{d.prefix}.in", w.in_line_counter
        )
        d.type_map.update(new_types)
        w.in_line_counter += lines_consumed

        # 4) next chunk from base
        w.write_base_chunk(sections_base[2] if len(sections_base) > 2 else "")

        # 5) next chunk from prefix.in
        w.write_in_chunk(
            sections_in[2] if len(sections_in) > 2 else "",
            # pylint: disable=fixme
            # TODO: The following line preserves erroneous behavior of the original
            # perl code, but should be eliminated once the perl code is removed.
            f'#line {w.in_line_counter} "{d.prefix}.in"\n',
        )

        f_out.write("\n#ifndef NOSYM\n")
        f_out.write(f'#include "{d.prefix}.symbols.h"\n')
        f_out.write("\n#endif\n")
        f_out.write("namespace run {\n")

        header_lines = [out_autogen, "#pragma once\n", "namespace run {\n"]
        builtin: List = []

        # 6) read remaining lines from sections_in[3...] for function definitions
        for function_count, section in enumerate(sections_in[3:]):
            try:
                fd = FunctionData(section, d.prefix, function_count, header_lines)
            except MatchNotFoundError:
                Er.report_error(
                    f"{d.prefix}.in",
                    w.in_line_counter,
                    "bad function definition",
                )
                continue

            # Build addFunc part
            builtin.append(
                fd.generate_addFunc(
                    in_line_counter=w.in_line_counter,
                    d=d,
                )
            )

            fd.write_cc(f_out, d, w.in_line_counter)

            w.in_line_counter += section.count("\n")

        f_out.write("} // namespace run\n\n")

        write_trans_namespace(f_out, d, builtin)

    return header_lines


def finalize_output(
    outHeaderFile: str, outSrcFile: str, header_lines: List
) -> None:
    header_lines.append("}\n\n")
    overwrite_if_changed(outHeaderFile, "".join(header_lines))

    if Er.errors:
        try:
            os.unlink(outHeaderFile)
        except FileNotFoundError:
            pass
        try:
            os.unlink(outSrcFile)
        except FileNotFoundError:
            pass
        sys.exit(1)


if __name__ == "__main__":
    runData = parse_args()
    main(runData)
