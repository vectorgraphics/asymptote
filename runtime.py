import argparse
import io
import os
import re
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opsym-file", dest="opsymbolsFile", required=True)
    parser.add_argument(
        "--runtime-base-file", dest="runtimeBaseFile", required=True
    )
    parser.add_argument(
        "--src-template-dir", dest="srcTemplateDir", required=True
    )
    parser.add_argument("--prefix", dest="prefix", required=True)
    parser.add_argument("--header-out-dir", dest="headerOutDir", required=True)
    parser.add_argument("--src-out-dir", dest="srcOutDir", required=True)
    args = parser.parse_args()

    opsymbolsFile = args.opsymbolsFile
    runtimeBaseFile = args.runtimeBaseFile
    srcTemplateDir = args.srcTemplateDir
    prefix = args.prefix
    headerOutDir = args.headerOutDir
    srcOutDir = args.srcOutDir

    outHeaderFile = os.path.join(headerOutDir, f"{prefix}.h")
    outSrcFile = os.path.join(srcOutDir, f"{prefix}.cc")

    stack = "Stack"
    errors = 0

    def report_error(filename, line, error):
        nonlocal errors
        sys.stderr.write(f"{filename}:{line}: {error}\n")
        errors = 1

    def assoc_error(filename, line, t):
        report_error(filename, line, f"no asy type associated to '{t}'")

    def clean_type(v):
        return re.sub(r"\s+", "", v)

    def clean_params(v: str) -> str:
        return v.replace("\n", "")

    type_map = {}

    def read_types(data, filename, start_line):
        nonlocal type_map
        lines = data.split("\n")
        line_index = start_line
        for l in lines:
            line_index += 1
            # Remove // comments in each line
            l = re.sub(r"//.*", "", l)

            # Skip empty lines
            if re.fullmatch(r"\s*", l):
                continue

            # Regex for matching something like "item => ITEMVAL"
            # (\w*(?:\s*\*)?) matches a type or pointer (e.g. "int", "item*", etc.)
            # \s*=>\s* captures the '=>' then
            # (.*) grabs the rest of the line as the code
            match = re.search(
                r"(\w*(?:\s*\*)?)"  # Type or pointer type
                r"\s*=>\s*"  # =>
                r"(.*)",  # Everything else
                l,
            )
            if not match:
                report_error(filename, line_index, "bad type declaration")
                continue
            t, code = match.groups()
            t = clean_type(t)
            type_map[t] = code

    opsymbols = {}
    with open(opsymbolsFile, "r", encoding="utf-8") as opf:
        for line in opf:
            # match OPSYMBOL("symname", token);
            m = re.search(r"^OPSYMBOL\(\"(.*)\", ([A-Za-z_]+)\);", line)
            if m:
                symname, token = m.groups()
                opsymbols[symname] = token

    def symbolize(name):
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

    def asy_params(params_string, filename, line):
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
                assoc_error(filename, line, t)
            has_def = "true" if d else "false"
            is_ex = "true" if explicit else "false"
            result.append(
                f"formal({type_map[t]}, {symbolize(n.lower())}, {has_def}, {is_ex})"
            )
        return result

    # Convert parameters into stack pop code in correct order
    def c_params(params_list):
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
            r = f"  {type_str} {name}=vm::pop{template}({stack}{def_val});\n"
            result.append(r)
        return result

    # Read base, prefix.in, etc., all separated by form-feed + newline
    with open(os.path.join(srcTemplateDir, f"{prefix}.in"), "rb") as f_in:
        # Convert to text with universal newline
        data_in = io.TextIOWrapper(f_in, encoding="utf-8", newline=None).read()
    with open(runtimeBaseFile, "rb") as f_base:
        # Convert to text with universal newline
        data_base = io.TextIOWrapper(
            f_base, encoding="utf-8", newline=None
        ).read()
    with open(outSrcFile, "w", newline="", encoding="utf-8") as f_out:

        out_autogen = (
            f"/***** Autogenerated from {prefix}.in; "
            "changes will be overwritten *****/\n\n"
        )
        # Output an autogenerated banner
        f_out.write(out_autogen)

        # We split and merge corresponding sections delineated by form-feed
        sections_base = data_base.split("\f\n")
        sections_base[:-1] = [section + "\f\n" for section in sections_base[:-1]]
        sections_in = data_in.split("\f\n")
        sections_in[:-1] = [section + "\f\n" for section in sections_in[:-1]]

        # Attempt to replicate the same line counting logic
        # Just track how many lines have passed
        base_line_counter = 1
        in_line_counter = 1

        # 1) runtimebase.in chunk #1
        f_out.write(
            f'#line {base_line_counter} "{srcTemplateDir}/runtimebase.in"\n'
        )
        baseheader = sections_base[0]
        f_out.write(baseheader)
        base_line_counter += baseheader.count("\n")
        # The starting line for the type map section in runtimebase.in.
        base_start_type = base_line_counter

        # 2) prefix.in chunk #1
        f_out.write(
            f'#line {in_line_counter} "{srcTemplateDir}/{prefix}.in"\n'
        )
        header = sections_in[0]
        f_out.write(header)
        in_line_counter += header.count("\n")
        # The starting line for the type map section in prefix.in.
        in_start_type = in_line_counter

        # 3) read next chunk from runtimebase.in for type mapping
        basetypes = sections_base[1] if len(sections_base) > 1 else ""
        base_line_counter += basetypes.count("\n")
        # ... and similarly the next chunk from prefix.in
        types = sections_in[1] if len(sections_in) > 1 else ""
        in_line_counter += types.count("\n")

        # 4) next chunk from base
        f_out.write(
            f'#line {base_line_counter} "{srcTemplateDir}/runtimebase.in"\n'
        )
        baseheader2 = sections_base[2] if len(sections_base) > 2 else ""
        f_out.write(baseheader2)
        base_line_counter += baseheader2.count("\n")

        # 5) next chunk from prefix.in
        f_out.write(f'#line {in_line_counter} "{prefix}.in"\n')
        header2 = sections_in[2] if len(sections_in) > 2 else ""
        f_out.write(header2)
        in_line_counter += header2.count("\n")

        f_out.write("\n#ifndef NOSYM\n")
        f_out.write(f'#include "{prefix}.symbols.h"\n')
        f_out.write("\n#endif\n")
        f_out.write("namespace run {\n")

        # Build type_map
        read_types(basetypes, "runtimebase.in", base_start_type)
        read_types(types, f"{prefix}.in", in_start_type)

        # Prepare header array
        header_lines = []
        header_lines.append(out_autogen)
        header_lines.append("#pragma once\n")
        header_lines.append("namespace run {\n")

        builtin: list[str] = []

        # 6) read remaining lines from sections_in[3..] for function definitions
        for function_count, section in enumerate(sections_in[3:]):
            # This regex attempts to parse a block of function definitions in the form:
            # optional comments, return type, asy function name,
            # optional C++ function name, parameters, code
            pat = re.compile(
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
                r"\{(.*)\}",  # $6: function body up to last closing brace in record
                re.DOTALL,
            )
            next_in_line_counter = in_line_counter + section.count("\n")
            md = pat.search(section)
            if not md:
                report_error(
                    f"{prefix}.in", in_line_counter, "bad function definition"
                )
                continue
            comments, type_, name, cname, params, code = md.groups()

            # Insert a fallback cName if needed
            if cname:
                header_lines.append(f"void {cname}(vm::stack *);\n")
            else:
                cname = f"gen_{prefix}{function_count}"
                # Added newlines here would mess up the line count
                assert cname.count("\n") == 0

            # Clean up types
            type_ = clean_type(type_)
            # Split the parameter list by commas (ignoring whitespace after each comma)
            param_list = re.split(r",\s*", params) if params else []

            # Build addFunc part
            if name:
                name = re.sub(
                    r"Operator\s*", "", name
                )  # If there's "Operator", remove it
                if type_ not in type_map:
                    assoc_error(f"{prefix}.in", in_line_counter, type_)
                asy_param_list = asy_params(
                    params, f"{prefix}.in", in_line_counter
                )
                asy_params_comma = ""
                if asy_param_list:
                    joined = ", ".join(asy_param_list)
                    asy_params_comma = f", {joined}"
                builtin.append(
                    f'#line {in_line_counter} "{srcTemplateDir}/{prefix}.in"\n'
                    f"  addFunc(ve, run::{cname}, {type_map[type_]}, "
                    f"{symbolize(name)}{asy_params_comma});\n"
                )
            else:
                # Note: earlier code ensures cname is not empty.
                # builtin with no name => REGISTER_BLTIN
                builtin.append(
                    f'#line {in_line_counter} "{srcTemplateDir}/{prefix}.in"\n'
                    f'  REGISTER_BLTIN(run::{cname},"{cname}");\n'
                )

            # Replace 'return X;' with push
            qualifier = "" if type_ == "item" else f"<{type_}>"
            code_transformed = re.sub(
                r"\breturn\s+([^;]*);",  # read until the next semicolon
                rf"{{{stack}->push{qualifier}(\1); return;}}",
                code,
            )

            # Build param popping lines
            param_code = "".join(c_params(param_list))

            # Write out any preceding comments
            f_out.write(comments)
            in_line_counter += comments.count("\n")
            f_out.write(
                f'#line {in_line_counter} "{srcTemplateDir}/{prefix}.in"\n'
            )

            # Write out the function prototype as a comment
            prototype = f"{type_} {name}({params});"
            in_line_counter += prototype.count("\n") + 1
            if name:
                prototype = clean_params(prototype)
                assert prototype.count("\n") == 0
                f_out.write(f"// {prototype}\n")

            # Actual function definition in prefix.cc
            param_name = stack if type_ != "void" or param_list else ""
            f_out.write(f"void {cname}(stack *{param_name})\n")
            f_out.write("{\n")
            assert not param_code or param_code[-1] == "\n"
            f_out.write(param_code)
            f_out.write(
                f'#line {in_line_counter} "{srcTemplateDir}/{prefix}.in"'
            )
            assert code_transformed[0] == "\n"
            f_out.write(code_transformed)
            f_out.write("}\n\n")

            in_line_counter = next_in_line_counter

        f_out.write("} // namespace run\n\n")

        f_out.write("namespace trans {\n\n")
        f_out.write(f"void gen_{prefix}_venv(venv &ve)\n")
        f_out.write("{\n")
        f_out.write("".join(builtin))
        f_out.write("}\n\n")
        f_out.write("} // namespace trans\n")

    # Finalize, compare header
    header_lines.append("}\n\n")
    new_header = "".join(header_lines)
    old_header = ""
    if os.path.exists(outHeaderFile):
        with open(outHeaderFile, "r", encoding="utf-8") as hf:
            old_header = hf.read()
    if new_header != old_header:
        with open(outHeaderFile, "w", encoding="utf-8") as hf:
            hf.write(new_header)

    if errors:
        try:
            os.unlink(outHeaderFile)
        except FileNotFoundError:
            pass
        try:
            os.unlink(outSrcFile)
        except FileNotFoundError:
            pass
        sys.exit(errors)


if __name__ == "__main__":
    main()
