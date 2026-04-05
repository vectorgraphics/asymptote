#!/usr/bin/env python3
"""
Verify that the split errortests/ files are equivalent to the original
errortest.asy and errors files.

This script has hardcoded filenames and line ranges and is intentionally
fragile.  Its only purpose is to prove that the new split tests contain
the same content as the old monolithic files.
"""

import re
import sys


def read_lines(filepath):
    with open(filepath) as f:
        return f.readlines()


def lines(all_lines, start, end):
    """Extract lines start..end (1-indexed, inclusive)."""
    return all_lines[start - 1 : end]


def strip_error_prefix(line):
    """Strip 'filename: NN.CC: ' prefix, returning just the message."""
    m = re.match(r"^[^:]+:\s+\d+\.\d+:\s+", line)
    if m:
        return line[m.end() :]
    return line


def check(label, a, b):
    if a == b:
        print(f"  OK: {label}")
        return True
    print(f"  FAIL: {label}")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"    First diff at offset {i}: {x!r} vs {y!r}")
            break
    if len(a) != len(b):
        print(f"    Length: {len(a)} vs {len(b)}")
    return False


def check_line(label, all_lines, lineno, expected):
    """Check that a single line has exact expected content."""
    actual = all_lines[lineno - 1]
    if actual == expected:
        print(f"  OK: {label}")
        return True
    print(f"  FAIL: {label}")
    print(f"    Expected: {expected!r}")
    print(f"    Actual:   {actual!r}")
    return False


def verify_full_coverage(name, covered_ranges, total_lines):
    """Verify that covered_ranges partitions 1..total_lines with no gaps or
    overlaps.  Returns True on success."""
    covered = set()
    for start, end in covered_ranges:
        span = set(range(start, end + 1))
        overlap = covered & span
        if overlap:
            print(
                f"  FAIL: {name} has overlapping coverage at lines"
                f" {sorted(overlap)[:10]}"
            )
            return False
        covered |= span
    expected = set(range(1, total_lines + 1))
    if covered == expected:
        print(
            f"  OK: All {total_lines} lines of {name} are covered"
            f" ({len(covered_ranges)} ranges, no gaps, no overlaps)."
        )
        return True
    missing = sorted(expected - covered)
    extra = sorted(covered - expected)
    if missing:
        print(f"  FAIL: {name} lines not covered: {missing}")
    if extra:
        print(f"  FAIL: {name} extra lines covered: {extra}")
    return False


def main():
    ok = True
    orig = read_lines("errortest.asy")
    errs = read_lines("errors")

    names = read_lines("errortests/names.asy")
    expressions = read_lines("errortests/expressions.asy")
    statements = read_lines("errortests/statements.asy")
    environment = read_lines("errortests/environment.asy")
    permissions = read_lines("errortests/permissions.asy")
    var = read_lines("errortests/var.asy")
    keyword_rest = read_lines("errortests/keyword_rest.asy")
    templates = read_lines("errortests/templates.asy")
    autounravel = read_lines("errortests/autounravel.asy")
    operators = read_lines("errortests/operators.asy")

    names_e = read_lines("errortests/names.errors")
    expressions_e = read_lines("errortests/expressions.errors")
    statements_e = read_lines("errortests/statements.errors")
    environment_e = read_lines("errortests/environment.errors")
    permissions_e = read_lines("errortests/permissions.errors")
    var_e = read_lines("errortests/var.errors")
    keyword_rest_e = read_lines("errortests/keyword_rest.errors")
    templates_e = read_lines("errortests/templates.errors")
    autounravel_e = read_lines("errortests/autounravel.errors")
    operators_e = read_lines("errortests/operators.errors")

    # ─── Part 1: Reconstruct errortest.asy from the split files ──────
    #
    # Each split file has a one-line header comment (line 1) that replaced
    # the original section comment.  Some split files also have a blank
    # line 2 after the header where the original section comment was
    # followed by code.
    #
    # permissions.asy and templates.asy each gather two non-contiguous
    # chunks from the original.  The rest are contiguous.

    print("Part 1: Verifying .asy content")

    # Accumulator: every (start, end) pair appended here must, taken
    # together, partition exactly lines 1..689 of errortest.asy.
    orig_ranges = []

    # Lines 1-8: header comment block, not in any split file.
    ok &= check_line("errortest.asy[1] == '/*****'", orig, 1, "/*****\n")
    ok &= check_line("errortest.asy[7] == ' *****/'", orig, 7, " *****/\n")
    ok &= check_line("errortest.asy[8] is blank", orig, 8, "\n")
    orig_ranges.append((1, 8))

    # Line 9: "// name.cc" — the original section comment, replaced by
    # names.asy line 1 ("// Name resolution errors (name.cc)").
    ok &= check_line("errortest.asy[9] == '// name.cc'", orig, 9, "// name.cc\n")
    orig_ranges.append((9, 9))

    # names.asy lines 2-71 == errortest.asy lines 10-79
    ok &= check(
        "names.asy[2:71] == errortest.asy[10:79]",
        lines(names, 2, 71),
        lines(orig, 10, 79),
    )
    orig_ranges.append((10, 79))

    # Line 80: blank separator.
    ok &= check_line("errortest.asy[80] is blank", orig, 80, "\n")
    orig_ranges.append((80, 80))

    # expressions.asy lines 3-111 == errortest.asy lines 81-189
    # (lines 1-2 are the new header + blank, not in orig)
    ok &= check(
        "expressions.asy[3:111] == errortest.asy[81:189]",
        lines(expressions, 3, 111),
        lines(orig, 81, 189),
    )
    orig_ranges.append((81, 189))

    # Line 190: blank separator.
    ok &= check_line("errortest.asy[190] is blank", orig, 190, "\n")
    orig_ranges.append((190, 190))

    # statements.asy lines 3-60 == errortest.asy lines 191-248
    ok &= check(
        "statements.asy[3:60] == errortest.asy[191:248]",
        lines(statements, 3, 60),
        lines(orig, 191, 248),
    )
    orig_ranges.append((191, 248))

    # Line 249: blank separator.
    ok &= check_line("errortest.asy[249] is blank", orig, 249, "\n")
    orig_ranges.append((249, 249))

    # environment.asy lines 3-52 == errortest.asy lines 250-299
    ok &= check(
        "environment.asy[3:52] == errortest.asy[250:299]",
        lines(environment, 3, 52),
        lines(orig, 250, 299),
    )
    orig_ranges.append((250, 299))

    # Line 300: blank separator.
    ok &= check_line("errortest.asy[300] is blank", orig, 300, "\n")
    orig_ranges.append((300, 300))

    # Line 301: "// Test permissions." — original section comment, replaced
    # by permissions.asy line 1 ("// Permission and access control errors.").
    ok &= check_line(
        "errortest.asy[301] == '// Test permissions.'",
        orig,
        301,
        "// Test permissions.\n",
    )
    orig_ranges.append((301, 301))

    # FIRST CHUNK: permissions.asy lines 2-115 == errortest.asy lines 302-415
    ok &= check(
        "permissions.asy[2:115] == errortest.asy[302:415]",
        lines(permissions, 2, 115),
        lines(orig, 302, 415),
    )
    orig_ranges.append((302, 415))

    # Line 416: blank separator.
    ok &= check_line("errortest.asy[416] is blank", orig, 416, "\n")
    orig_ranges.append((416, 416))

    # var.asy lines 1-55 == errortest.asy lines 417-471
    # (var's header "// Test cases where var ..." is identical to orig 417)
    ok &= check(
        "var.asy[1:55] == errortest.asy[417:471]",
        lines(var, 1, 55),
        lines(orig, 417, 471),
    )
    orig_ranges.append((417, 471))

    # Line 472: blank separator.
    ok &= check_line("errortest.asy[472] is blank", orig, 472, "\n")
    orig_ranges.append((472, 472))

    # keyword_rest.asy lines 1-42 == errortest.asy lines 473-514
    # (keyword_rest's header "// Keyword and rest errors." is identical
    # to orig 473)
    ok &= check(
        "keyword_rest.asy[1:42] == errortest.asy[473:514]",
        lines(keyword_rest, 1, 42),
        lines(orig, 473, 514),
    )
    orig_ranges.append((473, 514))

    # Line 515: blank separator.
    ok &= check_line("errortest.asy[515] is blank", orig, 515, "\n")
    orig_ranges.append((515, 515))

    # Line 516: "// template import errors" — original section comment,
    # replaced by templates.asy line 1 ("// Template import errors."
    # — different casing and trailing period).
    ok &= check_line(
        "errortest.asy[516] == '// template import errors'",
        orig,
        516,
        "// template import errors\n",
    )
    orig_ranges.append((516, 516))

    # FIRST CHUNK: templates.asy lines 2-29 == errortest.asy lines 517-544
    ok &= check(
        "templates.asy[2:29] == errortest.asy[517:544]",
        lines(templates, 2, 29),
        lines(orig, 517, 544),
    )
    orig_ranges.append((517, 544))

    # Line 545: blank separator.
    ok &= check_line("errortest.asy[545] is blank", orig, 545, "\n")
    orig_ranges.append((545, 545))

    # Line 546: "// autounravel errors" — original section comment,
    # replaced by autounravel.asy line 1 ("// Autounravel modifier errors."
    # — different wording).
    ok &= check_line(
        "errortest.asy[546] == '// autounravel errors'",
        orig,
        546,
        "// autounravel errors\n",
    )
    orig_ranges.append((546, 546))

    # autounravel.asy lines 2-29 == errortest.asy lines 547-574
    ok &= check(
        "autounravel.asy[2:29] == errortest.asy[547:574]",
        lines(autounravel, 2, 29),
        lines(orig, 547, 574),
    )
    orig_ranges.append((547, 574))

    # SECOND CHUNK of templates: templates.asy lines 30-59
    # == errortest.asy lines 575-604
    ok &= check(
        "templates.asy[30:59] == errortest.asy[575:604]",
        lines(templates, 30, 59),
        lines(orig, 575, 604),
    )
    orig_ranges.append((575, 604))

    # SECOND CHUNK of permissions: permissions.asy lines 116-151
    # == errortest.asy lines 605-640
    ok &= check(
        "permissions.asy[116:151] == errortest.asy[605:640]",
        lines(permissions, 116, 151),
        lines(orig, 605, 640),
    )
    orig_ranges.append((605, 640))

    # operators.asy lines 2-49 == errortest.asy lines 641-688
    # (operators.asy line 1 is "// Operator[] and iterator errors.",
    # a new header; the original had no section comment here)
    ok &= check(
        "operators.asy[2:49] == errortest.asy[641:688]",
        lines(operators, 2, 49),
        lines(orig, 641, 688),
    )
    orig_ranges.append((641, 688))

    # Line 689: "}" without trailing newline.  operators.asy line 50
    # is "}\n" — the split file adds a trailing newline.
    ok &= check_line("errortest.asy[689] == '}'", orig, 689, "}")
    ok &= check_line("operators.asy[50] == '}\\n'", operators, 50, "}\n")
    orig_ranges.append((689, 689))

    # Verify that every line of errortest.asy is accounted for.
    print()
    print("Coverage: errortest.asy")
    ok &= verify_full_coverage("errortest.asy", orig_ranges, len(orig))

    # ─── Part 2: Reconstruct errors from the split .errors files ─────
    #
    # The error messages are identical except filenames (errortest.asy vs
    # errortests/foo.asy) and line numbers differ.  We compare only the
    # message part after "filename: NN.CC: ".
    #
    # The original errors file has a specific ordering determined by asy's
    # error-reporting order: keyword/rest errors come first (reported during
    # parsing), then the rest in source order.  The split files interleave
    # differently because permissions and templates each contributed two
    # non-contiguous chunks to the original.

    print()
    print("Part 2: Verifying .errors content (message text only)")

    # Accumulator: must partition exactly lines 1..214 of errors.
    errs_ranges = []

    S = strip_error_prefix  # shorthand

    # errors lines 1-40: keyword_rest (all 40 lines)
    ok &= check(
        "keyword_rest.errors[1:40] == errors[1:40]",
        [S(l) for l in lines(keyword_rest_e, 1, 40)],
        [S(l) for l in lines(errs, 1, 40)],
    )
    errs_ranges.append((1, 40))

    # errors lines 41-57: names (all 17 lines)
    ok &= check(
        "names.errors[1:17] == errors[41:57]",
        [S(l) for l in lines(names_e, 1, 17)],
        [S(l) for l in lines(errs, 41, 57)],
    )
    errs_ranges.append((41, 57))

    # errors lines 58-76: expressions (all 19 lines)
    ok &= check(
        "expressions.errors[1:19] == errors[58:76]",
        [S(l) for l in lines(expressions_e, 1, 19)],
        [S(l) for l in lines(errs, 58, 76)],
    )
    errs_ranges.append((58, 76))

    # errors lines 77-85: statements (all 9 lines)
    ok &= check(
        "statements.errors[1:9] == errors[77:85]",
        [S(l) for l in lines(statements_e, 1, 9)],
        [S(l) for l in lines(errs, 77, 85)],
    )
    errs_ranges.append((77, 85))

    # errors lines 86-87: environment (all 2 lines)
    ok &= check(
        "environment.errors[1:2] == errors[86:87]",
        [S(l) for l in lines(environment_e, 1, 2)],
        [S(l) for l in lines(errs, 86, 87)],
    )
    errs_ranges.append((86, 87))

    # errors lines 88-132: permissions FIRST CHUNK (45 lines)
    ok &= check(
        "permissions.errors[1:45] == errors[88:132]",
        [S(l) for l in lines(permissions_e, 1, 45)],
        [S(l) for l in lines(errs, 88, 132)],
    )
    errs_ranges.append((88, 132))

    # errors lines 133-156: var (all 24 lines)
    ok &= check(
        "var.errors[1:24] == errors[133:156]",
        [S(l) for l in lines(var_e, 1, 24)],
        [S(l) for l in lines(errs, 133, 156)],
    )
    errs_ranges.append((133, 156))

    # errors lines 157-172: templates FIRST CHUNK (16 lines)
    ok &= check(
        "templates.errors[1:16] == errors[157:172]",
        [S(l) for l in lines(templates_e, 1, 16)],
        [S(l) for l in lines(errs, 157, 172)],
    )
    errs_ranges.append((157, 172))

    # errors lines 173-180: autounravel (all 8 lines)
    ok &= check(
        "autounravel.errors[1:8] == errors[173:180]",
        [S(l) for l in lines(autounravel_e, 1, 8)],
        [S(l) for l in lines(errs, 173, 180)],
    )
    errs_ranges.append((173, 180))

    # errors lines 181-196: templates SECOND CHUNK (16 lines)
    ok &= check(
        "templates.errors[17:32] == errors[181:196]",
        [S(l) for l in lines(templates_e, 17, 32)],
        [S(l) for l in lines(errs, 181, 196)],
    )
    errs_ranges.append((181, 196))

    # errors lines 197-205: permissions SECOND CHUNK (9 lines)
    ok &= check(
        "permissions.errors[46:54] == errors[197:205]",
        [S(l) for l in lines(permissions_e, 46, 54)],
        [S(l) for l in lines(errs, 197, 205)],
    )
    errs_ranges.append((197, 205))

    # errors lines 206-214: operators (all 9 lines)
    ok &= check(
        "operators.errors[1:9] == errors[206:214]",
        [S(l) for l in lines(operators_e, 1, 9)],
        [S(l) for l in lines(errs, 206, 214)],
    )
    errs_ranges.append((206, 214))

    # Verify that every line of errors is accounted for.
    print()
    print("Coverage: errors")
    ok &= verify_full_coverage("errors", errs_ranges, len(errs))

    print()
    if ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
