#!/usr/bin/env python3
"""Runtime error tests for Asymptote collections data structures.

The test harness runs a snippet of Asymptote code in a temporary file,
captures stderr, and checks it against a regex pattern.  Line numbers and
full file paths are deliberately excluded from patterns so that minor source
changes do not break the tests.

Usage:
    python3 tests/test_collections_errors.py           # run all tests
    python3 tests/test_collections_errors.py -v        # verbose
    python3 tests/test_collections_errors.py -k PAT    # filter tests by name
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Optional

# One directory up from this file is the asymptote source root.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ASY = os.path.join(SCRIPT_DIR, "asy")
_DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "base")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Counts:
    passed: int = 0
    failed: int = 0
    skipped: int = 0


@dataclasses.dataclass
class TestRunner:
    verbose: bool = False
    filter_pattern: Optional[str] = None
    asy: str = _DEFAULT_ASY
    base_dir: str = _DEFAULT_BASE_DIR
    simulate_failure: bool = False
    counts: _Counts = dataclasses.field(default_factory=_Counts, init=False)

    def run_asy(self, code: str) -> tuple[str, int]:
        """Write *code* to a temp file, run asy on it, return stderr and return code."""
        if self.simulate_failure:
            return "<simulated output that does not match any expected pattern>", 1
        fd, tmpfile = tempfile.mkstemp(suffix=".asy")
        try:
            # Asy code snippets are embedded in Python triple-quoted strings
            # which inherit Python indentation. Dedent them here so the
            # temporary .asy files contain nicely-formatted code for humans.
            cleaned = textwrap.dedent(code).lstrip("\n")
            with os.fdopen(fd, "w") as f:
                f.write(cleaned)
            result = subprocess.run(
                [self.asy, "-q", "-noautoplain", "-sysdir", self.base_dir, tmpfile],
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stderr, result.returncode
        finally:
            os.unlink(tmpfile)

    def check(self, name: str, code: str, expected_pattern: str) -> bool:
        """Run *code* and verify stderr matches *expected_pattern* (regex).

        Returns True if the test passed (or was skipped), False on failure.
        """
        if self.filter_pattern and not re.search(
            self.filter_pattern, name, re.IGNORECASE
        ):
            self.counts.skipped += 1
            return True

        if self.verbose:
            sys.stdout.write(f"  {name} ... ")
            sys.stdout.flush()
        actual, returncode = self.run_asy(code)
        if returncode == 0 or not re.search(expected_pattern, actual):
            if self.verbose:
                print("FAILED")
                print(f"    Expected pattern: {expected_pattern!r}")
                print(f"    Got:              {actual!r}")
            elif not self.counts.failed:
                print(f"\n  {name} ... FAILED")
                print(f"    Expected pattern: {expected_pattern!r}")
                print(f"    Got:              {actual!r}")
            else:
                sys.stdout.write("F")
                sys.stdout.flush()
            self.counts.failed += 1
            return False
        if self.verbose:
            print("PASSED")
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        self.counts.passed += 1
        return True

    def summary(self) -> bool:
        """Print a summary line and return True iff all tests passed."""
        total = self.counts.passed + self.counts.failed + self.counts.skipped
        run = total - self.counts.skipped
        parts = [f"{self.counts.passed}/{run} passed"]
        if self.counts.failed:
            parts.append(f"{self.counts.failed} failed")
        if self.counts.skipped:
            parts.append(f"{self.counts.skipped} skipped")
        print("\n" + ", ".join(parts))
        return self.counts.failed == 0


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def run_tests(runner: TestRunner) -> None:
    # Tracks whether a non-verbose group is open (needs "PASSED" to close it).
    group_started = False
    failures_at_group_start = 0

    def print_header(title: str) -> None:
        nonlocal group_started, failures_at_group_start
        to_print = f"Testing runtime errors ({title})"
        if runner.verbose:
            print(to_print)
        else:
            if group_started:
                print(
                    "  FAILED"
                    if failures_at_group_start < runner.counts.failed
                    else "PASSED"
                )  # close previous group, newline before next header
            group_started = True
            failures_at_group_start = runner.counts.failed
            print(to_print, end="")

    def end_groups() -> None:
        """Close the final group without a trailing newline."""
        if not runner.verbose and group_started:
            print(
                (
                    "  FAILED"
                    if failures_at_group_start < runner.counts.failed
                    else "PASSED"
                ),
                end="",
            )

    # -----------------------------------------------------------------------
    # Queue
    # -----------------------------------------------------------------------
    print_header("Queue")

    runner.check(
        "pop from empty queue",
        """
        from collections.queue(T=int) access makeQueue;
        var q = makeQueue(new int[]);
        q.pop();
        """,
        r"assert FAILED: Queue is empty",
    )

    runner.check(
        "peek at empty queue",
        """
        from collections.queue(T=int) access makeQueue;
        var q = makeQueue(new int[]);
        q.peek();
        """,
        r"assert FAILED: Queue is empty",
    )

    runner.check(
        "iterator undermined by push",
        """
        from collections.queue(T=int) access makeQueue;
        var q = makeQueue(new int[]{1, 2, 3});
        for (int x : q) {
          q.push(99);
        }
        """,
        r"assert FAILED: Iterator undermined\.",
    )

    runner.check(
        "iterator undermined by pop",
        """
        from collections.queue(T=int) access makeQueue;
        var q = makeQueue(new int[]{1, 2, 3});
        for (int x : q) {
          q.pop();
        }
        """,
        r"assert FAILED: Iterator undermined\.",
    )

    # -----------------------------------------------------------------------
    # HashMap
    # -----------------------------------------------------------------------
    print_header("HashMap")

    runner.check(
        "get missing key (no nullValue)",
        """
        from collections.hashmap(K=string, V=int) access HashMap_K_V;
        HashMap_K_V h;
        int v = h["missing"];
        """,
        r"assert FAILED: Key not found in map",
    )

    runner.check(
        "delete nonexistent key",
        """
        from collections.hashmap(K=string, V=int) access HashMap_K_V;
        HashMap_K_V h;
        h["a"] = 1;
        h.delete("missing");
        """,
        r"assert FAILED: Nonexistent key cannot be deleted",
    )

    runner.check(
        "iterator concurrent modification",
        """
        from collections.hashmap(K=string, V=int) access HashMap_K_V;
        HashMap_K_V h;
        h["a"] = 1;
        h["b"] = 2;
        for (string k : h) {
          h["new_key"] = 99;
        }
        """,
        r"assert FAILED: Concurrent modification",
    )

    runner.check(
        "nullValue not satisfying isNullValue",
        """
        from collections.hashmap(K=string, V=int) access HashMap_K_V;
        HashMap_K_V(nullValue=0, isNullValue=new bool(int v) { return v == -1; });
        """,
        r"assert FAILED: nullValue must satisfy isNullValue",
    )

    runner.check(
        "randomKey from empty map",
        """
        from collections.hashmap(K=string, V=int) access HashMap_K_V;
        HashMap_K_V h;
        h.randomKey();
        """,
        r"assert FAILED: Cannot get a random key from an empty map",
    )

    # -----------------------------------------------------------------------
    # BTreeMap
    # -----------------------------------------------------------------------
    print_header("BTreeMap")

    runner.check(
        "get missing key (no nullValue)",
        """
        from collections.btreemap(K=string, V=int) access BTreeMap_K_V;
        BTreeMap_K_V h;
        int v = h["missing"];
        """,
        r"assert FAILED: Key not found in map",
    )

    runner.check(
        "delete nonexistent key",
        """
        from collections.btreemap(K=string, V=int) access BTreeMap_K_V;
        BTreeMap_K_V h;
        h["a"] = 1;
        h.delete("missing");
        """,
        r"assert FAILED: Nonexistent key cannot be deleted",
    )

    runner.check(
        "iterator concurrent modification",
        """
        from collections.btreemap(K=string, V=int) access BTreeMap_K_V;
        BTreeMap_K_V h;
        h["a"] = 1;
        h["b"] = 2;
        for (string k : h) {
          h["new_key"] = 99;
        }
        """,
        r"assert FAILED: Concurrent modification",
    )

    # -----------------------------------------------------------------------
    # HashSet
    # -----------------------------------------------------------------------
    print_header("HashSet")

    runner.check(
        "get missing item (no nullT)",
        """
        from collections.hashset(T=int) access HashSet_T;
        HashSet_T s;
        s.get(42);
        """,
        r"assert FAILED: Item is not present\.",
    )

    runner.check(
        "push new item (no nullT)",
        """
        from collections.hashset(T=int) access HashSet_T;
        HashSet_T s;
        s.push(42);
        """,
        r"assert FAILED: Adding item via push\(\) without defining nullT\.",
    )

    runner.check(
        "extract missing item (no nullT)",
        """
        from collections.hashset(T=int) access HashSet_T;
        HashSet_T s;
        s.extract(42);
        """,
        r"assert FAILED: Item is not present\.",
    )

    runner.check(
        "getRandom from empty set (no nullT)",
        """
        from collections.hashset(T=int) access HashSet_T;
        HashSet_T s;
        s.getRandom();
        """,
        r"assert FAILED: Cannot get a random item from an empty set",
    )

    runner.check(
        "iterator concurrent modification",
        """
        from collections.hashset(T=int) access HashSet_T;
        HashSet_T s;
        s.add(1);
        s.add(2);
        for (int x : s) {
          s.add(99);
        }
        """,
        r"assert FAILED: Concurrent modification",
    )

    # -----------------------------------------------------------------------
    # BTreeSet  (accessed via collections.btree)
    # -----------------------------------------------------------------------
    print_header("BTreeSet")

    runner.check(
        "get missing item (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.get(42);
        """,
        r"assert FAILED: Item is not present\.",
    )

    runner.check(
        "push new item (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.push(42);
        """,
        r"assert FAILED: Adding item via push\(\) without defining nullT\.",
    )

    runner.check(
        "extract missing item (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.extract(42);
        """,
        r"assert FAILED: Item not found",
    )

    runner.check(
        "min from empty set (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.min();
        """,
        r"assert FAILED: No minimum element to return",
    )

    runner.check(
        "max from empty set (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.max();
        """,
        r"assert FAILED: No maximum element to return",
    )

    runner.check(
        "popMin from empty set (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.popMin();
        """,
        r"assert FAILED: No minimum element to pop",
    )

    runner.check(
        "popMax from empty set (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.popMax();
        """,
        r"assert FAILED: No maximum element to pop",
    )

    runner.check(
        "after: no element strictly greater (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.add(1);
        bs.after(5);
        """,
        r"assert FAILED: No element after item to return",
    )

    runner.check(
        "before: no element strictly less (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.add(5);
        bs.before(1);
        """,
        r"assert FAILED: No element before item to return",
    )

    runner.check(
        "atOrAfter: no element >= item (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.add(1);
        bs.atOrAfter(5);
        """,
        r"assert FAILED: No element after item to return",
    )

    runner.check(
        "atOrBefore: no element <= item (no nullT)",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.add(5);
        bs.atOrBefore(1);
        """,
        r"assert FAILED: No element before item to return",
    )

    runner.check(
        "iterator concurrent modification",
        """
        from collections.btree(T=int) access BTreeSet_T;
        var bs = BTreeSet_T();
        bs.add(1);
        bs.add(2);
        for (int x : bs) {
          bs.add(99);
        }
        """,
        r"assert FAILED: Concurrent modification",
    )

    end_groups()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Runtime error tests for Asymptote collections."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument(
        "-k",
        "--filter",
        metavar="PATTERN",
        help="only run tests whose name matches PATTERN (case-insensitive regex)",
    )
    parser.add_argument(
        "--asy",
        default=_DEFAULT_ASY,
        help="path to the asy executable (default: %(default)s)",
    )
    parser.add_argument(
        "--base-dir",
        default=_DEFAULT_BASE_DIR,
        help="path to the asy base/sysdir (default: %(default)s)",
    )
    parser.add_argument(
        "--simulate-failure",
        action="store_true",
        help="simulate non-matching asy output so every test fails",
    )
    args = parser.parse_args()

    runner = TestRunner(
        verbose=args.verbose,
        filter_pattern=args.filter,
        asy=args.asy,
        base_dir=args.base_dir,
        simulate_failure=args.simulate_failure,
    )

    print(
        f"Running collections runtime error tests (asy = {runner.asy})\n",
        end="\n" if runner.verbose else "",
    )
    run_tests(runner)
    passed = runner.summary()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
