#!/usr/bin/env python3
"""Relocatable sysdir-resolution test matrix.

Exercises settings::resolveSysdir() (locate.cc) -- the logic that decides where
``asy`` looks for ``base/`` when no -dir is given -- by staging the built binary
into the layouts a real deployment produces and probing the resolved directory
with ``asy -c "write(settings.sysdir);"``.  The probe recovers the resolved
value even where asy cannot start; see ``probe``.

Pure stdlib, and runs anywhere asy does (Linux, macOS, Windows, FreeBSD).  A
quiet run prints one character per scenario -- ``.`` pass, ``-`` skip -- under a
per-axis heading; failures always print in full, and ``-v`` prints everything.

The model under test
--------------------

resolveSysdir() searches exactly one *candidate location*, accepted only if it
contains plain.asy (isBaseDir() in locate.cc):

  K1  <exedir>/base                  build tree, relocatable distribution

Otherwise the compiled-in ASYMPTOTE_SYSDIR is returned unchanged and
unvalidated.  The compiled-in path and the texmf tree are often described as
candidates 2 and 3, but neither is selected by a plain.asy test: the first is
the default, and the second (kpsewhich TEXMFMAIN, in initDir()) is consulted iff
the sysdir is empty.  They are two settings of one variable, C.

Two *non-candidates* are staged alongside K1 anyway, as negative controls -- a
base/ at either must be ignored, whatever K1 holds -- since both were candidates
once, behind a build flag since removed:

  K2  <exedir>/../share/asymptote    a relocated GNU-layout install
  K3  <exedir>                       base files flat beside the binary

That leaves 3x3x2 = 18 layout states (K3 is never absent: the executable's own
directory always exists), enumerated exhaustively rather than sampled.  The
other axes feed, or post-process, that same core:

  ROUTE   how asy was launched -- varies only the string executablePath()
          returns, so one discriminating layout per route suffices.
  OVR     -sysdir / -dir / ASYMPTOTE_SYSDIR, applied after resolution.
  C       the compiled-in sysdir: absent / valid / decoy / empty (CTAN), each
          against one layout where K1 fires and one where it does not.
  REG     Windows only: queryRegistry() overwrites systemDir unless it was
          resolved relative to the executable.

Scenario IDs are ``core/<states>`` -- B = holds plain.asy, D = decoy, A =
absent, in K1 K2 K3 order -- and ``<axis>/<case>`` elsewhere.  Requirements not
scriptable in-suite (a compiled-in ``NUL``, a deployed texmf tree) are SKIPped.

Three asy paths are named apart throughout: ``asy_under_test`` is the binary
given on the command line, ``staged_asy`` a copy placed into a staged layout,
and ``asy_to_run`` the probe layer's parameter, which takes either.
"""

import argparse
import contextlib
import itertools
import ntpath
import os
import shutil
import subprocess
import sys
import tempfile
from enum import Enum, IntEnum
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

# Annotations are written in the 3.7-compatible spelling (see py-version in
# .pylintrc): typing aliases rather than builtin generics (3.9) or PEP 604
# unions (3.10), so they stay valid at runtime on the oldest interpreter.

# ---------------------------------------------------------------------------
# result bookkeeping
# ---------------------------------------------------------------------------


class Status(Enum):
    """The outcome of one scenario.

    A plain Enum rather than a ``str`` mixin, which formats as ``PASS`` up to
    3.10 but ``Status.PASS`` from 3.11 on; hence the explicit ``.value`` below.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


_results: List[Tuple[str, Status, str]] = []  # (scenario, status, message)


class _Log:  # pylint: disable=too-few-public-methods
    """How much is printed, and whether a progress line is currently open.

    Module state, so that no axis has to thread a verbosity argument down to
    record(); a class rather than two variables so neither needs a ``global``.
    """

    verbose = False
    line_open = False  # characters written since the last newline

    @classmethod
    def close_line(cls) -> None:
        """End the progress line, if one is open, without leaving a blank."""
        if cls.line_open:
            print()
            cls.line_open = False


def record(scenario: str, status: Status, message: str = "") -> None:
    """Log one scenario outcome.

    A quiet run collapses a pass or a skip to a single character and prints
    failures in full, so no failing run needs -v to be readable.
    """
    _results.append((scenario, status, message))
    if _Log.verbose:
        print(f"  [{status.value}] {scenario:<18} {message}")
    elif status is Status.FAIL:
        _Log.close_line()
        print(f"  [{status.value}] {scenario:<18} {message}")
    else:
        sys.stdout.write("." if status is Status.PASS else "-")
        sys.stdout.flush()
        _Log.line_open = True


def section(title: str) -> None:
    """Start a group of scenarios: a heading when verbose, otherwise the label
    this group's progress characters trail after."""
    _Log.close_line()
    if _Log.verbose:
        print(f"\n{title}")
    else:
        print(f"{title:<12} ", end="")
        sys.stdout.flush()


def norm(path: Optional[str]) -> str:
    """Canonical form for comparing two paths across OSes (case, symlinks, ..).
    An absent or empty path normalizes to "", so two of those compare equal."""
    if not path:
        return ""
    return os.path.normcase(os.path.realpath(path))


# ---------------------------------------------------------------------------
# the child environment
# ---------------------------------------------------------------------------

# An envSetting beats the resolved value, so an ASYMPTOTE_SYSDIR exported in
# the calling shell would be the answer to every probe and collapse the whole
# matrix onto one value.  ASYMPTOTE_DIR is the same hazard one step removed: it
# prepends to the import search path, which decides whether asy runs at all.
# Both are dropped from the children's environment only, never from os.environ.
_STRIPPED = ("ASYMPTOTE_SYSDIR", "ASYMPTOTE_DIR")

# ASYMPTOTE_HOME is redirected rather than stripped, in main(), once there is a
# scratch directory to point it at: dropping it would fall back to $HOME/.asy,
# the *more* likely place to hold a config.asy.  Such a file cannot reach sysdir
# (setOptions restores it across the config load) but can set anything else.
_CHILD_ENV: Dict[str, str] = {k: v for k, v in os.environ.items() if k not in _STRIPPED}


def child_env(**overrides: str) -> Dict[str, str]:
    """The environment probes run in, plus any per-scenario additions.  Use this
    rather than ``dict(os.environ, ...)``, which carries the stripped ones back."""
    return dict(_CHILD_ENV, **overrides)


def silence_error_dialogs() -> None:
    """Keep a child that cannot start from opening a modal dialog.

    Many scenarios deliberately stage a binary that will fail, and on Windows a
    failure to *start* is a loader "critical error", which pops a message box
    per launch and waits for a click nobody gives it under ctest.  The error
    mode is inherited, so setting it once here covers every probe.  Only the
    display is suppressed; the exit status the scenarios read is unaffected.
    """
    if sys.platform != "win32":
        return
    # Local import, guarded on sys.platform, for the reason given in
    # windows_docdir(): ctypes.windll exists only on Windows.
    import ctypes  # pylint: disable=import-outside-toplevel

    sem_failcriticalerrors = 0x0001
    sem_nogpfaulterrorbox = 0x0002
    ctypes.windll.kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfaulterrorbox)


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------


def brief(text: str, limit: int = 120) -> str:
    """Collapse a multi-line error to one line, for the single-line records.
    Only used where the message is a footnote; failures keep the full text."""
    one = " ".join(text.split())
    return one if len(one) <= limit else one[: limit - 3] + "..."


def is_base_dir(path: Optional[str]) -> bool:
    """Mirror of isBaseDir() (locate.cc): a directory counts as a base
    directory only if plain.asy is in it.  Mere existence proves nothing."""
    if not path:  # None, and also "" -- the empty TeXLive sysdir
        return False
    return os.path.isfile(os.path.join(path, "plain.asy"))


def launch_target(
    asy_to_run: str, cwd: Optional[str], env: Mapping[str, str]
) -> Optional[str]:
    """The absolute path behind ``asy_to_run``, or None if it cannot be found.

    The ROUTE axis names the executable the awkward ways a user can, and on
    Windows both lookups belong to CreateProcess, which ignores the arguments
    subprocess passes for them: a relative path resolves against *this* process's
    cwd (WinError 2), and a bare name is searched on the *parent's* PATH -- which
    silently launched whichever asy was already installed.  So on Windows the
    route is resolved here and passed as ``executable=``, costing the axis
    nothing (GetModuleFileNameW reports the loader's path) and leaving a symlink
    unresolved, since neither abspath nor which() follows one.

    POSIX forks, chdirs and execs, so both lookups behave as the call reads;
    there this result is only a reachability check.
    """
    if os.path.dirname(asy_to_run):  # a path, absolute or relative
        return os.path.abspath(os.path.join(cwd or os.getcwd(), asy_to_run))
    return shutil.which(asy_to_run, path=env.get("PATH"))  # a bare name


def _run_probe(
    asy_to_run: str,
    cwd: Optional[str],
    env: Optional[Mapping[str, str]],
    args: Sequence[str],
) -> Tuple[bool, str, str]:
    """One ``asy ... -c 'write(settings.sysdir);'`` run.

    Returns (ok, stdout, error): on success stdout is the resolved sysdir; on
    failure error is the (trimmed) stderr.  ``env`` of None means the sanitized
    default, not "inherit".

    ``executable=`` is substituted on Windows only (launch_target says why), and
    withheld on POSIX for macOS's sake: _NSGetExecutablePath hands back the
    string passed to execve, so substituting a resolved path would leave
    realpath() nothing to do and route/relative and route/symlink-rel would pass
    without testing their branch.
    """
    cmd = [asy_to_run, *args, "-c", "write(settings.sysdir);"]
    child = _CHILD_ENV if env is None else env
    target = launch_target(asy_to_run, cwd, child)
    if target is None:
        return False, "", f"could not launch: {asy_to_run!r} is not on the child PATH"
    try:
        # A non-zero exit is a result we report, not an error: check=False.
        run = subprocess.run(
            cmd,
            executable=target if sys.platform == "win32" else None,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=child,
            timeout=120,
            check=False,
        )
    except OSError as exc:
        return False, "", f"could not launch: {exc}"
    if run.returncode == 0:
        return True, run.stdout.strip(), ""
    return False, "", (run.stderr.strip() or f"exit {run.returncode}")


def probe(
    asy_to_run: str,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    extra_args: Sequence[str] = (),
    rescue_base: Optional[str] = None,
) -> Tuple[bool, str, Optional[str]]:
    """Ask asy what it resolved sysdir to.

    Returns (ok, value, resolved):

      ok        the run exited 0 -- asy found a usable base/ and ran the
                command.
      value     the resolved sysdir when ok, else the (trimmed) stderr.
      resolved  the resolved sysdir whenever we could obtain it, including from
                a failing run, else None.  "" -- what a TeXLive build reports
                when no candidate matched -- is a value, not a failure.

    plain.asy itself executes the -c command, so a run that cannot load plain
    prints nothing.  That is too coarse: resolveSysdir() never fails, it returns
    a compiled-in path that may not exist, so a non-zero exit conflates "no
    sysdir" with "an unusable sysdir" and with an unrelated failure.
    ``rescue_base`` closes the gap by retrying with ``-dir <rescue_base>``,
    which feeds plain without altering settings.sysdir (what ovr/dir asserts).
    -noautoplain would not do: it suppresses the -c command and exits 0.
    """
    ok, out, err = _run_probe(asy_to_run, cwd, env, args=extra_args)
    if ok:
        return True, out, out
    resolved = None
    if rescue_base is not None:
        # A repeated -dir replaces rather than accumulates (last one wins), so
        # the rescue goes first: a caller that passes its own -dir keeps it.
        rok, rout, _ = _run_probe(
            asy_to_run, cwd, env, args=("-dir", rescue_base, *extra_args)
        )
        if rok:
            resolved = rout
    return False, err, resolved


def expect(
    scenario: str,
    asy_to_run: str,
    expected: str,
    *,
    runs: bool,
    note: str = "",
    **kw: Any,
) -> bool:
    """Assert asy resolves sysdir to ``expected``, and then starts iff ``runs``.

    Two-part, because resolveSysdir() cannot report failure (see ``probe``): the
    resolved path must be the predicted one *and* asy must run exactly when the
    callsite says.  The exit status alone would score a PASS for any unrelated
    startup failure.

    ``runs`` is stated at the callsite rather than derived, so a scenario reads
    without first working out whether ``expected`` was staged with plain.asy;
    it is cross-checked against is_base_dir(expected), and a disagreement is a
    test bug.  Rows whose answer depends on the host pass
    ``runs=is_base_dir(...)``, naming that dependency where it arises; ovr/dir
    is the one row outside the model, its success licensed by ``-dir``.

    ``**kw`` is forwarded verbatim to ``probe``, whose signature is the
    authority on the accepted keys.
    """
    # Checked before probing: if the claim and the staging disagree there is no
    # scenario to run, and asy's behaviour would only obscure that.
    if is_base_dir(expected) != runs:
        record(
            scenario,
            Status.FAIL,
            f"test bug: expected sysdir {expected!r} was declared "
            f"{'usable' if runs else 'unusable'} but "
            f"{'holds no plain.asy' if runs else 'holds plain.asy'}",
        )
        return False
    suffix = f" ({note})" if note else ""
    ok, val, resolved = probe(asy_to_run, **kw)
    if resolved is None:
        record(
            scenario,
            Status.FAIL,
            f"expected sysdir {expected!r}{suffix}, but asy failed and the "
            f"value could not be recovered: {brief(val)}",
        )
        return False
    if norm(resolved) != norm(expected):
        record(scenario, Status.FAIL, f"expected sysdir {expected!r}, got {resolved!r}")
        return False
    if ok != runs:
        outcome = "ran" if ok else f"failed ({brief(val)})"
        record(
            scenario,
            Status.FAIL,
            f"sysdir {resolved!r} as expected, but asy {outcome} while "
            f"{'a usable' if runs else 'no usable'} base/ was expected there",
        )
        return False
    shown = resolved if resolved else "<empty>"
    tail = "" if runs else " (unusable, as expected)"
    record(scenario, Status.PASS, f"{shown}{tail}{suffix}")
    return True


# ---------------------------------------------------------------------------
# staging: materializing one point of the layout space
# ---------------------------------------------------------------------------


class State(Enum):
    """The state of one candidate location.  The value is its letter in the
    core scenario IDs; declaration order is the order they are enumerated in."""

    VALID = "B"  # B for base directory, not the member's own initial
    DECOY = "D"
    ABSENT = "A"

    @property
    def label(self) -> str:
        """The spelled-out name, used in the C-axis scenario IDs."""
        return self.name.lower()


class Location(IntEnum):
    """The executable-relative locations the matrix stages: K1 the candidate,
    K2 and K3 the negative controls (see the module docstring).  The value is
    the position in a States or Paths triple, so a Location subscripts either.
    """

    K1 = 0
    K2 = 1
    K3 = 2

    @property
    def where(self) -> str:
        """Human-readable location, for the scenario notes."""
        return {
            Location.K1: "<exedir>/base",
            Location.K2: "<exedir>/../share/asymptote",
            Location.K3: "<exedir>",
        }[self]


# One point of the layout space: the state of (K1, K2, K3), in that order.
States = Tuple[State, State, State]

# The same three locations as staged paths, so Paths[i] has state States[i].
Paths = Tuple[str, str, str]


def copy_base_into(base_dir: str, dst: str, with_plain: bool = True) -> None:
    """Copy the base files into ``dst``, creating it if need be.

    ``with_plain=False`` builds a decoy: everything but plain.asy, the only file
    resolveSysdir() looks for.  Hand-rolled because it must tolerate an existing
    destination (the flat layout) and copytree's dirs_exist_ok is 3.8+.
    """
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(base_dir):
        if name == "plain.asy" and not with_plain:
            continue
        src = os.path.join(base_dir, name)
        tgt = os.path.join(dst, name)
        if os.path.isdir(src):
            shutil.copytree(src, tgt)
        else:
            shutil.copy2(src, tgt)


def materialize(path: str, state: State, base_dir: str) -> None:
    """Put location ``path`` into ``state``."""
    if state is State.ABSENT:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    copy_base_into(base_dir, path, with_plain=state is State.VALID)


# Shared libraries a deployment bundles *beside* the executable, by platform.
# Carrying its own runtime this way is part of what "relocatable" means, so they
# are staged with the binary: otherwise the copy is not the deployment shape the
# matrix claims to test, and on Windows it does not start at all (0xC0000135,
# before main()).  macOS lists both suffixes since dyld loads either; everything
# else takes the ELF .so default, FreeBSD included (its sys.platform carries the
# major version, so it could not be a key here anyway).
_LIB_SUFFIXES = {"win32": (".dll",), "darwin": (".dylib", ".so")}
_lib_suffixes: Tuple[str, ...] = _LIB_SUFFIXES.get(sys.platform, (".so",))


def is_bundled_lib(name: str) -> bool:
    """Is ``name`` a shared library that travels with the executable?

    Matched on suffix, not an exact list, so a changed dependency needs no edit
    here; the ``.so.`` test catches the versioned ``libfoo.so.1.2`` spelling.
    """
    lower = name.lower()
    if lower.endswith(_lib_suffixes):
        return True
    if ".so" in _lib_suffixes and ".so." in lower:
        return True
    return False


def copy_bundled_libs(asy_under_test: str, dst_dir: str) -> None:
    """Copy the shared libraries bundled beside ``asy_under_test`` into dst_dir.

    None of them is plain.asy, so staging them cannot change which candidate
    resolveSysdir() selects -- only whether the copy gets far enough to report.
    """
    os.makedirs(dst_dir, exist_ok=True)
    srcdir = os.path.dirname(asy_under_test)
    for name in os.listdir(srcdir):
        src = os.path.join(srcdir, name)
        if is_bundled_lib(name) and os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, name))
    # A macOS bundle collects them into lib/ beside the binary instead, with the
    # references rewritten to @executable_path/lib/, so that directory travels
    # whole.  The guard keeps the destination fresh, so copytree needs no
    # dirs_exist_ok (3.8+).
    libdir = os.path.join(srcdir, "lib")
    staged_libdir = os.path.join(dst_dir, "lib")
    if os.path.isdir(libdir) and not os.path.exists(staged_libdir):
        shutil.copytree(libdir, staged_libdir)


def stage_binary(dst_dir: str, asy_under_test: str) -> str:
    """Copy the asy binary, and its bundled libraries, into dst_dir; return the
    path to the copy."""
    os.makedirs(dst_dir, exist_ok=True)
    staged_asy = os.path.join(dst_dir, os.path.basename(asy_under_test))
    shutil.copy2(asy_under_test, staged_asy)
    copy_bundled_libs(asy_under_test, dst_dir)
    return staged_asy


def stage_layout(
    root: str, asy_under_test: str, base_dir: str, states: States
) -> Tuple[str, Paths]:
    """Stage a prefix realizing one (K1, K2, K3) state triple.

    The layout is <root>/bin/asy, so all three locations are distinct paths and
    can be populated independently.  Returns (staged_asy, paths) with paths[i]
    the directory location i names.
    """
    bindir = os.path.join(root, "bin")
    staged_asy = stage_binary(bindir, asy_under_test)
    paths = (
        os.path.join(bindir, "base"),
        os.path.join(root, "share", "asymptote"),
        bindir,
    )
    for k in (Location.K1, Location.K2):
        materialize(paths[k], states[k], base_dir)
    # K3 is the executable's own directory, so it always exists: ABSENT is
    # unreachable and DECOY means "populated but no plain.asy", not "empty".
    assert states[Location.K3] is not State.ABSENT, "<exedir> cannot be absent"
    copy_base_into(base_dir, bindir, with_plain=states[Location.K3] is State.VALID)
    return staged_asy, paths


def winner(states: States) -> Optional[Location]:
    """The oracle: the location resolveSysdir() must select, or None if it must
    fall through to the compiled-in value.  K1 is the entire search; the other
    two are controls, and their state cannot change the answer."""
    if states[Location.K1] is State.VALID:
        return Location.K1
    return None


def nearest_existing(path: str) -> str:
    """Return the closest existing ancestor of path (which itself is absent)."""
    p = os.path.dirname(os.path.abspath(path))
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def can_create(path: str) -> bool:
    """True if we can materialize ``path`` -- it is absent and some existing
    ancestor is writable so os.makedirs would succeed."""
    if os.path.exists(path):
        return False
    anc = nearest_existing(path)
    return bool(anc) and os.access(anc, os.W_OK)


def rollback_target(path: str) -> str:
    """The topmost directory os.makedirs(path) would create, for later removal.
    Removing it undoes the whole chain without touching pre-existing dirs."""
    anc = nearest_existing(path)
    rel = os.path.relpath(os.path.abspath(path), anc)
    first = rel.split(os.sep, maxsplit=1)[0]
    return os.path.join(anc, first)


@contextlib.contextmanager
def temporary_tree(path: str) -> Generator[None, None, None]:
    """Let the body materialize ``path``, then remove it again on the way out.

    The rollback target is computed on entry, while ``path`` is still absent;
    that is what makes it the topmost *newly created* directory.  So the body
    must create ``path``, not the caller."""
    undo = rollback_target(path)
    try:
        yield
    finally:
        shutil.rmtree(undo, ignore_errors=True)


# ---------------------------------------------------------------------------
# the fall-through value
# ---------------------------------------------------------------------------


def windows_docdir() -> str:
    """The value queryRegistry() will impose on a non-relocated sysdir.

    queryRegistry() ends with ``systemDir = docdir`` unless the sysdir came from
    the executable, and docdir is the registry's App Paths\\Asymptote entry or a
    never-empty hard-coded default.  So on Windows this, not the compiled-in
    path, is the fall-through value.  An unreadable registry yields the same
    default, mirroring settings.cc rather than reporting a failure.
    """
    default = "c:\\Program Files\\Asymptote"
    if sys.platform != "win32":
        return default
    # Guarded on sys.platform rather than ImportError so a type checker can see
    # winreg is only touched where it exists.  mypy then prunes everything below
    # on Linux, which is why misc-sanity-checks.yml also types this file with
    # --platform=win32.
    import winreg  # pylint: disable=import-outside-toplevel,import-error

    key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Asymptote"
    # HKEY_LOCAL_MACHINE first, the order getEntry() searches the roots in
    # (settings.cc:259) -- the whole answer when both are set to different paths.
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(root, key) as handle:
                value, kind = winreg.QueryValueEx(handle, "Path")
        except OSError:
            continue  # not under this root; RegGetValueA fails here too
        # QueryValueEx is typed Any; RRF_RT_REG_SZ makes RegGetValueA skip a
        # non-string root as well.
        if not isinstance(value, str):
            continue
        # An empty string counts as found: getEntry() returns optional("") and
        # stops (settings.cc:266), so docdir keeps its default.
        if not value:
            break
        # RegGetValueA expands environment strings (no RRF_NOEXPAND); ntpath so
        # the expansion is Windows' even off Windows.
        if kind == winreg.REG_EXPAND_SZ:
            return ntpath.expandvars(value)
        return value
    return default


def fallback_sysdir(compiled_in: Optional[str]) -> Optional[str]:
    """What asy must report when no candidate fires, or None if unpredictable.

    Off Windows that is the compiled-in value verbatim, so it is predictable
    only when the caller named it.  A compiled-in "" (the CTAN build) hands off
    to kpsewhich instead, which run_ctan_axis checks.
    """
    if os.name == "nt":
        return windows_docdir()
    return compiled_in


# ---------------------------------------------------------------------------
# the core matrix: every reachable (K1, K2, K3) state, K1 the only candidate
# ---------------------------------------------------------------------------


class Ctx(NamedTuple):
    """Everything the checks need that is fixed for the whole run."""

    asy_under_test: str  # the binary under test
    base_dir: str  # its build-tree base/, used as the -dir rescue
    compiled_in: Optional[str]  # ASYMPTOTE_SYSDIR, if the caller named it
    work: str  # scratch root; also the cwd every probe runs in

    def stage(
        self, name: str, states: States, asy_under_test: Optional[str] = None
    ) -> Tuple[str, Paths]:
        """Stage layout ``states`` under <work>/<name>; see stage_layout.
        ``asy_under_test`` overrides the run's binary for this one layout, which
        is how the CTAN axis stages a different binary into the same shapes."""
        return stage_layout(
            os.path.join(self.work, name),
            asy_under_test or self.asy_under_test,
            self.base_dir,
            states,
        )

    def expect(
        self, scenario: str, asy_to_run: str, expected: str, *, runs: bool, **kw: Any
    ) -> bool:
        """Call expect() with this run's cwd and rescue filled in.

        ``**kw`` is forwarded to the module-level ``expect``, whose signature is
        the authority on the accepted keys.  Note that the two setdefault keys
        below must stay spelled as ``probe``'s parameters: a typo would add an
        unused key and drop the intended default rather than error.
        """
        kw.setdefault("cwd", self.work)
        kw.setdefault("rescue_base", self.base_dir)
        return expect(scenario, asy_to_run, expected, runs=runs, **kw)


def state_code(states: States) -> str:
    """The letter triple that names a core scenario and its staging directory."""
    return "".join(s.value for s in states)


def check_layout(ctx: Ctx, scenario: str, states: States) -> None:
    """Stage one state triple and assert the oracle's prediction."""
    staged_asy, paths = ctx.stage(os.path.join("core", state_code(states)), states)
    idx = winner(states)
    if idx is not None:
        # The winner is by definition a location staged with plain.asy.
        ctx.expect(
            scenario,
            staged_asy,
            paths[idx],
            runs=True,
            note=f"{idx.name} {idx.where}",
        )
        return

    # No winner found by oracle
    fallback = fallback_sysdir(ctx.compiled_in)
    if fallback is not None:
        # Whether the fall-through value is usable is a property of the host,
        # not of the staging, so it is measured rather than declared.
        ctx.expect(
            scenario,
            staged_asy,
            fallback,
            runs=is_base_dir(fallback),
            note="fallback",
        )
        return

    # No --compiled-in given: we cannot name the fall-through value, but we can
    # still assert the negative half -- that no candidate fired.
    ok, val, resolved = probe(staged_asy, cwd=ctx.work, rescue_base=ctx.base_dir)
    if resolved is None:
        record(scenario, Status.SKIP, f"sysdir not recoverable: {brief(val)}")
    elif any(norm(p) == norm(resolved) for p in paths):
        record(scenario, Status.FAIL, f"no candidate should fire, but got {resolved!r}")
    elif ok != is_base_dir(resolved):
        ran = "ran" if ok else "failed"
        record(scenario, Status.FAIL, f"asy {ran} with sysdir {resolved!r}")
    else:
        record(scenario, Status.PASS, f"fell through to {resolved!r} (unverified)")


def run_core_matrix(ctx: Ctx) -> None:
    """All 3x3x2 layout states, exhaustively.

    Enumerating the space outright is cheaper than arguing about which rows
    would suffice, and it *measures* rather than assumes the two properties
    that keep K2 and K3 from creeping back in as candidates: that a decoy is
    equivalent to an absent directory, and that the result ignores both.
    """
    for states in itertools.product(State, State, (State.VALID, State.DECOY)):
        check_layout(ctx, "core/" + state_code(states), states)


# ---------------------------------------------------------------------------
# ROUTE: how the executable was launched
# ---------------------------------------------------------------------------


# A layout in which the answer depends on where the executable is: K1 holds the
# only base/, so a wrong exedir yields a visibly different path rather than the
# same one.
ROUTE_LAYOUT: States = (State.VALID, State.ABSENT, State.DECOY)


def run_route_axis(ctx: Ctx) -> None:
    """Every way of naming the executable must resolve the same layout.

    These cases vary only executablePath(), so one discriminating layout per
    route is enough; ROUTE x layout would multiply the matrix by 6 and prove
    nothing the core matrix does not.
    """
    states = ROUTE_LAYOUT
    root = os.path.join(ctx.work, "route")
    staged_asy, paths = ctx.stage("route", states)
    idx = winner(states)
    assert idx is not None, "the route layout must have a winner"
    expected = paths[idx]
    name = os.path.basename(staged_asy)

    # Every row here resolves the winning candidate, which holds plain.asy, so
    # the route is right only if asy also starts: runs=True throughout.
    ctx.expect("route/direct", staged_asy, expected, runs=True)

    link_dir = os.path.join(ctx.work, "route-link")
    os.makedirs(link_dir)
    try:
        os.symlink(staged_asy, os.path.join(link_dir, name))
    except (OSError, NotImplementedError, AttributeError) as exc:
        record("route/symlink", Status.SKIP, f"symlinks unavailable: {exc}")
    else:
        # Windows' loader takes the *link's* directory as the application
        # directory for the DLL search, so a link into a bundled install cannot
        # start without the libraries beside it too.  This cannot blunt the row:
        # link_dir still holds no base/, so an exedir that stayed here resolves
        # a different sysdir.  A no-op where the link is resolved first.
        copy_bundled_libs(ctx.asy_under_test, link_dir)
        # Homebrew/MacPorts shape.  This row is for the two platforms that
        # report the link rather than its target -- macOS, where realpath()
        # resolves it, and Windows, where canonicalPath() does; elsewhere it is
        # a regression test that the answer comes back resolved regardless.
        ctx.expect("route/symlink", os.path.join(link_dir, name), expected, runs=True)
        # ... and reached by a relative path, which _NSGetExecutablePath() hands
        # back verbatim for realpath() to resolve against the child's cwd.  It
        # arrives that way only because _run_probe withholds executable=.
        ctx.expect(
            "route/symlink-rel",
            os.path.join(os.curdir, name),
            expected,
            runs=True,
            cwd=link_dir,
        )

    bindir = os.path.dirname(staged_asy)
    env = child_env(PATH=bindir + os.pathsep + _CHILD_ENV.get("PATH", ""))
    ctx.expect("route/path", name, expected, runs=True, env=env)

    # The other row that reaches macOS as a relative string; see route/symlink-rel.
    ctx.expect(
        "route/relative",
        os.path.join(os.curdir, os.path.relpath(staged_asy, root)),
        expected,
        runs=True,
        cwd=root,
    )

    # Moving the whole tree: nothing may have been baked in at build time.
    # Done last -- it invalidates the symlink and the PATH entry above.  relpath
    # is lexical, so it does not matter that staged_asy no longer exists.
    moved = os.path.join(ctx.work, "route-moved")
    shutil.move(root, moved)
    ctx.expect(
        "route/moved",
        os.path.join(moved, os.path.relpath(staged_asy, root)),
        os.path.join(moved, os.path.relpath(expected, root)),
        runs=True,
    )


# ---------------------------------------------------------------------------
# OVR: the command-line and environment overrides
# ---------------------------------------------------------------------------


def run_override_axis(ctx: Ctx) -> None:
    """-sysdir / -dir / ASYMPTOTE_SYSDIR, applied on top of the resolved value.

    Each override points somewhere resolution would not have chosen, so that
    "applied" and "ignored" are distinguishable.  Layout is held fixed: these
    act on the result of resolution, not on its inputs.
    """
    alt = os.path.join(ctx.work, "altbase")
    materialize(alt, State.VALID, ctx.base_dir)
    bogus = os.path.join(ctx.work, "does-not-exist")

    # Run in place rather than staged: the layout resolved from is irrelevant.
    # -sysdir replaces the resolved value outright, valid or not -- asy runs
    # from a real base/ and cannot start from a nonexistent one, and both are
    # equally the override having been applied.
    ctx.expect(
        "ovr/sysdir", ctx.asy_under_test, alt, runs=True, extra_args=("-sysdir", alt)
    )
    ctx.expect(
        "ovr/sysdir-bogus",
        ctx.asy_under_test,
        bogus,
        runs=False,
        extra_args=("-sysdir", bogus),
    )

    # -dir only prepends to the import search path.  This row is what licenses
    # the rescue -dir probe() uses elsewhere, so it takes no rescue itself.
    ctx.expect(
        "ovr/dir",
        ctx.asy_under_test,
        ctx.base_dir,  # sysdir, not dir
        runs=True,
        extra_args=("-dir", alt),
        rescue_base=None,
    )

    # The environment variable overrides everything (envSetting, settings.cc).
    # The one row that puts ASYMPTOTE_SYSDIR back, which is what makes its
    # result attributable.
    ctx.expect(
        "ovr/env",
        ctx.asy_under_test,
        alt,
        runs=True,
        env=child_env(ASYMPTOTE_SYSDIR=alt),
    )


# ---------------------------------------------------------------------------
# C: the compiled-in sysdir
# ---------------------------------------------------------------------------


# The only two layouts at which C, and the Windows registry, can matter: one
# where no candidate fires, so the fallback decides, and one where K1 fires, so
# the fallback must be ignored.  Every other layout is equivalent to one of
# these as far as those variables go, which is what makes the axes |values| x 2
# rather than |values| x 18.  Named for the candidate search, not the fallback.
MISS = (State.ABSENT, State.ABSENT, State.DECOY)
HIT = (State.VALID, State.ABSENT, State.DECOY)


def run_compiled_in_axis(ctx: Ctx) -> None:
    """C = absent / valid / decoy, at MISS and at HIT.

    The compiled-in path is mutated only when that is safe: the caller named
    it, it does not already exist (never delete a real install), and its parent
    is writable.  Otherwise this reports what the host happens to provide.
    """
    if os.name == "nt":
        record("C/*", Status.SKIP, "on Windows the fallback is the registry docdir")
        return
    if ctx.compiled_in is None:
        record("C/*", Status.SKIP, "no --compiled-in given")
        return
    compiled_in = ctx.compiled_in

    staged_miss, _ = ctx.stage("c-miss", MISS)
    staged_hit, hit_paths = ctx.stage("c-hit", HIT)

    if os.path.exists(compiled_in):
        # A real tree lives there; verify it is used but do not touch it.  What
        # it holds is the host's business, so -miss takes its expectation from
        # the measurement that picked the label.
        state = State.VALID if is_base_dir(compiled_in) else State.DECOY
        ctx.expect(
            f"C/{state.label}-miss",
            staged_miss,
            compiled_in,
            runs=state is State.VALID,
        )
        ctx.expect(
            f"C/{state.label}-hit", staged_hit, hit_paths[Location.K1], runs=True
        )
        record("C/absent-*", Status.SKIP, "compiled-in path exists (real install)")
        return

    if not can_create(compiled_in):
        record(
            "C/*", Status.SKIP, f"cannot create the compiled-in path {compiled_in!r}"
        )
        return

    # The absent case: it is returned anyway, unvalidated and unusable.
    ctx.expect("C/absent-miss", staged_miss, compiled_in, runs=False)
    ctx.expect("C/absent-hit", staged_hit, hit_paths[Location.K1], runs=True)

    # Here the compiled-in tree is staged by this loop, so -miss runs exactly
    # when the state being staged is the one with plain.asy in it.
    for state in (State.VALID, State.DECOY):
        with temporary_tree(compiled_in):
            materialize(compiled_in, state, ctx.base_dir)
            ctx.expect(
                f"C/{state.label}-miss",
                staged_miss,
                compiled_in,
                runs=state is State.VALID,
            )
            # The regression the reordering exists for: an unrelated install at
            # the compiled-in path must not preempt the binary's own base/.
            ctx.expect(
                f"C/{state.label}-hit", staged_hit, hit_paths[Location.K1], runs=True
            )


def run_ctan_axis(ctx: Ctx, asy_ctan: str) -> None:
    """C = "" -- the CTAN/TeXLive binary, which defers to kpsewhich.

    Same two layouts: at HIT the adjacent base/ must still win (kpsewhich is the
    last resort, not a mode), and at MISS the texmf tree answers -- and in
    particular the answer must not be a build or staging path.
    """
    staged_hit, hit_paths = ctx.stage("ctan-hit", HIT, asy_under_test=asy_ctan)
    ctx.expect("ctan/hit", staged_hit, hit_paths[Location.K1], runs=True)

    # Deployed TeXLive shape: bin/<platform>/asy, no adjacent base/.  Where the
    # texmf tree has no asymptote/ the binary cannot start, but the rescue -dir
    # reports what it resolved anyway; an empty sysdir (no kpsewhich answer at
    # all) is legitimate, hence the `is None` test.  The platform name is a
    # stand-in -- nothing reads it, and the shape only needs a directory with no
    # base/ beside it and no share/asymptote above it.
    platform_dir = os.path.join(ctx.work, "ctan-miss", "bin", "x86_64-linux")
    staged_asy = stage_binary(platform_dir, asy_ctan)
    _, val, resolved = probe(staged_asy, cwd=ctx.work, rescue_base=ctx.base_dir)
    if resolved is None:
        record(
            "ctan/miss", Status.SKIP, f"deployed binary would not start: {brief(val)}"
        )
        return
    bad_roots = [r for r in (norm(ctx.base_dir), norm(ctx.work)) if r]
    if resolved and any(norm(resolved).startswith(r) for r in bad_roots):
        record(
            "ctan/miss", Status.FAIL, f"resolved into build/staging tree: {resolved!r}"
        )
    elif not shutil.which("kpsewhich"):
        record(
            "ctan/miss",
            Status.SKIP,
            f"no kpsewhich; got {resolved!r}, not a build path",
        )
    else:
        record("ctan/miss", Status.PASS, f"kpsewhich/texmf result: {resolved!r}")


# ---------------------------------------------------------------------------
# REG: the Windows registry override
# ---------------------------------------------------------------------------


def run_registry_axis(ctx: Ctx) -> None:
    """Only a non-relocated sysdir may be replaced by queryRegistry().

    Read-only: the registry describes whatever Asymptote is installed on the
    machine, so this reports rather than arranges.
    """
    if os.name != "nt":
        record("reg/*", Status.SKIP, "Windows only")
        return
    staged_hit, hit_paths = ctx.stage("reg-hit", HIT)
    staged_miss, _ = ctx.stage("reg-miss", MISS)
    ctx.expect(
        "reg/hit",
        staged_hit,
        hit_paths[Location.K1],
        runs=True,
        note="the registry must not override an exe-relative sysdir",
    )
    # Read-only, so this machine's install decides whether the registry's
    # docdir holds a usable base/: measured, not declared.
    ctx.expect(
        "reg/miss",
        staged_miss,
        windows_docdir(),
        runs=is_base_dir(windows_docdir()),
        note="registry/docdir applies when nothing was relocated",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--asy", required=True, help="path to the asy binary under test")
    ap.add_argument(
        "--asy-base-dir",
        required=True,
        dest="base_dir",
        help="the build tree's base/ directory (== settings.sysdir "
        "when asy is run in place)",
    )
    ap.add_argument(
        "--asy-ctan",
        default=None,
        help='path to the CTAN/TeXLive binary (enables the C="" axis)',
    )
    ap.add_argument(
        "--compiled-in",
        default=None,
        help="the compiled-in ASYMPTOTE_SYSDIR value; without it the "
        "fall-through rows and the C axis can only be observed, not asserted",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log every scenario in full, not just the failures",
    )
    return ap.parse_args()


def banner(ctx: Ctx) -> None:
    if not _Log.verbose:
        print("sysdir resolution")
        return
    print(f"binary:           {ctx.asy_under_test}")
    print(f"build-tree base:  {ctx.base_dir}")
    if ctx.compiled_in:
        exists = "exists" if os.path.exists(ctx.compiled_in) else "absent"
        print(f"compiled-in path: {ctx.compiled_in} ({exists})")
    print()
    print("candidate     -- K1 <exedir>/base")
    print("controls      -- K2 <exedir>/../share/asymptote, K3 <exedir>")
    print("staged states -- B = has plain.asy, D = decoy, A = absent")


def run_all(ctx: Ctx, asy_ctan: Optional[str]) -> None:
    banner(ctx)
    section("core matrix")
    run_core_matrix(ctx)

    section("route")
    run_route_axis(ctx)

    section("overrides")
    run_override_axis(ctx)

    # C and REG are one section: both are about what happens when no candidate
    # fires, and exactly one of them applies on any given OS.
    section("fallback")
    run_compiled_in_axis(ctx)
    run_registry_axis(ctx)

    section("ctan")
    if not asy_ctan:
        # Reported rather than passed over: a silently absent axis reads like a
        # passing one.
        record("ctan/*", Status.SKIP, "no --asy-ctan given")
    elif os.path.exists(asy_ctan):
        run_ctan_axis(ctx, asy_ctan)
    else:
        record("ctan/*", Status.SKIP, f"asy-ctan not found: {asy_ctan}")


def main() -> None:
    args = parse_args()
    _Log.verbose = args.verbose
    silence_error_dialogs()

    asy_under_test = os.path.abspath(args.asy)
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(asy_under_test):
        sys.exit(f"asy binary not found: {asy_under_test}")
    if not os.path.isdir(base_dir):
        sys.exit(f"base dir not found: {base_dir}")

    work = tempfile.mkdtemp(prefix="asy-relocatable-")
    # Point the children's config directory at the (config.asy-free) scratch
    # tree, before the first probe.
    _CHILD_ENV["ASYMPTOTE_HOME"] = work
    try:
        ctx = Ctx(
            asy_under_test=asy_under_test,
            base_dir=base_dir,
            compiled_in=(
                os.path.abspath(args.compiled_in) if args.compiled_in else None
            ),
            work=work,
        )
        run_all(ctx, os.path.abspath(args.asy_ctan) if args.asy_ctan else None)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    fails = [r for r in _results if r[1] is Status.FAIL]
    npass = sum(1 for r in _results if r[1] is Status.PASS)
    nskip = sum(1 for r in _results if r[1] is Status.SKIP)
    _Log.close_line()  # the last section's progress characters, if any
    print()
    print(f"summary: {npass} passed, {len(fails)} failed, {nskip} skipped")
    if fails:
        print("FAILED: " + ", ".join(r[0] for r in fails))
        sys.exit(1)


if __name__ == "__main__":
    main()
