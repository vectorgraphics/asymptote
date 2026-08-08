#!/usr/bin/env python3
"""Relocatable sysdir-resolution test matrix.

Exercises settings::resolveSysdir() (locate.cc) -- the logic that decides where
``asy`` looks for ``base/`` when no -dir is given -- by staging the built binary
into the layouts a real deployment produces and probing the resolved directory
with ``asy -c "write(settings.sysdir);"``.

That probe reports the resolved value even in the scenarios where asy cannot
start, by retrying through a known-good base/ -- see ``probe`` for why the exit
status alone is not a usable signal.

The derivation of the matrix is given below.  This file is pure Python stdlib
and runs on Linux, macOS and Windows.  The same script covers both the
ENABLE_RELOCATABLE=ON matrix and the OFF regression: callers say which one they
built with ``--mode on`` / ``--mode off``.  There is also a fallback,
``--mode auto``, which checks that the binary is consistent in whether it
behaves like a relocatable build or not.

The model under test
--------------------

The resolver, resolveSysdir(), is a first-match search over three *candidate
locations*, each accepted only if it contains plain.asy (isBaseDir,
locate.cc:76):

  K1  <exedir>/base                  build tree / flat install   always live
  K2  <exedir>/../share/asymptote    install tree                live iff RELOC
  K3  <exedir>                       flat layout (NSIS)          live iff RELOC

If none matches, the compiled-in ASYMPTOTE_SYSDIR is returned *unchanged and
unvalidated* (locate.cc:139).  Two further locations are often described as
candidates 4 and 5, but neither is selected by a plain.asy test:

  * the compiled-in path is the default, not a candidate -- its own state never
    changes which path is resolved, only whether the resolved path works;
  * the texmf tree (kpsewhich TEXMFMAIN, settings.cc:2022) is consulted iff the
    sysdir is *empty*, i.e. iff the compiled-in value is the empty string.

So they are not two independent locations but two settings of one variable, C,
the compiled-in sysdir.  That projection is what keeps the matrix small: the
selector reads exactly three booleans, so the layout space is 3x3x2 = 18 states
(K3 has no "absent" state -- the executable's own directory always exists) and
is enumerated exhaustively below rather than sampled.

Everything else is an axis that feeds, or post-processes, that same core:

  ROUTE   how asy was launched -- changes only the string getExecutablePath()
          returns, so it is tested once per route against one layout where a
          wrong exedir would be visible.
  OVR     -sysdir / -dir / ASYMPTOTE_SYSDIR, applied after resolution.
  C       the compiled-in sysdir: absent / valid / decoy / empty (CTAN).
          Tested against two layout rows: one where no candidate fires (C
          decides) and one where K1 fires (C must be ignored).
  REG     Windows only: queryRegistry() overwrites systemDir unless it was
          resolved relative to the executable (settings.cc:298).

Scenario IDs are ``core/<states>`` for the exhaustive rows -- B = contains
plain.asy, D = decoy (exists, no plain.asy), A = absent, in K1 K2 K3 order --
and ``<axis>/<case>`` elsewhere.

Requirements not scriptable in-suite -- a second build for the OFF matrix, a
compiled-in value of Windows' literal ``NUL``, a deployed texmf tree -- are
handled by a second invocation under ``--mode off``, or reported as SKIP rather
than failing the run.


Naming
------

Three different asy paths are in play, and which one a variable holds decides
what the assertion means, so they are named apart throughout:

  asy_under_test  the binary given on the command line (--asy, or --asy-ctan
                  for that axis).  Staging copies *from* it; the OVR axis is
                  the one place it is also run directly, since overrides act on
                  the result of resolution rather than on the layout.
  staged_asy      a copy placed into a staged layout -- the binary whose
                  <exedir> is the point of the scenario.  Where several are
                  live at once they are staged_hit / staged_miss, after the
                  layout each realizes.
  asy_to_run      the probe layer's parameter (_run_probe, probe, expect,
                  Ctx.expect): any executable to invoke.  It deliberately
                  accepts either of the above, which is why it is named after
                  neither.
"""

import argparse
import contextlib
import itertools
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

    A plain Enum rather than a ``str`` mixin: a mixin formats as ``PASS`` up to
    3.10 but as ``Status.PASS`` from 3.11 on, which would make the log depend on
    the interpreter.  Hence the explicit ``.value`` where it is printed.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


_results: List[Tuple[str, Status, str]] = []  # (scenario, status, message)


def record(scenario: str, status: Status, message: str = "") -> None:
    """Log one scenario outcome."""
    _results.append((scenario, status, message))
    print(f"  [{status.value}] {scenario:<18} {message}")


def norm(path: Optional[str]) -> str:
    """Canonical form for comparing two paths across OSes (case, symlinks, ..).

    An absent or empty path normalizes to "", so the result is always a string
    and two absent paths still compare equal.
    """
    if not path:
        return ""
    return os.path.normcase(os.path.realpath(path))


# ---------------------------------------------------------------------------
# the child environment
# ---------------------------------------------------------------------------

# Every setting is also readable as ASYMPTOTE_<NAME> (GetEnv, settings.cc:736)
# and an envSetting beats the resolved value, so an ASYMPTOTE_SYSDIR exported in
# the shell that started this script would be the answer to every probe: the
# whole matrix would collapse onto one value, and ovr/env would still pass since
# it sets that variable itself.  ASYMPTOTE_DIR is the same hazard one step
# removed -- it prepends to the import search path, which decides the second
# half of every assertion (whether asy runs at all).  Both are dropped from the
# environment handed to the children only, never from os.environ, so this
# process's own environment -- and hence the caller's, and anything else either
# may go on to launch -- is left exactly as it was found.
_STRIPPED = ("ASYMPTOTE_SYSDIR", "ASYMPTOTE_DIR")

# ASYMPTOTE_HOME is not stripped but redirected, in main(), once there is a
# scratch directory to point it at: dropping it would fall back to $HOME/.asy
# (settings.cc:2036), the *more* likely place to hold a config.asy.  That file
# cannot reach sysdir -- setOptions saves the value across the config load and
# restores it afterwards (settings.cc:2313) -- but it can set anything else,
# including dir, so the run is kept out of it either way.
_CHILD_ENV: Dict[str, str] = {k: v for k, v in os.environ.items() if k not in _STRIPPED}


def child_env(**overrides: str) -> Dict[str, str]:
    """The environment probes run in, plus any per-scenario additions.

    Callers that want one extra variable use this rather than
    ``dict(os.environ, X=...)``, which would carry the stripped ones back in.
    """
    return dict(_CHILD_ENV, **overrides)


def silence_error_dialogs() -> None:
    """Keep a child that cannot start from opening a modal dialog.

    Many scenarios deliberately stage a binary that will fail, and on Windows a
    failure to *start* -- a missing DLL, an unrunnable image -- is a loader
    "critical error", which by default pops a message box per launch and waits
    for someone to click it.  Under ctest nobody does: a regression that broke
    startup turned a matrix of ~30 probes into ~30 stuck dialogs.  The error
    mode is inherited by child processes, so setting it once here covers every
    probe without changing how any of them is invoked.

    This suppresses only the *display*; the failure is still reported through
    the child's exit status, which is what the scenarios read.
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
    """Mirror of isBaseDir() (locate.cc:76): a directory counts as a base
    directory only if plain.asy is in it.  Mere existence proves nothing."""
    if not path:  # None, and also "" -- the empty TeXLive sysdir
        return False
    return os.path.isfile(os.path.join(path, "plain.asy"))


def launch_target(
    asy_to_run: str, cwd: Optional[str], env: Mapping[str, str]
) -> Optional[str]:
    """The absolute path behind ``asy_to_run``, or None if it cannot be found.

    On Windows this is the path the OS is told to start; on POSIX it serves
    only as a reachability check, for the reason _run_probe gives.

    The ROUTE axis names the executable the awkward ways a user can -- by a
    relative path, or by a bare name on PATH -- and on Windows neither is
    resolved the way the call reads, because both lookups belong to
    CreateProcess, which consults neither argument subprocess passes for them:

      * a relative program path is resolved against *this* process's working
        directory, not the ``cwd=`` handed to the child (CreateProcess has no
        notion of the latter when it looks the program up), so ``./asy.exe``
        with ``cwd=<staged>`` failed with WinError 2;
      * a bare name is searched on the *parent's* PATH, not the ``env=`` one,
        so route/path silently launched whichever asy was already installed on
        the developer's PATH -- it reported that binary's sysdir and failed
        against an expectation about the staged one.

    The second is the dangerous one: it fails by testing the wrong program
    rather than by not running.  So on Windows the route is resolved here and
    passed to subprocess as ``executable=``, while ``asy_to_run`` stays argv[0]
    -- which is what the route axis is varying.  Substituting it there costs
    the axis nothing: GetModuleFileNameA (locate.cc:74) reports the path the
    loader recorded for the image, which no spelling of the program argument
    changes.  The symlink survives resolution, and deliberately so -- neither
    abspath nor which() follows one, so route/symlink still hands asy the link
    to resolve for itself (locate.cc:83).

    POSIX needs none of this: subprocess forks, chdirs to ``cwd=`` and execs
    there, so a relative program resolves the way the call reads, and a bare
    name is looked up on the ``env=`` PATH (Popen builds its candidate list
    from os.get_exec_path(env)).  There the resolved path is used only to tell
    "not on the child PATH" apart from the ways a found binary can fail.
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
    failure error is the (trimmed) stderr.

    ``env`` of None means the sanitized default, not "inherit": the child never
    sees a raw os.environ, since the variables removed from it are exactly the
    ones that would answer the question being asked.

    ``executable=`` is substituted on Windows only, where launch_target explains
    why it is needed and why it is free.  It is withheld on POSIX because there
    it would not be free: getExecutablePath() asks the OS which image is
    running, and on two platforms the answer cannot see how the program was
    named -- GetModuleFileNameA reports the loader's recorded path, and
    /proc/self/exe is a kernel symlink that is already fully resolved.  macOS is
    the exception.  _NSGetExecutablePath (locate.cc:88) hands back the string
    that was passed to execve, which is exactly what ``executable=`` replaces,
    so substituting a resolved path would leave the realpath() call at
    locate.cc:100 with nothing relative left to resolve: route/relative and
    route/symlink-rel would still pass, but they would no longer be testing the
    branch they exist for.
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
      resolved  the resolved sysdir whenever we could obtain it -- including
                from a failing run -- or None if even that was impossible.
                Note "" (an empty sysdir, which is what a TeXLive build reports
                when no candidate matched) is a value, not a failure.

    The command is executed by plain.asy itself (base/plain.asy:11), so a run
    that cannot load plain prints nothing at all and we would learn only that
    it failed.  That is too coarse: resolveSysdir() never fails -- when no
    candidate matches it returns the compiled-in path unchanged (locate.cc:139),
    which may not exist -- so "asy exited non-zero" conflates "no sysdir" with
    "a sysdir that isn't usable" and with "failed for some unrelated reason".

    The ``rescue_base`` parameter closes that gap: on failure we retry with
    ``-dir <rescue_base>``, which feeds plain from a known-good directory
    without altering settings.sysdir (this is what ovr/dir asserts), so the
    failing run's resolved value is recovered.  Passing -noautoplain instead
    would not work -- it suppresses the -c command entirely and exits 0
    silently.
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

    Two-part assertion, because resolveSysdir() cannot report failure (see
    ``probe``): the resolved path must be the predicted one, *and* asy must run
    exactly when the callsite says it will.  Checking only the exit status would
    score a PASS for any unrelated startup failure, and would report a
    compiled-in path that merely does not exist as "resolution failed" when in
    fact resolution returned it.

    ``runs`` is stated at the callsite rather than derived here, so that reading
    a scenario tells you which way its second half points without first working
    out whether ``expected`` was staged with plain.asy.  It is still checked
    against is_base_dir(expected) -- the model being that a resolved sysdir
    containing plain.asy must work and one without it must not -- and a
    disagreement is reported as a test bug, since it means the scenario did not
    stage what its callsite claims.  The rows whose answer genuinely depends on
    the environment rather than on staging (the fall-through rows, and the C
    axis run against a pre-existing install) pass ``runs=is_base_dir(...)``,
    which names that dependency at the callsite instead of hiding it here.

    One row does already sit outside the model: ovr/dir passes ``-dir`` at a
    usable base/, so its success is licensed by that argument rather than by the
    resolved sysdir, and it satisfies the cross-check only because the sysdir it
    resolves is a real base/ too.  Any future row that pairs ``-dir`` with a
    deliberately unusable resolved sysdir would need a way to opt out of that
    cross-check, ``runs`` alone being unable to express it.

    ``**kw`` is forwarded verbatim to ``probe``; PEP 692 typed kwargs are far
    newer than 3.7, so it can only be spelled ``Any`` here.  The accepted keys
    are therefore exactly ``probe``'s keyword parameters, restated:

      cwd          Optional[str]       directory to run asy in; None = inherit
                                       this process's cwd.
      env          Optional[Mapping[str, str]]
                                       full environment for the child; None =
                                       the sanitized default (see child_env).
                                       It replaces rather than extends, so
                                       callers adding one variable pass
                                       child_env(X=...) -- not dict(os.environ,
                                       X=...), which would undo the sanitizing.
      extra_args   Sequence[str]       arguments inserted before the probe's own
                                       ``-c``; default ().
      rescue_base  Optional[str]       base/ to retry through with ``-dir`` when
                                       the run fails, so the resolved sysdir is
                                       still recovered; None = do not retry, in
                                       which case a failing run yields
                                       resolved=None and this function FAILs
                                       with "value could not be recovered".

    This list is the fragile part: nothing checks it against ``probe``, and a
    misspelled key here becomes a TypeError only when that branch runs.  If it
    looks stale, ``probe``'s signature is the authority -- and note that
    ``asy_to_run`` is *not* forwardable, since it is passed positionally from
    this function's own ``asy_to_run`` parameter.
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

    # "B" is for base directory: the letters are the ones the core scenario ID
    # triples are written in, so this one is not the member's own initial the
    # way D and A happen to be.
    VALID = "B"
    DECOY = "D"
    ABSENT = "A"

    @property
    def label(self) -> str:
        """The spelled-out name, used in the C-axis scenario IDs."""
        return self.name.lower()


class Candidate(IntEnum):
    """The locations resolveSysdir() searches, in resolution order.

    The value is the position in a States or Paths triple, so a Candidate
    subscripts either one directly.
    """

    K1 = 0
    K2 = 1
    K3 = 2

    @property
    def where(self) -> str:
        """Human-readable location, for the scenario notes."""
        return {
            Candidate.K1: "<exedir>/base",
            Candidate.K2: "<exedir>/../share/asymptote",
            Candidate.K3: "<exedir>",
        }[self]


# One point of the layout space: the state of (K1, K2, K3), in resolution
# order.  Always exactly three -- the selector reads three locations, and
# everything downstream indexes all three positionally.
States = Tuple[State, State, State]

# The three candidate locations as staged paths, in the same resolution order,
# so Paths[i] is the directory whose state is States[i].
Paths = Tuple[str, str, str]


def copy_base_into(base_dir: str, dst: str, with_plain: bool = True) -> None:
    """Copy the base files into ``dst``, creating it if need be.

    Passing ``with_plain=False`` builds a decoy: everything except plain.asy,
    which is the only file resolveSysdir() looks for.  Unlike copytree this
    tolerates an existing destination (the flat layout copies base files in
    beside asy) -- copytree's dirs_exist_ok is 3.8+.
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
# Being able to carry its own runtime this way is part of what "relocatable"
# means: the NSIS install ships these next to asy.exe, a macOS bundle puts
# .dylibs next to the binary (or at @executable_path/../lib), and an $ORIGIN
# rpath does the same for .so files.  So they are staged along with the binary
# rather than left behind -- otherwise the copy is not the deployment shape the
# matrix claims to be testing, and on Windows it does not start at all
# (STATUS_DLL_NOT_FOUND, 0xC0000135, before main() and before any sysdir logic).
#
# macOS is listed with both suffixes: dyld loads either.
_LIB_SUFFIXES = {"win32": (".dll",), "darwin": (".dylib", ".so")}
_lib_suffixes: Tuple[str, ...] = _LIB_SUFFIXES.get(sys.platform, (".so",))


def is_bundled_lib(name: str) -> bool:
    """Is ``name`` a shared library that travels with the executable?

    Matched on suffix rather than an exact list so that a build that gains or
    loses a dependency needs no change here.  The ``.so`` case also accepts the
    versioned ``libfoo.so.1.2`` spelling, hence the ``.so.`` test.
    """
    lower = name.lower()
    if lower.endswith(_lib_suffixes):
        return True
    if ".so" in _lib_suffixes and ".so." in lower:
        return True
    return False


def copy_bundled_libs(asy_under_test: str, dst_dir: str) -> None:
    """Copy the shared libraries bundled beside ``asy_under_test`` into dst_dir.

    They are the executable's own runtime, not part of the layout under test:
    none of them is plain.asy, so staging them cannot change which candidate
    resolveSysdir() selects, only whether the copy gets far enough to report
    one.
    """
    os.makedirs(dst_dir, exist_ok=True)
    srcdir = os.path.dirname(asy_under_test)
    for name in os.listdir(srcdir):
        src = os.path.join(srcdir, name)
        if is_bundled_lib(name) and os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


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

    The layout is <root>/bin/asy, so all three candidate locations exist as
    distinct paths under <root> and can be populated independently.

    Returns (staged_asy, paths) with paths in resolution order, so paths[i] is
    what asy must report when candidate i wins.
    """
    bindir = os.path.join(root, "bin")
    staged_asy = stage_binary(bindir, asy_under_test)
    paths = (
        os.path.join(bindir, "base"),
        os.path.join(root, "share", "asymptote"),
        bindir,
    )
    for k in (Candidate.K1, Candidate.K2):
        materialize(paths[k], states[k], base_dir)
    # K3 is the executable's own directory: it always exists (the staged binary
    # is in it), so ABSENT is unreachable and DECOY means "populated but no
    # plain.asy" rather than "empty".
    assert states[Candidate.K3] is not State.ABSENT, "<exedir> cannot be absent"
    copy_base_into(base_dir, bindir, with_plain=states[Candidate.K3] is State.VALID)
    return staged_asy, paths


def winner(states: States, relocatable: bool) -> Optional[Candidate]:
    """The oracle: the candidate resolveSysdir() must select, or None if it
    must fall through to the compiled-in value.  First match wins, over the
    candidates the build has live."""
    live = tuple(Candidate) if relocatable else (Candidate.K1,)
    for i in live:
        if states[i] is State.VALID:
            return i
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
    first = rel.split(os.sep)[0]
    return os.path.join(anc, first)


@contextlib.contextmanager
def temporary_tree(path: str) -> Generator[None, None, None]:
    """Let the body materialize ``path``, then remove it again on the way out.

    The rollback target is computed on entry, while ``path`` is still absent --
    that is what makes it the topmost *newly created* directory rather than a
    pre-existing one.  So the body must create ``path``, not the caller."""
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

    On Windows initSettings() ends with ``systemDir = docdir`` unless the
    sysdir came from the executable (settings.cc:298), and docdir is the
    registry's App Paths\\Asymptote entry, or a hard-coded default that is never
    empty (settings.cc:143).  So on Windows the fall-through value is this, not
    the compiled-in path.  An unreadable registry yields that same default,
    which mirrors what settings.cc does rather than reporting a failure.
    """
    default = "c:\\Program Files\\Asymptote"
    if sys.platform != "win32":
        return default
    # Guarded on sys.platform rather than ImportError so that a type checker
    # can see winreg is only touched where it exists.  The cost is that mypy
    # prunes everything below on any other platform -- an undefined winreg name
    # here passes a Linux run -- so misc-sanity-checks.yml types this file a
    # second time with --platform=win32, which needs no Windows host.
    # Disabling import-error is deliberate: winreg does not exist off Windows.
    import winreg  # pylint: disable=import-outside-toplevel,import-error

    key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Asymptote"
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, key) as handle:
                value, _ = winreg.QueryValueEx(handle, "Path")
            # QueryValueEx is typed Any: the key could hold a REG_DWORD, and
            # returning that would quietly violate this function's contract.
            if isinstance(value, str) and value:
                return value
        except OSError:
            continue
    return default


def fallback_sysdir(compiled_in: Optional[str]) -> Optional[str]:
    """What asy must report when no candidate fires, or None if unpredictable.

    Off Windows that is the compiled-in value verbatim -- so we can only
    predict it when the caller told us what it is.  A compiled-in value of ""
    (the CTAN/TeXLive build) is not handled here: it hands off to kpsewhich,
    which is checked separately in the C axis.
    """
    if os.name == "nt":
        return windows_docdir()
    return compiled_in


# ---------------------------------------------------------------------------
# the core matrix: every reachable (K1, K2, K3) state
# ---------------------------------------------------------------------------


class Ctx(NamedTuple):
    """Everything the checks need that is fixed for the whole run."""

    asy_under_test: str  # the binary under test
    base_dir: str  # its build-tree base/, used as the -dir rescue
    compiled_in: Optional[str]  # ASYMPTOTE_SYSDIR, if the caller named it
    work: str  # scratch root; also the cwd every probe runs in
    relocatable: bool  # whether K2 and K3 are live

    def stage(
        self, name: str, states: States, asy_under_test: Optional[str] = None
    ) -> Tuple[str, Paths]:
        """Stage layout ``states`` under <work>/<name>; see stage_layout.

        ``asy_under_test`` overrides the run's binary for this one layout; the
        CTAN axis uses it to stage a *different* binary into the same shapes.
        """
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

        ``runs`` -- whether asy is expected to start once it has resolved
        ``expected`` -- is required, and named at the callsite rather than
        inferred from the staged layout; see the module-level ``expect``.

        ``**kw`` is forwarded to the module-level ``expect``, so it accepts that
        function's remaining keyword parameters plus the ones it in turn
        forwards to ``probe``:

          note         str              parenthetical appended to the record.
          cwd          Optional[str]    defaults to ctx.work here, not to None.
          env          Optional[Mapping[str, str]]
                                        full child environment; None = the
                                        sanitized default (see child_env).
          extra_args   Sequence[str]    arguments placed before the probe's -c.
          rescue_base  Optional[str]    defaults to ctx.base_dir here; pass None
                                        explicitly to forgo the -dir retry (only
                                        ovr/dir does, since it is what licenses
                                        that retry in the first place).

        Fragile parts, in order: the two setdefault keys above must stay spelled
        exactly as ``probe``'s parameters -- a typo would silently add an unused
        key and drop the intended default rather than error -- and the list
        itself is a hand-kept copy of two signatures.  Check it against
        ``expect`` and ``probe``; ``scenario``, ``asy_to_run`` and ``expected``
        are passed positionally, and ``runs`` explicitly, so none of the four
        can appear here.
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
    idx = winner(states, ctx.relocatable)
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

    No representatives, no pruning: after projecting out the compiled-in value
    and the texmf tree (neither is selected by a plain.asy test) the selector
    reads exactly three locations, and <exedir> cannot be absent.  Enumerating
    the space outright is cheaper than arguing about which rows would suffice,
    and it *measures* rather than assumes both the first-match ordering and the
    equivalence of a decoy directory with an absent one.
    """
    for states in itertools.product(State, State, (State.VALID, State.DECOY)):
        check_layout(ctx, "core/" + state_code(states), states)


# ---------------------------------------------------------------------------
# ROUTE: how the executable was launched
# ---------------------------------------------------------------------------


def layout_for_route(relocatable: bool) -> States:
    """A layout in which the answer depends on where the executable is.

    Relocatable builds use the install tree (K2), the shape a packaged binary
    actually has; an OFF build resolves only K1, so use that.  Either way a
    wrong exedir yields a visibly different path rather than the same one.
    """
    if relocatable:
        return (State.ABSENT, State.VALID, State.DECOY)
    return (State.VALID, State.ABSENT, State.DECOY)


def run_route_axis(ctx: Ctx) -> None:
    """Every way of naming the executable must resolve the same layout.

    The only per-OS part of the mechanism is getExecutablePath(), and it is the
    only thing these cases vary; one discriminating layout is therefore enough
    per route.  ROUTE x layout would multiply the matrix by 6 and prove nothing
    the core matrix does not already prove.
    """
    states = layout_for_route(ctx.relocatable)
    root = os.path.join(ctx.work, "route")
    staged_asy, paths = ctx.stage("route", states)
    idx = winner(states, ctx.relocatable)
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
        # The bundled runtime does not follow a symlink on Windows: the loader
        # takes the *link's* directory as the application directory for the DLL
        # search, so a link into a bundled install cannot start (0xC0000135)
        # unless the libraries are reachable from the link too.  Staging them
        # beside it keeps the row asserting rather than skipped, and cannot
        # blunt it -- link_dir still holds no base/, so an exedir that stayed
        # here resolves the absent <exedir>/base and reports a visibly
        # different sysdir.  A no-op on platforms that resolve the link first.
        copy_bundled_libs(ctx.asy_under_test, link_dir)
        # Homebrew/MacPorts shape.  Linux resolves the link before asy can see
        # it (/proc/self/exe), so the platforms this row is for are the two that
        # do not: macOS, where _NSGetExecutablePath() reports the link and
        # realpath() (locate.cc:100) is what resolves it, and Windows, where
        # GetModuleFileNameA() reports the link and canonicalPath()
        # (locate.cc:83) is.
        ctx.expect("route/symlink", os.path.join(link_dir, name), expected, runs=True)
        # ... and reached by a relative path, where _NSGetExecutablePath() hands
        # the relative string back verbatim and realpath() has to resolve it
        # against the child's cwd.  It arrives in that form only because
        # _run_probe withholds executable= on POSIX.
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
    # Done last -- it invalidates the symlink and the PATH entry above.  Both
    # paths below are the old ones re-rooted at `moved`; relpath is purely
    # lexical, so it does not matter that staged_asy no longer exists.
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

    Each override is tested against a value that differs from what resolution
    would produce on its own, so "override applied" and "override ignored" are
    distinguishable; pointing an override at the already-resolved directory
    would pass either way.  Layout is held fixed: these act on the result of
    resolution, not on its inputs.
    """
    alt = os.path.join(ctx.work, "altbase")
    materialize(alt, State.VALID, ctx.base_dir)
    bogus = os.path.join(ctx.work, "does-not-exist")

    # These run the binary in place rather than a staged copy: the overrides act
    # on the result of resolution, so the layout it resolves from is irrelevant.
    # -sysdir replaces the resolved value outright, valid or not: pointed at a
    # real base/ asy runs, pointed at a nonexistent one it cannot start, and
    # both are equally an override having been applied.
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

    # -dir only prepends to the import search path.  This is what licenses the
    # rescue -dir that probe() uses everywhere else, so it takes no rescue
    # itself -- and it points at a *different* base/ than the resolved one.
    ctx.expect(
        "ovr/dir",
        ctx.asy_under_test,
        ctx.base_dir,  # sysdir, not dir
        runs=True,
        extra_args=("-dir", alt),
        rescue_base=None,
    )

    # The environment variable overrides everything (envSetting, settings.cc).
    # This is the one row that puts ASYMPTOTE_SYSDIR back: every other probe
    # runs without it, which is what makes this row's result attributable.
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
# the fallback must be ignored whatever state it is in.  Every other layout is
# equivalent to one of these two as far as those variables are concerned, which is what
# makes these axes |values| x 2 rather than |values| x 18.  They are named for what
# the candidate search does, not for what happens afterwards.
MISS = (State.ABSENT, State.ABSENT, State.DECOY)
HIT = (State.VALID, State.ABSENT, State.DECOY)


def run_compiled_in_axis(ctx: Ctx) -> None:
    """C = absent / valid / decoy, at MISS and at HIT.

    We only mutate the compiled-in path when that is safe: the caller named it,
    it does not already exist (never delete a real install), and its parent is
    writable.  Otherwise we report what the environment happens to provide.
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
        # the same measurement that picked the label.
        state = State.VALID if is_base_dir(compiled_in) else State.DECOY
        ctx.expect(
            f"C/{state.label}-miss",
            staged_miss,
            compiled_in,
            runs=state is State.VALID,
        )
        ctx.expect(
            f"C/{state.label}-hit", staged_hit, hit_paths[Candidate.K1], runs=True
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
    ctx.expect("C/absent-hit", staged_hit, hit_paths[Candidate.K1], runs=True)

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
                f"C/{state.label}-hit", staged_hit, hit_paths[Candidate.K1], runs=True
            )


def run_ctan_axis(ctx: Ctx, asy_ctan: str) -> None:
    """C = "" -- the CTAN/TeXLive binary, which defers to kpsewhich.

    Same two layouts: at HIT the adjacent base/ must still win (kpsewhich is the
    *last* resort, not a mode), and at MISS the texmf tree answers -- and in
    particular the answer must not be a build or staging path.
    """
    staged_hit, hit_paths = ctx.stage("ctan-hit", HIT, asy_under_test=asy_ctan)
    ctx.expect("ctan/hit", staged_hit, hit_paths[Candidate.K1], runs=True)

    # Deployed TeXLive shape: bin/<platform>/asy, no adjacent base/.  On a host
    # whose texmf tree has no asymptote/ directory the binary cannot start; the
    # rescue -dir reports what it resolved anyway, so this is checkable either
    # way.  An empty sysdir (no kpsewhich answer at all) is a legitimate
    # outcome, hence the `is None` test: only an unlaunchable binary is
    # unmeasurable.
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
    machine, so this reports rather than arranges.  Skipped off Windows, where
    queryRegistry() is not compiled in at all.
    """
    if os.name != "nt":
        record("reg/*", Status.SKIP, "Windows only")
        return
    staged_hit, hit_paths = ctx.stage("reg-hit", HIT)
    staged_miss, _ = ctx.stage("reg-miss", MISS)
    ctx.expect(
        "reg/hit",
        staged_hit,
        hit_paths[Candidate.K1],
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


def detect_relocatable(asy_under_test: str, base_dir: str, work: str) -> bool:
    """Empirically decide whether the binary was built relocatable.

    Stage an install-tree layout (K2, gated on IS_RELOCATABLE) and see whether
    it resolves.  This avoids having to plumb the build flag through
    configure/CMake -- the binary's own behaviour is the source of truth.

    Takes the three fields it needs rather than a Ctx: its answer is what
    fixes Ctx.relocatable, so it has to run before there is a Ctx to pass.
    """
    root = os.path.join(work, "detect")
    states = (State.ABSENT, State.VALID, State.DECOY)
    staged_asy, paths = stage_layout(root, asy_under_test, base_dir, states)
    _, _, resolved = probe(staged_asy, rescue_base=base_dir)
    shutil.rmtree(root, ignore_errors=True)
    return norm(resolved) == norm(paths[Candidate.K2])


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
        "--mode",
        choices=("auto", "on", "off"),
        default="auto",
        help="relocatable mode; 'auto' (default) falls back to probing the binary",
    )
    return ap.parse_args()


def banner(ctx: Ctx) -> None:
    live = "K1, K2, K3" if ctx.relocatable else "K1 only (K2, K3 gated off)"
    print(f"relocatable:      {'on' if ctx.relocatable else 'off'} -- live: {live}")
    print(f"binary:           {ctx.asy_under_test}")
    print(f"build-tree base:  {ctx.base_dir}")
    if ctx.compiled_in:
        exists = "exists" if os.path.exists(ctx.compiled_in) else "absent"
        print(f"compiled-in path: {ctx.compiled_in} ({exists})")
    print()
    print("core matrix -- K1 <exedir>/base, K2 <exedir>/../share/asymptote,")
    print("               K3 <exedir>; B = has plain.asy, D = decoy, A = absent")


def run_all(ctx: Ctx, asy_ctan: Optional[str]) -> None:
    banner(ctx)
    run_core_matrix(ctx)

    print()
    run_route_axis(ctx)

    print()
    run_override_axis(ctx)

    print()
    run_compiled_in_axis(ctx)
    run_registry_axis(ctx)

    print()
    if not asy_ctan:
        # Reported rather than passed over: whether the C="" axis ran is a
        # property of how the caller was configured, not of this run, and a
        # silently absent axis reads like a passing one.
        record("ctan/*", Status.SKIP, "no --asy-ctan given")
    elif os.path.exists(asy_ctan):
        run_ctan_axis(ctx, asy_ctan)
    else:
        record("ctan/*", Status.SKIP, f"asy-ctan not found: {asy_ctan}")


def main() -> None:
    args = parse_args()
    silence_error_dialogs()

    asy_under_test = os.path.abspath(args.asy)
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(asy_under_test):
        sys.exit(f"asy binary not found: {asy_under_test}")
    if not os.path.isdir(base_dir):
        sys.exit(f"base dir not found: {base_dir}")

    work = tempfile.mkdtemp(prefix="asy-relocatable-")
    # Point the children's configuration directory at the (config.asy-free)
    # scratch tree, before the first probe -- detect_relocatable runs one.
    _CHILD_ENV["ASYMPTOTE_HOME"] = work
    try:
        if args.mode == "auto":
            relocatable = detect_relocatable(asy_under_test, base_dir, work)
        else:
            relocatable = args.mode == "on"
        ctx = Ctx(
            asy_under_test=asy_under_test,
            base_dir=base_dir,
            compiled_in=(
                os.path.abspath(args.compiled_in) if args.compiled_in else None
            ),
            work=work,
            relocatable=relocatable,
        )
        run_all(ctx, os.path.abspath(args.asy_ctan) if args.asy_ctan else None)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    fails = [r for r in _results if r[1] is Status.FAIL]
    npass = sum(1 for r in _results if r[1] is Status.PASS)
    nskip = sum(1 for r in _results if r[1] is Status.SKIP)
    print()
    print(f"summary: {npass} passed, {len(fails)} failed, {nskip} skipped")
    if fails:
        print("FAILED: " + ", ".join(r[0] for r in fails))
        sys.exit(1)


if __name__ == "__main__":
    main()
