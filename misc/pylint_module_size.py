"""A pylint plugin gating module size by statement count rather than by lines.

Pylint's built-in size gate for a module is ``too-many-lines`` (C0302), and it
counts raw physical lines: comments, docstrings and blank lines all tell against
the limit (pylint/checkers/format.py:442 does ``line_num -= 1  # to be ok with
"wc -l"``, with no filtering by token type).  A heavily documented file can
therefore trip it while containing very little code, which is the opposite of
what a complexity limit is for -- it penalizes exactly the thing we want.

This checker adds ``too-many-module-statements`` (R9001), which counts
*statements* in the module and so is blind to prose.  Enable it and turn
``too-many-lines`` off (see .pylintrc) to gate on code volume alone.

The count mirrors pylint's own ``too-many-statements`` (R0915) accounting for
functions -- same ``node.is_statement`` test, driven from the same
``visit_default`` hook (pylint/checkers/design_analysis.py:642) -- so a module's
number is directly comparable to the per-function limit, and to the statement
total that ``pylint --reports=y`` prints.  The one deliberate difference is that
bare string-constant statements are not counted: on an astroid tree a docstring
is normally lifted out of the body into ``doc_node``, but a string used as a
comment in statement position is prose either way and must not count as code.

Loaded via ``load-plugins`` in .pylintrc, which resolves it as an import, so
pylint must run with the repository root on sys.path.  It is *not* there by
default: pylint 3 does not add the current directory, and ``python3 -m pylint``
does not either, since pylint rewrites sys.path itself.  The ``init-hook`` in
.pylintrc puts it back, which makes the whole arrangement depend on pylint being
run from the repository root -- the same assumption the relative
``--rcfile=.pylintrc`` in .github/workflows/misc-sanity-checks.yml already makes.
Symptom if that breaks: ``E0013 bad-plugin-value ... No module named 'misc'``,
followed by every option below being reported as unrecognized.
"""

from typing import TYPE_CHECKING, List

from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH

if TYPE_CHECKING:  # pragma: no cover -- import cycle at runtime, fine for typing
    from pylint.lint import PyLinter

# Annotations use the 3.7-compatible spelling (typing aliases rather than
# builtin generics or PEP 604 unions) because pylint lints this file against
# py-version=3.7 like every other script here -- not because it runs there: as a
# pylint plugin it runs on whatever interpreter runs pylint, which for pylint 3.3
# is 3.9+.
#
# Note for anyone adding a type checker over this tree later: this file is a bad
# candidate to check at that 3.7 floor.  A checker follows the import into
# pylint's own source, which uses 3.8+ syntax ("Assignment expressions are only
# supported in Python 3.8 and greater" in checkers/utils.py), and astroid ships
# no py.typed, so `from astroid import nodes` is untyped either way.


def is_prose_statement(node: nodes.NodeNG) -> bool:
    """True for a statement that is just a string literal.

    Astroid normally lifts a docstring out of the body into ``doc_node``, so this
    mostly catches strings used as block comments.  Either way the content is
    prose, and counting it would reintroduce the bias this checker exists to
    remove.
    """
    return (
        isinstance(node, nodes.Expr)
        and isinstance(node.value, nodes.Const)
        and isinstance(node.value.value, str)
    )


class ModuleSizeChecker(BaseChecker):
    """Report modules whose statement count exceeds max-module-statements."""

    name = "module-size"

    msgs = {
        "R9001": (
            "Too many statements in module (%d/%d)",
            "too-many-module-statements",
            "Used when a module has too many statements, counting neither "
            "comments nor docstrings.  A prose-blind replacement for "
            "too-many-lines (C0302), which counts raw physical lines.",
        ),
    }

    options = (
        (
            "max-module-statements",
            {
                "default": 500,
                "type": "int",
                "metavar": "<int>",
                "help": "Maximum number of statements in a module.  Counted as "
                "for too-many-statements, so comparable to max-statements, but "
                "ignoring comments and docstrings.",
            },
        ),
    )

    def __init__(self, linter: "PyLinter") -> None:
        super().__init__(linter)
        # A stack, not a scalar: nested modules do not occur, but a stack costs
        # nothing and keeps the count well defined if a walk is ever re-entered.
        self._statements: List[int] = []

    def visit_module(self, _node: nodes.Module) -> None:
        """Start counting for a new module."""
        self._statements.append(0)

    def visit_default(self, node: nodes.NodeNG) -> None:
        """Count every statement, except one that is only a string literal."""
        if not self._statements:  # a node visited outside any module
            return
        if node.is_statement and not is_prose_statement(node):
            self._statements[-1] += 1

    def leave_module(self, node: nodes.Module) -> None:
        """Report the module if it is over the limit."""
        count = self._statements.pop()
        limit = self.linter.config.max_module_statements
        if count > limit:
            self.add_message(
                "too-many-module-statements",
                node=node,
                args=(count, limit),
                confidence=HIGH,
            )


def register(linter: "PyLinter") -> None:
    """Entry point used by pylint's load-plugins."""
    linter.register_checker(ModuleSizeChecker(linter))
