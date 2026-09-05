"""A pylint plugin gating module size by statement count rather than by lines.

Pylint's built-in gate, ``too-many-lines`` (C0302), counts raw physical lines,
so comments and docstrings tell against it: a heavily documented file can trip
it while containing very little code.  This checker adds
``too-many-module-statements`` (R9001), which counts *statements* and so is
blind to prose.  Enable it and turn ``too-many-lines`` off (see .pylintrc).

The count mirrors pylint's own ``too-many-statements`` (R0915) accounting for
functions -- same ``node.is_statement`` test, same ``visit_default`` hook -- so
a module's number is comparable to the per-function limit.  The one deliberate
difference is that bare string-constant statements do not count; see
is_prose_statement.

``load-plugins`` resolves this as an import, so pylint must run with the
repository root on sys.path, which it does not add by default.  The
``init-hook`` in .pylintrc puts it back, making the arrangement depend on pylint
being run from the repository root -- the same assumption the relative
``--rcfile=.pylintrc`` in CI already makes.  Symptom if that breaks: ``E0013
bad-plugin-value ... No module named 'misc'``.
"""

from typing import TYPE_CHECKING, List

from astroid import nodes
from pylint.checkers import BaseChecker
from pylint.interfaces import HIGH

if TYPE_CHECKING:  # pragma: no cover -- import cycle at runtime, fine for typing
    from pylint.lint import PyLinter

# Annotations use the 3.7-compatible spelling because pylint lints this file
# against py-version=3.7 like every other script here -- not because it runs
# there; as a plugin it runs on whatever interpreter runs pylint.  Do not add it
# to mypy.ini at that floor: a checker follows the import into pylint's own
# source, which uses 3.8+ syntax, and astroid ships no py.typed anyway.


def is_prose_statement(node: nodes.NodeNG) -> bool:
    """True for a statement that is just a string literal.

    Astroid lifts a docstring out of the body into ``doc_node``, so this mostly
    catches strings used as block comments.  Either way it is prose, and
    counting it would reintroduce the bias this checker exists to remove.
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
        # A stack, so the count stays well defined if a walk is re-entered.
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
