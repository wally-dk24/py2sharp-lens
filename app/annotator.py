"""Deterministic Python-code annotator.

Walks the abstract syntax tree (built by the stdlib ``ast`` module) and
detects the constructs listed in ``catalog.json``.  This is the "eyes" of
Py2Sharp Lens: everything it finds becomes a learning note, without any
LLM involved.

A C# reader can think of ``ast`` like Roslyn's syntax tree API:
``ast.parse(source)`` is ``CSharpSyntaxTree.ParseText(source)`` and each
node type (``ast.For``, ``ast.BinOp``, ...) is a ``SyntaxNode`` subclass.
``NodeVisitor`` dispatches on method names: defining ``visit_For`` means
"call me for every ``for`` statement", just like overriding
``VisitForStatement`` in a Roslyn ``CSharpSyntaxWalker``.
"""

from __future__ import annotations

import ast
from typing import Any, Optional


class Finding(dict):
    """A single detected construct.

    Behaves like a plain dict so it serialises to JSON trivially, but the
    class gives us a named type for documentation.  Keys:
    ``line`` (1-based), ``end_line`` (1-based, inclusive) and
    ``construct_key`` (matches a key in catalog.json).
    """


def _add(
    findings: list[Finding],
    node: ast.AST,
    construct_key: str,
    line: Optional[int] = None,
) -> None:
    """Record one finding, defaulting to the node's own source position."""
    lineno = line if line is not None else getattr(node, "lineno", 1)
    end_lineno = getattr(node, "end_lineno", None) or lineno
    findings.append(
        Finding(line=lineno, end_line=end_lineno, construct_key=construct_key)
    )


def _decorator_name(decorator: ast.expr) -> str:
    """Best-effort dotted name of a decorator expression.

    ``@dataclass``      -> "dataclass"
    ``@dataclasses.dataclass`` -> "dataclasses.dataclass"
    ``@property``       -> "property"
    Anything fancier (e.g. ``@app.route("/x")``) -> "" (we only need the
    simple cases for catalog matching).
    """
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Call):
        # like C# reading attribute usage: unwrap @foo(...) to foo
        return _decorator_name(decorator.func)
    return ""


class Annotator(ast.NodeVisitor):
    """ast.NodeVisitor that collects construct findings.

    Method naming rule: ``visit_<NodeClassName>``.  Each visitor records
    its finding(s) and then calls ``self.generic_visit(node)`` so the walk
    continues into child nodes — forgetting that call would prune the
    whole subtree (a common beginner bug).
    """

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        # Stack of enclosing nodes, used for context-sensitive checks
        # (e.g. "is this lambda inside a loop?" for LateBinding).
        self._stack: list[ast.AST] = []

    # -- plumbing ------------------------------------------------------
    def visit(self, node: ast.AST) -> Any:
        """Push/pop the ancestor stack around the normal dispatch."""
        self._stack.append(node)
        try:
            return super().visit(node)
        finally:
            self._stack.pop()

    def _inside(self, *types: type) -> bool:
        """True if any enclosing node (excluding self) is of the given types."""
        return any(isinstance(n, types) for n in self._stack[:-1])

    # -- comprehensions & generators -----------------------------------
    def visit_ListComp(self, node: ast.ListComp) -> None:
        _add(self.findings, node, "ListComp")
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        _add(self.findings, node, "DictComp")
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        _add(self.findings, node, "SetComp")
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        _add(self.findings, node, "GenExp")
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        _add(self.findings, node, "Yield")
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        _add(self.findings, node, "Yield")
        self.generic_visit(node)

    # -- statements ----------------------------------------------------
    def visit_With(self, node: ast.With) -> None:
        _add(self.findings, node, "With")
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        _add(self.findings, node, "With")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        _add(self.findings, node, "Assert")
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        _add(self.findings, node, "Del")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        _add(self.findings, node, "Match")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        _add(self.findings, node, "Global")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        _add(self.findings, node, "Nonlocal")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        _add(self.findings, node, "TryExcept")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if node.orelse:
            _add(self.findings, node, "ForElse")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if node.orelse:
            _add(self.findings, node, "ForElse")
        # Truthiness: a bare name/call as the condition depends on
        # Python's truthiness rules (unlike C#, which needs a real bool).
        if not isinstance(node.test, (ast.Compare, ast.BoolOp, ast.Constant)):
            _add(self.findings, node.test, "Truthiness")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        # `if __name__ == "__main__":` — the classic entry-point guard.
        if self._is_main_guard(node.test):
            _add(self.findings, node, "MainGuard")
        elif not isinstance(node.test, (ast.Compare, ast.BoolOp, ast.Constant)):
            _add(self.findings, node.test, "Truthiness")
        self.generic_visit(node)

    @staticmethod
    def _is_main_guard(test: ast.expr) -> bool:
        """Detect ``__name__ == "__main__"`` (either operand order)."""
        if not isinstance(test, ast.Compare):
            return False
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            return False
        sides = [test.left, *test.comparators]
        names = {
            s.id for s in sides if isinstance(s, ast.Name)
        }
        mains = {
            s.value for s in sides if isinstance(s, ast.Constant)
        }
        return "__name__" in names and "__main__" in mains

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.cause is not None:
            _add(self.findings, node, "RaiseFrom")
        self.generic_visit(node)

    # -- functions -----------------------------------------------------
    def _check_decorated(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        names = [_decorator_name(d) for d in node.decorator_list]
        short = {n.split(".")[-1] for n in names if n}
        if "dataclass" in short:
            _add(self.findings, node, "Dataclass")
        if "property" in short:
            _add(self.findings, node, "Property")
        if node.decorator_list:
            _add(self.findings, node, "Decorator")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_decorated(node)
        if node.name == "__init__":
            _add(self.findings, node, "InitSelf")
        if node.name in ("__str__", "__repr__"):
            _add(self.findings, node, "StrRepr")
        if node.args.vararg is not None or node.args.kwarg is not None:
            _add(self.findings, node, "StarArgs")
        if node.args.defaults or node.args.kw_defaults:
            if self._has_mutable_default(node.args):
                _add(self.findings, node, "MutableDefault")
        if self._has_annotations(node):
            _add(self.findings, node, "TypeHints")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_decorated(node)
        _add(self.findings, node, "AsyncDef")
        if node.args.vararg is not None or node.args.kwarg is not None:
            _add(self.findings, node, "StarArgs")
        if self._has_annotations(node):
            _add(self.findings, node, "TypeHints")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_decorated(node)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        _add(self.findings, node, "Lambda")
        # Late-binding gotcha: a lambda created inside a loop captures the
        # loop *variable*, so every lambda sees its final value.
        if self._inside(ast.For, ast.AsyncFor, ast.While):
            _add(self.findings, node, "LateBinding")
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        _add(self.findings, node, "Await")
        self.generic_visit(node)

    @staticmethod
    def _has_mutable_default(args: ast.arguments) -> bool:
        """True if any default value is a list/dict/set literal."""
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d]
        return any(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in defaults)

    @staticmethod
    def _has_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """True if the def carries any type hints (params or return)."""
        args = node.args
        all_args = args.args + args.kwonlyargs
        if args.vararg is not None:
            all_args.append(args.vararg)
        if args.kwarg is not None:
            all_args.append(args.kwarg)
        return any(a.annotation is not None for a in all_args) or (
            node.returns is not None
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        _add(self.findings, node, "TypeHints")
        self.generic_visit(node)

    # -- expressions ---------------------------------------------------
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.FloorDiv):
            _add(self.findings, node, "FloorDiv")
        elif isinstance(node.op, ast.Pow):
            _add(self.findings, node, "Power")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) > 1:
            _add(self.findings, node, "ChainedComp")
        for op in node.ops:
            if isinstance(op, (ast.Is, ast.IsNot)):
                _add(self.findings, node, "IsOp")
                break
        for op in node.ops:
            if isinstance(op, (ast.In, ast.NotIn)):
                _add(self.findings, node, "InOp")
                break
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        sl = node.slice
        if isinstance(sl, ast.Slice):
            _add(self.findings, node, "Slice")
        elif self._is_negative_index(sl):
            _add(self.findings, node, "NegIndex")
        self.generic_visit(node)

    @staticmethod
    def _is_negative_index(sl: ast.expr) -> bool:
        """Detect ``a[-1]``: UnaryOp(USub, Constant) or a negative Constant."""
        if isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub):
            return isinstance(sl.operand, ast.Constant)
        return isinstance(sl, ast.Constant) and isinstance(sl.value, int) and sl.value < 0

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        _add(self.findings, node, "FString")
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        _add(self.findings, node, "Walrus")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        _add(self.findings, node, "Ternary")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                _add(self.findings, node, "TupleUnpack")
                break
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        # {**a, **b}: a None key means "unpack this mapping here".
        if any(k is None for k in node.keys):
            _add(self.findings, node, "DictMerge")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Builtins with a direct C# story: enumerate / zip / range.
        if isinstance(node.func, ast.Name):
            if node.func.id == "enumerate":
                _add(self.findings, node, "Enumerate")
            elif node.func.id == "zip":
                _add(self.findings, node, "Zip")
            elif node.func.id == "range":
                _add(self.findings, node, "Range")
        # Method calls: d.get(k) and ", ".join(items).
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "get":
                _add(self.findings, node, "DictGet")
            elif node.func.attr == "join":
                _add(self.findings, node, "StrJoin")
        # Call-site unpacking: f(*args) or f(**kwargs).
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            kw.arg is None for kw in node.keywords
        ):
            _add(self.findings, node, "StarUnpack")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value is None:
            _add(self.findings, node, "NoneLit")
        # No generic_visit needed: constants have no children, but calling
        # it is harmless and keeps the pattern uniform.
        self.generic_visit(node)


def annotate(python_code: str) -> list[Finding]:
    """Parse *python_code* and return one Finding per detected construct.

    Raises ``SyntaxError`` (from ``ast.parse``) on invalid Python — callers
    turn that into the ``errors`` field instead of calling any LLM.
    Findings are sorted by (line, end_line) so notes read top-to-bottom,
    like a compiler's diagnostic list.
    """
    tree = ast.parse(python_code)
    visitor = Annotator()
    visitor.visit(tree)
    visitor.findings.sort(key=lambda f: (f["line"], f["end_line"]))
    return visitor.findings
