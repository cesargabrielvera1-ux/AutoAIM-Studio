"""Regression test: no undefined names in type annotations.

v2.0: A `NameError: name 'Optional' is not defined` crash slipped into a
PyInstaller build because a signature annotation used a typing name that was
never imported. Function-signature annotations are evaluated at class/function
definition time (runtime), so this is NOT caught by `py_compile` (syntax only).
This AST-based test scans every app module and fails if any annotation
references a name that is neither imported, defined at module level, nor a
builtin. It needs no GUI / heavy deps.
"""

import ast
import builtins
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2] / "app"

_BUILTIN = set(dir(builtins)) | {"self", "cls"}


def _module_defined_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _annotation_names(node, acc):
    if node is None or isinstance(node, ast.Constant):
        return  # quoted forward refs are lazy — never evaluated at def time
    if isinstance(node, ast.Name):
        acc.add(node.id)
    elif isinstance(node, ast.Attribute):
        _annotation_names(node.value, acc)
    elif isinstance(node, ast.Subscript):
        _annotation_names(node.value, acc)
        _annotation_names(node.slice, acc)
    elif isinstance(node, ast.Tuple):
        for e in node.elts:
            _annotation_names(e, acc)
    elif isinstance(node, ast.BinOp):  # X | Y unions
        _annotation_names(node.left, acc)
        _annotation_names(node.right, acc)


def _undefined_annotation_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defined = _module_defined_names(tree) | _BUILTIN
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            used = set()
            for arg in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs):
                _annotation_names(arg.annotation, used)
            for arg in (node.args.vararg, node.args.kwarg):
                if arg is not None:
                    _annotation_names(arg.annotation, used)
            _annotation_names(node.returns, used)
            for name in sorted(used):
                if name not in defined:
                    problems.append((node.lineno, node.name, name))
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            used = set()
            _annotation_names(node.annotation, used)
            for name in sorted(used):
                if name not in defined:
                    problems.append((node.lineno, "<assign>", name))
    return problems


def _all_app_modules():
    return sorted(APP_ROOT.rglob("*.py"))


@pytest.mark.parametrize("pyfile", _all_app_modules(),
                         ids=lambda p: str(p.relative_to(APP_ROOT)))
def test_no_undefined_annotation_names(pyfile):
    """Every annotation name must be imported / module-defined / a builtin."""
    problems = _undefined_annotation_names(pyfile)
    assert not problems, (
        f"{pyfile.relative_to(APP_ROOT)} has annotation(s) referencing undefined "
        f"names (NameError at class/def time, breaks frozen builds):\n"
        + "\n".join(f"  line {ln} in '{fn}': '{nm}'" for ln, fn, nm in problems)
    )


def test_app_modules_present():
    """Sanity: the audit actually found app modules to check."""
    assert len(_all_app_modules()) > 10
