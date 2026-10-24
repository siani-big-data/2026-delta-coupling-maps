import argparse
import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source_files():
    return sorted(
        path for folder in ("src", "tests", "tools") for path in (ROOT / folder).rglob("*.py")
    )


def _without_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body.pop(0)
                if not node.body and not isinstance(node, ast.Module):
                    node.body.append(ast.Pass())
    return tree


def _clean(text):
    description = ast.get_docstring(ast.parse(text))
    if description is not None:
        text = text.replace("description=__doc__", f"description={description!r}")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        expression = node.body[0]
        if not isinstance(expression.value, ast.Constant) or not isinstance(
            expression.value.value, str
        ):
            continue
        replacement = (
            " " * expression.col_offset + "pass\n"
            if len(node.body) == 1 and not isinstance(node, ast.Module)
            else "\n"
        )
        lines[expression.lineno - 1] = replacement
        for index in range(expression.lineno, expression.end_lineno):
            lines[index] = "\n"
    tokens = tokenize.generate_tokens(io.StringIO("".join(lines)).readline)
    cleaned = tokenize.untokenize(
        token._replace(string="") if token.type == tokenize.COMMENT else token for token in tokens
    )
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip() + "\n"
    expected = ast.dump(_without_docstrings(ast.parse(text)), include_attributes=False)
    actual = ast.dump(_without_docstrings(ast.parse(cleaned)), include_attributes=False)
    if expected != actual:
        raise RuntimeError("Removing comments changed executable syntax")
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = []
    for path in _source_files():
        before = path.read_text()
        after = _clean(before)
        if before != after:
            changed.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.write_text(after)
    if args.check and changed:
        raise SystemExit("Comments/docstrings remain: " + ", ".join(changed))
    print(f"Checked {len(_source_files())} Python files; changed {len(changed)}")


if __name__ == "__main__":
    main()
