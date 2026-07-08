#!/usr/bin/env python3
"""Validate generated notebooks:
  1) every notebook parses as nbformat v4
  2) solution code cells are syntactically valid Python (magics stripped)
  3) participant notebooks actually contain blanks (____) or TODOs
"""
import ast
import glob
import os
import sys

import nbformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sanitize(src):
    """Drop IPython magics so the remaining code can be parsed by ast."""
    out = []
    for i, ln in enumerate(src.split("\n")):
        s = ln.lstrip()
        if i == 0 and s.startswith("%%"):
            continue  # cell magic header, e.g. %%writefile path
        if s.startswith("%") or s.startswith("!"):
            continue  # line magic or shell escape
        out.append(ln)
    return "\n".join(out)


def check_dir(folder, expect_blanks):
    files = sorted(glob.glob(os.path.join(ROOT, folder, "*.ipynb")))
    assert files, f"no notebooks in {folder}"
    problems = []
    for path in files:
        nb = nbformat.read(path, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        blanks = 0
        for idx, c in enumerate(code_cells):
            src = c.source
            if "____" in src or "NotImplementedError" in src:
                blanks += 1
            # syntax check (solutions must be clean; participants have valid-identifier blanks)
            try:
                ast.parse(sanitize(src))
            except SyntaxError as e:
                problems.append(f"{os.path.basename(path)} code cell #{idx}: {e}")
        tag = "blanks" if expect_blanks else "clean"
        print(f"  {os.path.basename(path):26s} code_cells={len(code_cells):2d} {tag}={blanks}")
        if expect_blanks and blanks == 0:
            # Cleanup notebook is intentionally complete (no blanks needed)
            if "cleanup" not in os.path.basename(path).lower():
                problems.append(f"{os.path.basename(path)}: participant notebook has NO blanks")
    return problems


def main():
    problems = []
    print("[solutions/] (must be syntactically clean)")
    problems += check_dir("solutions", expect_blanks=False)
    print("[notebooks/] (participant - must contain blanks)")
    problems += check_dir("notebooks", expect_blanks=True)

    print()
    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
