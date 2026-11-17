"""From `coverage annotate` output (files ending in ,cover): per source file, the functions whose BODY ran, with executed / total lines.
A `def` line alone is executed at import time, so a function counts only if a line inside it ran.

    python devtools/coverage_functions.py <annotate dir>
"""
import ast
import sys
from pathlib import Path

for cover in sorted(Path(sys.argv[1]).glob("*,cover")):
    lines = cover.read_text().splitlines()
    marks = [ln[:1] for ln in lines]                       # '>' executed, '!' not executed, ' ' not a statement
    source = "\n".join(ln[2:] for ln in lines)
    rows = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = range(node.body[0].lineno - 1, node.end_lineno)
            ran = sum(1 for i in body if marks[i] == ">")
            total = sum(1 for i in body if marks[i] in ">!")
            if ran:
                rows.append((node.lineno, node.name, ran, total))
    if rows:
        print(f"{cover.name[:-6]}  ({sum(r[2] for r in rows)} lines ran inside {len(rows)} functions)")
        for lineno, name, ran, total in sorted(rows):
            print(f"    {lineno:5d} {name:38s} {ran:4d}/{total}")
