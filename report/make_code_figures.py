"""Light-mode images of the key functions, read straight from the source so they always match the code.

Needs matplotlib. Run:  python report/make_code_figures.py     Saves PNG files in report/figures/.
"""
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "figures"
FIGURES = [("fig_4_5_code_solve_grams", "src/planner.py", "solve_grams"),
           ("fig_4_6_code_update_taste", "src/feedback.py", "update_taste"),
           ("fig_4_7_code_health_upgrade", "src/parser.py", "health_upgrade")]
WIDTH, FONT = 96, 9.2                   # characters per row and font size: stays readable on the page


def source_of(path, name):
    text = (ROOT / path).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines()
    return node.lineno, lines[node.lineno - 1:node.end_lineno]


def wrapped(lines, start, width=WIDTH):
    """(line number or "", text) for each row: a long source line continues on the next row, indented."""
    rows = []
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip()) + 6
        first = True
        while len(line) > width:
            cut = line.rfind(" ", indent, width)
            cut = cut if cut > indent else width
            rows.append((start + i if first else "", line[:cut]))
            line = " " * indent + line[cut:].lstrip()
            first = False
        rows.append((start + i if first else "", line))
    return rows


def draw(name, path, function):
    start, lines = source_of(path, function)
    rows = wrapped(lines, start)
    row = 0.2
    height = 0.12 + row * (len(rows) + 1)
    fig = plt.figure(figsize=(7.4, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    step = row / height
    y = 1 - step * 0.8
    ax.text(0.008, y, path, fontsize=FONT, family="Consolas", color="#5a616b", va="center", fontweight="bold")
    y -= step
    for number, text in rows:
        ax.text(0.008, y, f"{number:>4}", fontsize=FONT, family="Consolas", color="#9aa1ab", va="center")
        ax.text(0.058, y, text, fontsize=FONT, family="Consolas", color="#1b1f24", va="center")
        y -= step
    fig.savefig(OUT / f"{name}.png", dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(name, len(rows), "rows")


if __name__ == "__main__":
    for figure in FIGURES:
        draw(*figure)
