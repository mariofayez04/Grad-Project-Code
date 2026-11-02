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


def source_of(path, name):
    text = (ROOT / path).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines()
    return node.lineno, lines[node.lineno - 1:node.end_lineno]


def draw(name, path, function):
    start, lines = source_of(path, function)
    height = 0.3 + 0.19 * (len(lines) + 1)
    fig = plt.figure(figsize=(8.6, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    step = 0.19 / height
    y = 1 - step
    ax.text(0.012, y, f"{path}", fontsize=8.5, family="Consolas", color="#5a616b", va="center", fontweight="bold")
    y -= step
    for i, line in enumerate(lines):
        ax.text(0.012, y, f"{start + i:>4}", fontsize=8.5, family="Consolas", color="#9aa1ab", va="center")
        ax.text(0.06, y, line, fontsize=8.5, family="Consolas", color="#1b1f24", va="center")
        y -= step
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(name, len(lines), "lines; longest", max(len(x) for x in lines))


if __name__ == "__main__":
    for figure in FIGURES:
        draw(*figure)
