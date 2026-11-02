"""The report's result charts, drawn from the saved evaluation files in data/evaluation/.

Needs matplotlib. Run:  python report/make_charts.py     Saves PNG files in report/figures/.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "evaluation"
OUT = Path(__file__).parent / "figures"
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, GREY, AMBER = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#eda100"
BEFORE = {"Set 1": (18, 27), "Set 2": (7, 30)}                # parser at commit e75a8e4, from simulated_testers.md


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def learning():
    rows = list(csv.DictReader(open(DATA / "simulated_testers.csv", encoding="utf-8")))
    rows = [r for r in rows if r["wording"] == "clean" and r["dislike"] != "no hidden dislike"]
    eaten, served = defaultdict(list), defaultdict(list)
    for r in rows:
        eaten[r["arm"], int(r["week"])].append(float(r["meals_eaten"]) * 100)
        served[r["arm"], int(r["week"])].append(float(r["disliked_served"]))
    mean = lambda v: sum(v) / len(v)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    weeks = [1, 2, 3, 4]
    panels = ((axes[0], eaten, "Meals eaten (%)", (60, 100), "{:.0f}%"), (axes[1], served, "Disliked foods served per week", (0, 7), "{:.1f}"))
    for ax, table, title, ylim, fmt in panels:
        style(ax)
        for arm, color in (("with learning", BLUE), ("no learning", ORANGE)):
            ys = [mean(table[arm, w]) for w in weeks]
            ax.plot(weeks, ys, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5)
            ax.annotate(f"{arm}  " + fmt.format(ys[-1]), (4, ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_xlim(0.6, 6.1)
        ax.tick_params(axis="x", pad=6)
        ax.set_xticks(weeks, [f"Week {w}" for w in weeks])
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_5_1_learning.png", dpi=200, facecolor="white")
    plt.close(fig)


def parser_sets():
    rows = list(csv.DictReader(open(DATA / "new_sentences_scored.csv", encoding="utf-8")))
    labels, blocked, asked, silent = [], [], [], []
    for key, label in (("Set 1", "Set 1, after the fix\n(used to build it)"), ("Set 2", "Set 2, after the fix\n(failures seen)"),
                       ("Set 3", "Set 3, after the fix\n(kept apart, scored once)")):
        items = [r for r in rows if r["set"].startswith(key)]
        total = sum(int(r["allergy_expected"]) for r in items)
        got = sum(int(r["allergy_blocked"]) for r in items)
        ask = sum(int(r["allergy_expected"]) - int(r["allergy_blocked"]) for r in items if r["warned"] == "True")
        if key in BEFORE:
            b, t = BEFORE[key]
            labels.append(f"{key}, before the fix")
            blocked.append(b / t * 100)
            asked.append(0)
            silent.append(100 - b / t * 100)
        labels.append(label)
        blocked.append(got / total * 100)
        asked.append(ask / total * 100)
        silent.append((total - got - ask) / total * 100)
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    y = list(range(len(labels)))[::-1]
    ax.barh(y, blocked, color=BLUE, height=0.62, edgecolor="white", lw=2, label="Blocked")
    ax.barh(y, asked, left=blocked, color=AMBER, height=0.62, edgecolor="white", lw=2, label="Not blocked, the app asks")
    ax.barh(y, silent, left=[a + b for a, b in zip(blocked, asked)], color=ORANGE, height=0.62, edgecolor="white", lw=2, label="Missed with no warning")
    for yi, b, a, s in zip(y, blocked, asked, silent):
        ax.text(b - 1.5, yi, f"{b:.0f}%", ha="right", va="center", fontsize=8.5, color="white", fontweight="bold")
        if a >= 6:
            ax.text(b + a - 1.5, yi, f"{a:.0f}%", ha="right", va="center", fontsize=8.5, color=INK, fontweight="bold")
        if s >= 6:
            ax.text(b + a + s - 1.5, yi, f"{s:.0f}%", ha="right", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks(y, labels, fontsize=8.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_title("Share of allergy items", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_2_parser_sets.png", dpi=200, facecolor="white")
    plt.close(fig)


def nutrients():
    text = (DATA / "results.md").read_text(encoding="utf-8")
    section = text[text.index("## 8. Nutrients"):text.index("Days over a daily cap")]
    rows = re.findall(r"^\| (\w[\w ]+) \| ([\d.]+)% \| ([\d.]+)% \|$", section, flags=re.M)
    names = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    style(ax)
    x = range(len(names))
    w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], [float(r[1]) for r in rows], w, color=ORANGE, label="Macros only")
    ax.bar([i + w / 2 + 0.01 for i in x], [float(r[2]) for r in rows], w, color=BLUE, label="Nutrient-aware (final)")
    for i, r in enumerate(rows):
        ax.text(i - w / 2 - 0.01, float(r[1]) + 1.5, f"{float(r[1]):.0f}", ha="center", fontsize=8, color=MUTED)
        ax.text(i + w / 2 + 0.01, float(r[2]) + 1.5, f"{float(r[2]):.0f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(list(x), names, fontsize=9, color=INK)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Planned days on target", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_5_3_nutrients.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    learning()
    parser_sets()
    nutrients()
    print("charts saved")
