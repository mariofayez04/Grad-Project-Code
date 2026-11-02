"""Draws the report's diagrams from what the code really does (tables in src/db.py, modules in src/).

Needs matplotlib. Run:  python report/make_diagrams.py     Saves PNG files in report/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
INK, MUTED, LINE = "#1b1f24", "#5a616b", "#6b7280"
BLUE, GREEN, AMBER, GREY, RED = "#dbe8f7", "#dcefe3", "#fbeccb", "#eef0f2", "#f8dfdc"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fill=BLUE, size=9.5, bold=False, edge=LINE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fill, ec=edge, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, a, b, text="", dashed=False, offset=(0, 0.1), size=8, rad=0, both=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>" if both else "-|>", mutation_scale=11, lw=1, color=LINE,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, ha="center", va="center", fontsize=size,
                color=MUTED, linespacing=1.15, bbox=dict(fc="white", ec="none", pad=0.6))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)


def methodology():
    steps = [("1  Data", "USDA SR Legacy cleaned\nto 3,184 foods; allergen\nand diet tags; 20\nEgyptian dishes added"),
             ("2  Targets", "Mifflin-St Jeor, activity\nand goal give calories;\nmacros and nutrient\ntargets follow"),
             ("3  Language", "NER model and\ndictionary read typed\npreferences; a safety\nlayer guards allergies"),
             ("4  Embeddings", "Pre-trained model turns\nevery food into a\n384-number vector,\ncomputed once"),
             ("5  Planner", "Filter, score, combine,\nthen solve the grams\nwith bounded least\nsquares"),
             ("6  Web app", "Streamlit pages, sign-in\nwith PIN, database,\nshopping list, PDF,\nweight tracking"),
             ("7  Learning", "Eaten, skipped, swapped\nand chosen foods update\nthe taste vector and\nfood penalties"),
             ("8  Evaluation", "Accuracy, allergy safety,\nparser tests, simulated\ntesters, hosted\ndeployment")]
    fig, ax = canvas(9.4, 4.5)
    w, h = 2.1, 1.75
    for i, (title, text) in enumerate(steps):
        x, y = 0.05 + (i % 4) * 2.35, 2.55 if i < 4 else 0.25
        box(ax, x, y, w, h, "", fill=GREEN if i in (2, 3, 6) else BLUE)
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + 0.68, text, ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.25)
        if i % 4 != 3:
            arrow(ax, (x + w, y + h / 2), (x + 2.35, y + h / 2))
    arrow(ax, (0.05 + 3 * 2.35 + w / 2, 2.55), (0.05 + w / 2, 2.0), rad=0)
    ax.text(0.05, 0.02, "Green: the steps that use AI models. Blue: formulas, optimization and software engineering.", fontsize=8, color=MUTED)
    save(fig, "fig_1_1_methodology")


def recommender_families():
    fig, ax = canvas(8.6, 3.5)
    box(ax, 3.0, 2.8, 2.6, 0.55, "Recommender systems", fill=GREY, bold=True)
    fams = [("Content-based", "Compares the properties of items\nwith what this user liked.\nWorks with one user's data.", GREEN),
            ("Collaborative filtering", "Uses the behavior of many\nsimilar users. Needs a large\nuser base from the start.", BLUE),
            ("Hybrid", "Combines techniques to cover\nthe weaknesses of each one.", BLUE)]
    for i, (t, d, c) in enumerate(fams):
        x = 0.05 + i * 2.95
        box(ax, x, 0.75, 2.6, 1.45, "", fill=c)
        ax.text(x + 1.3, 1.95, t, ha="center", fontsize=10, fontweight="bold", color=INK, va="center")
        ax.text(x + 1.3, 1.25, d, ha="center", va="center", fontsize=8.5, color=INK, linespacing=1.3)
        arrow(ax, (4.3, 2.8), (x + 1.3, 2.2))
    ax.text(0.05, 0.3, "Green: the family closest to this project, which adds nutrition targets and hard safety rules\nthat a general recommender does not have.",
            fontsize=8.5, color=MUTED, va="center", linespacing=1.25)
    save(fig, "fig_2_1_recommender_families")


def ner_example():
    words = [("I'm", "O"), ("allergic", "cue"), ("to", "O"), ("shrimp", "FOOD"), ("and", "O"), ("I", "O"), ("hate", "cue"), ("mushrooms", "FOOD")]
    fig, ax = canvas(9, 3.3)
    x = 0.05
    centers = {}
    for w, tag in words:
        width = 0.5 + 0.083 * len(w)
        fill = {"FOOD": GREEN, "cue": AMBER, "O": GREY}[tag]
        box(ax, x, 2.45, width, 0.5, w, fill=fill, size=10)
        ax.text(x + width / 2, 2.25, {"FOOD": "FOOD", "cue": "cue phrase", "O": ""}[tag], ha="center", fontsize=8, color=MUTED, va="center")
        centers[w] = x + width / 2
        x += width + 0.08
    box(ax, 0.9, 0.3, 3.3, 1.05, "Block\nshrimp + the shellfish group\n(from the allergen dictionary)", fill=RED, size=9)
    box(ax, 5.0, 0.3, 3.3, 1.05, "Avoid\nmushroom\n(soft penalty when ranking)", fill=AMBER, size=9)
    arrow(ax, (centers["shrimp"], 2.1), (2.55, 1.35))
    arrow(ax, (centers["mushrooms"], 2.1), (6.65, 1.35))
    save(fig, "fig_2_2_ner_example")


def use_case():
    fig, ax = canvas(9.2, 7.2)
    ax.add_patch(Rectangle((1.7, 0.1), 7.4, 7.0, fc="white", ec=LINE, lw=1.1))
    ax.text(5.4, 6.9, "NutriPlan AI", ha="center", fontsize=11, fontweight="bold", color=INK, va="center")
    # actor
    ax.add_patch(plt.Circle((0.7, 4.15), 0.17, fc="white", ec=INK, lw=1.2))
    ax.plot([0.7, 0.7], [3.98, 3.45], color=INK, lw=1.2)
    ax.plot([0.35, 1.05], [3.8, 3.8], color=INK, lw=1.2)
    ax.plot([0.7, 0.4], [3.45, 3.05], color=INK, lw=1.2)
    ax.plot([0.7, 1.0], [3.45, 3.05], color=INK, lw=1.2)
    ax.text(0.7, 2.85, "User", ha="center", fontsize=10, fontweight="bold", color=INK)

    def uc(x, y, text, fill=BLUE, w=2.4, h=0.62):
        ax.add_patch(Ellipse((x, y), w, h, fc=fill, ec=LINE, lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.4, color=INK, linespacing=1.1)

    left = ["Sign up / sign in\nwith a PIN", "Enter profile and\npreferences", "Generate weekly plan", "View plan, nutrients\nand shopping list",
            "Tick meals eaten", "Swap a food", "Ask for a new\nmeal or day", "Download PDF or CSV", "Log weight", "Finish the week and\nrate it"]
    ys = [6.4 - i * 0.67 for i in range(len(left))]
    for y, t in zip(ys, left):
        uc(3.2, y, t)
        arrow(ax, (1.08, 3.6), (2.0, y), both=False)
    inc = {2: [("Calculate calorie\nand macro targets", 7.1, 5.95), ("Read preferences\nand allergies", 7.1, 5.2), ("Build meals from\nfood data", 7.1, 4.45)],
           6: [("Compare current and\nnew version", 7.1, 2.6)], 9: [("Learn from eaten and\nskipped meals", 7.1, 1.3)]}
    for i, items in inc.items():
        for t, x, y in items:
            uc(x, y, t, fill=GREEN)
            arrow(ax, (4.4, ys[i]), (5.9, y), dashed=True, text="«include»", size=7)
    save(fig, "fig_3_1_use_case")


def dfd():
    fig, ax = canvas(9.6, 6.6)
    box(ax, 0.05, 2.9, 1.1, 0.8, "User", fill=GREY, bold=True)

    def proc(x, y, n, text):
        ax.add_patch(Ellipse((x, y), 2.05, 0.95, fc=BLUE, ec=LINE, lw=1))
        ax.text(x, y, f"{n}\n{text}", ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.15)

    def store(x, y, name, text):
        ax.plot([x, x + 2.15], [y + 0.5, y + 0.5], color=LINE, lw=1)
        ax.plot([x, x + 2.15], [y, y], color=LINE, lw=1)
        ax.plot([x + 0.42, x + 0.42], [y, y + 0.5], color=LINE, lw=1)
        ax.text(x + 0.21, y + 0.25, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + 1.28, y + 0.25, text, ha="center", va="center", fontsize=7.9, color=INK, linespacing=1.05)

    proc(3.1, 5.75, "1.0", "Sign in and\nsave profile")
    proc(3.1, 4.2, "2.0", "Calculate\ntargets")
    proc(3.1, 2.65, "3.0", "Read preferences\n(NER + rules)")
    proc(5.75, 3.45, "4.0", "Plan the\nweek's meals")
    proc(3.1, 1.0, "5.0", "Track use: ticks,\nswaps, weight")
    proc(5.75, 1.0, "6.0", "Learn from\nfeedback")
    store(7.4, 5.5, "D1", "users")
    store(7.4, 4.3, "D2", "plans,\nplan_items")
    store(7.4, 3.2, "D3", "foods_clean.csv,\nembeddings.npy")
    store(7.4, 0.75, "D4", "swaps, choices,\nweights, surveys")
    box(ax, 7.6, 2.05, 1.75, 0.6, "USDA SR Legacy\n(cleaned once)", fill=GREY, size=8)
    arrow(ax, (1.15, 3.55), (2.15, 5.55), "name, PIN,\nprofile", offset=(-0.12, 0.2))
    arrow(ax, (3.1, 5.27), (3.1, 4.68), "profile")
    arrow(ax, (1.15, 3.2), (2.08, 2.75), "typed text", offset=(0, 0.12))
    arrow(ax, (1.15, 3.0), (2.25, 1.2), "ticks, swaps,\nweights", offset=(-0.12, -0.15))
    arrow(ax, (4.1, 5.75), (7.4, 5.75), "profile, PIN hash", offset=(0, 0.13), both=True)
    arrow(ax, (4.03, 4.1), (4.95, 3.75), "targets", offset=(-0.05, 0.14))
    arrow(ax, (4.1, 2.75), (4.9, 3.15), "allow / avoid /\nprefer", offset=(0.05, -0.27))
    arrow(ax, (8.47, 2.65), (8.47, 3.2), dashed=True)
    arrow(ax, (7.4, 3.45), (6.78, 3.45), "foods,\nvectors", offset=(0, 0.25))
    arrow(ax, (6.5, 3.8), (7.4, 4.5), "week's plan", offset=(-0.22, 0.17))
    arrow(ax, (4.8, 3.1), (1.15, 3.35), "", dashed=False)
    ax.text(2.0, 3.53, "plan, nutrients,\nshopping list, PDF", fontsize=7.8, color=MUTED, ha="center", va="center", linespacing=1.1,
            bbox=dict(fc="white", ec="none", pad=0.5))
    arrow(ax, (4.1, 1.0), (4.72, 1.0), "finished\nweek", offset=(0, 0.28))
    arrow(ax, (7.4, 1.0), (6.78, 1.0))
    arrow(ax, (3.1, 0.52), (8.47, 0.75), "ticks (saved in D2); swaps, choices, weights, ratings (D4)", offset=(0, -0.3), rad=0.12)
    arrow(ax, (5.75, 1.47), (5.75, 2.97), "taste vector,\npenalties", offset=(-0.55, 0))
    save(fig, "fig_3_2_dfd")


def erd():
    tables = {
        "users": (0.1, 3.25, ["PK id", "name (unique)", "age, sex", "weight_kg, height_cm", "activity_level, goal", "preferences_text", "meals_per_day", "health_modes", "egyptian_dishes", "pin_hash, pin_salt", "failed_logins, locked_until"]),
        "weights": (0.1, 0.55, ["PK id", "FK user_id", "day (unique per user)", "weight_kg"]),
        "plans": (3.55, 3.7, ["PK id", "FK user_id", "week_number, seed", "targets_json", "prefs_json", "taste_vector", "finished"]),
        "plan_items": (7.0, 4.15, ["PK id", "FK plan_id", "day_index, day, slot", "fdc_id, name, grams", "calories, protein_g", "carbs_g, fat_g", "eaten"]),
        "swaps": (7.0, 2.2, ["PK id", "FK plan_id", "day_index, slot", "old_fdc_id, old_name", "new_fdc_id"]),
        "choices": (7.0, 0.1, ["PK id", "FK plan_id", "day_index, slot", "kept", "current_foods", "new_foods"]),
        "surveys": (3.55, 0.55, ["PK id", "FK plan_id (unique)", "liked, portions, easy", "would_use", "comment"]),
    }
    fig, ax = canvas(9.5, 7.0)
    w, rh = 2.35, 0.235
    pos = {}
    for name, (x, y, cols) in tables.items():
        h = rh * (len(cols) + 1.25)
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=LINE, lw=1))
        ax.add_patch(Rectangle((x, y + h - rh * 1.25), w, rh * 1.25, fc=BLUE, ec=LINE, lw=1))
        ax.text(x + w / 2, y + h - rh * 0.62, name, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
        for i, c in enumerate(cols):
            ax.text(x + 0.08, y + h - rh * 1.25 - rh * (i + 0.5), c, va="center", fontsize=7.8, color=INK)
        pos[name] = (x, y, w, h)

    def link(a, b, pa, pb):
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=LINE, lw=1.1)
        for share, label in ((0.2, "1"), (0.8, b)):
            ax.text(pa[0] + (pb[0] - pa[0]) * share, pa[1] + (pb[1] - pa[1]) * share, label, fontsize=9, fontweight="bold", color=INK,
                    ha="center", va="center", bbox=dict(fc="white", ec="none", pad=1))

    link("users", "N", (2.45, 4.9), (3.55, 4.9))
    link("users", "N", (1.27, 3.25), (1.27, 1.87))
    link("plans", "N", (5.9, 5.0), (7.0, 5.0))
    link("plans", "N", (5.9, 4.5), (7.0, 3.0))
    link("plans", "N", (5.9, 4.1), (7.0, 1.0))
    link("plans", "0..1", (4.72, 3.7), (4.72, 2.2))
    box(ax, 3.55, 5.95, 2.35, 0.8, "Food table (foods_clean.csv)\nread-only file, not in the database;\nitems refer to it by fdc_id", fill=GREY, size=7.4)
    ax.plot([5.9, 7.0], [6.35, 5.9], color=LINE, lw=1, linestyle="--")
    save(fig, "fig_3_3_erd")


def architecture():
    fig, ax = canvas(9.4, 6.3)

    def layer(y, h, title):
        ax.add_patch(Rectangle((0.05, y), 9.25, h, fc="#fafbfc", ec="#c9ced6", lw=1))
        ax.text(0.15, y + h - 0.17, title, fontsize=8.6, fontweight="bold", color=MUTED, va="center")

    layer(5.15, 1.05, "Browser")
    box(ax, 2.6, 5.25, 4.2, 0.62, "Web page: Plan, Nutrients, Shopping, Progress, Profile", size=9, fill=GREY)
    layer(4.0, 1.0, "User interface")
    box(ax, 2.6, 4.1, 4.2, 0.62, "app.py (Streamlit): pages, forms, sign-in, dialogs", size=9)
    layer(2.0, 1.85, "Application logic (src/)")
    mods = ["calculator\ntargets", "parser\npreferences", "planner\nmeals, swaps", "feedback\nlearning", "nutrients\ndiet modes",
            "shopping\nlist", "progress\nweight", "report\nPDF", "dishes\nEgyptian", "db\nstorage"]
    for i, m in enumerate(mods):
        box(ax, 0.2 + (i % 5) * 1.82, 2.87 - (i // 5) * 0.76, 1.7, 0.66, m, size=8.3, fill=GREEN if m.startswith(("parser", "planner", "feedback")) else BLUE)
    layer(0.05, 1.8, "Models and data")
    box(ax, 0.2, 0.9, 2.15, 0.62, "FoodBaseBERT-NER\nfinds food words", size=8.1, fill=GREEN)
    box(ax, 2.5, 0.9, 2.15, 0.62, "all-MiniLM-L6-v2\nfood vectors", size=8.1, fill=GREEN)
    box(ax, 0.2, 0.15, 4.45, 0.62, "foods_clean.csv, embeddings.npy, dish_ingredients.csv", size=8.1, fill=GREY)
    box(ax, 4.95, 0.9, 2.1, 0.62, "SQLite file\n(running on a PC)", size=8.1, fill=AMBER)
    box(ax, 7.1, 0.9, 2.1, 0.62, "Neon Postgres\n(hosted site)", size=8.1, fill=AMBER)
    ax.text(7.08, 0.42, "Same queries: db.py chooses by\nthe DATABASE_URL setting", fontsize=8, color=MUTED, ha="center", va="center", linespacing=1.2)
    arrow(ax, (4.7, 5.25), (4.7, 4.72), both=True)
    arrow(ax, (4.7, 4.1), (4.7, 3.53))
    arrow(ax, (2.42, 2.05), (2.42, 1.52))
    arrow(ax, (7.07, 2.05), (7.07, 1.52))
    save(fig, "fig_4_1_architecture")


def parser_flow():
    fig, ax = canvas(9.2, 4.2)
    box(ax, 0.05, 3.2, 1.75, 0.8, "Typed text\n\"im alergic to\npeanuts\"", fill=GREY, size=8.2)
    box(ax, 2.1, 3.2, 1.75, 0.8, "Fix spelling\n(alergic: allergic)", fill=GREEN, size=8.4)
    box(ax, 4.15, 3.2, 1.75, 0.8, "Find foods\nNER model +\ndictionary", fill=GREEN, size=8.4)
    box(ax, 6.2, 3.2, 2.9, 0.8, "Label each food by its nearest cue:\nblock, avoid or prefer", size=8.4)
    for a, b in ((1.8, 2.1), (3.85, 4.15), (5.9, 6.2)):
        arrow(ax, (a, 3.6), (b, 3.6))
    box(ax, 6.2, 1.85, 2.9, 0.85, "Health wording overrules softer labels:\nreactions, doctor's advice, conditions\n(celiac, favism)", fill=RED, size=8.2)
    arrow(ax, (7.65, 3.2), (7.65, 2.7))
    box(ax, 3.1, 1.85, 2.75, 0.85, "Allergen dictionary cross-check:\nno machine learning, tags the\ngroups (peanut, milk, gluten...)", fill=AMBER, size=8.2)
    arrow(ax, (6.2, 2.27), (5.85, 2.27))
    box(ax, 0.05, 1.85, 2.7, 0.85, "Rules for diets:\nvegetarian, vegan, halal,\npescatarian", size=8.2)
    arrow(ax, (1.4, 1.85), (3.1, 0.85))
    box(ax, 3.1, 0.3, 2.75, 0.95, "Never included, Less often,\nMore often: shown to the user\nand used to filter foods", size=8.4)
    arrow(ax, (4.47, 1.85), (4.47, 1.25))
    box(ax, 6.2, 0.3, 2.9, 0.95, "Could not tell what was meant:\nthe app asks instead of\nstaying silent", fill=RED, size=8.4)
    arrow(ax, (7.65, 1.85), (7.65, 1.25))
    save(fig, "fig_4_2_parser")


def planner_flow():
    fig, ax = canvas(9.2, 3.3)
    steps = [("Allowed foods", "allergies, diet rules and\ndiet modes removed", GREY),
             ("Score candidates", "job fit + taste + likes\n+ nutrients - repeats\n- unusual foods - learned", BLUE),
             ("Combine", "top foods per part of\nthe meal; also one dish\nplus optional sides", BLUE),
             ("Solve grams", "bounded least squares\nagainst what is left\nof the day", GREEN),
             ("Best meal wins", "fit + score, under the\ndaily nutrient caps", BLUE)]
    w = 1.68
    for i, (t, d, c) in enumerate(steps):
        x = 0.03 + i * 1.86
        box(ax, x, 1.15, w, 1.5, "", fill=c)
        ax.text(x + w / 2, 2.4, t, ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
        ax.text(x + w / 2, 1.73, d, ha="center", va="center", fontsize=7.8, color=INK, linespacing=1.25)
        if i < 4:
            arrow(ax, (x + w, 1.9), (x + 1.86, 1.9))
    arrow(ax, (8.3, 1.15), (0.9, 1.15), rad=-0.13)
    ax.text(4.6, 0.33, "Repeated for every meal of the 7 days; each meal aims at what is still left of that day.",
            fontsize=8.5, color=MUTED, ha="center", va="center")
    save(fig, "fig_4_3_planner")


def feedback_loop():
    fig, ax = canvas(9, 3.4)
    box(ax, 0.1, 1.3, 1.9, 0.95, "Week's plan\nshown to the user", fill=GREY, size=9)
    box(ax, 2.55, 2.2, 1.95, 0.85, "Eaten and skipped\nmeals", size=9)
    box(ax, 2.55, 0.35, 1.95, 0.85, "Swaps and before /\nafter choices", size=9)
    box(ax, 5.05, 2.2, 3.85, 0.85, "Taste vector\nnormalize(old + 0.3 x eaten - 0.15 x skipped)", fill=GREEN, size=8.6)
    box(ax, 5.05, 0.35, 3.85, 0.85, "Food and family penalties\nskipped +0.8, eaten -0.2, older weeks fade", fill=GREEN, size=8.6)
    arrow(ax, (2.0, 2.0), (2.55, 2.5))
    arrow(ax, (2.0, 1.55), (2.55, 0.9))
    arrow(ax, (4.5, 2.62), (5.05, 2.62))
    arrow(ax, (4.5, 0.77), (5.05, 0.77))
    arrow(ax, (4.5, 2.3), (5.05, 1.0))
    arrow(ax, (4.5, 1.0), (5.05, 2.3))
    box(ax, 5.8, 1.42, 2.35, 0.55, "Next week's plan", size=9, bold=True)
    arrow(ax, (6.97, 2.2), (6.97, 1.97))
    arrow(ax, (6.97, 1.2), (6.97, 1.42))
    arrow(ax, (5.8, 1.7), (2.0, 1.7), dashed=True)
    save(fig, "fig_4_4_feedback")


if __name__ == "__main__":
    for draw in (methodology, recommender_families, ner_example, use_case, dfd, erd, architecture, parser_flow, planner_flow, feedback_loop):
        draw()
    print(sorted(p.name for p in OUT.glob("*.png")))
