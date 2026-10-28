"""Feature #2: held-out evaluation of the preference parser with sentences written by other people.

The parser's own test sentences (tests/test_parser.py) were written while building the rules, so their 100% is
optimistic. Here friends / classmates write sentences the way they would type them, and say what the app should
understand:
    never        foods or groups that must never be served (allergies, "I don't eat pork", "I'm vegetarian")
    less_often   dislikes
    more_often   likes
Each item is compared with what parse_preferences() understood:
    recall     share of the expected items the parser found with the right label
    precision  share of the parser's items (for that label) the person also listed
    allergy recall  share of "never" items that are blocked - the safety number (missing one is the worst error)

Make the sheet:   python -m src.heldout --template      (data/evaluation/heldout_sentences.xlsx)
Score it:         python -m src.heldout                 (also part of python -m src.evaluate)
"""
import re
import sys

import pandas as pd

from src.allergens import ALLERGEN_GROUPS, tag_allergens
from src.config import DATA_DIR
from src.parser import CATEGORY_GROUPS, normalize_term, parse_preferences

HELDOUT_XLSX = DATA_DIR / "evaluation" / "heldout_sentences.xlsx"
COLUMNS = ["sentence", "never", "less_often", "more_often", "written_by"]
EXAMPLES = [
    ("I'm allergic to peanuts and I really don't like broccoli", "peanuts", "broccoli", "", "example"),
    ("No pork please, but I love grilled chicken and rice", "pork", "", "chicken, rice", "example"),
    ("I'm vegetarian and I'm not a fan of mushrooms", "meat", "mushrooms", "", "example"),
]
INSTRUCTIONS = [
    "How to fill in this sheet (thank you for helping!)",
    "",
    "1. In the 'sentence' column, write how YOU would tell a meal-planning app what you like, dislike or can't eat.",
    "   Write naturally - long or short, typos are fine. One sentence (or a few) per row. Please write 5-10 rows.",
    "2. Then fill in what the app SHOULD understand from your sentence (separate several items with commas):",
    "   never       - foods you must never get: allergies, foods you don't eat for health or religion (e.g. shrimp, pork, meat, gluten)",
    "   less_often  - foods you dislike but could eat (e.g. mushrooms)",
    "   more_often  - foods you like (e.g. grilled chicken, lentils)",
    "3. Put your first name or initials in 'written_by'. The grey rows marked 'example' are only examples - leave them.",
    "",
    "Please don't look at the app's code or test sentences first - the point is to see how it handles new wording.",
]


def make_template(path=HELDOUT_XLSX):
    """Excel sheet for the volunteers: a 'Sentences' tab (with 3 grey example rows) and an 'Instructions' tab."""
    from openpyxl.styles import Font, PatternFill
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(EXAMPLES, columns=COLUMNS).to_excel(xw, sheet_name="Sentences", index=False)
        pd.DataFrame({"Instructions": INSTRUCTIONS}).to_excel(xw, sheet_name="Instructions", index=False)
        ws = xw.sheets["Sentences"]
        for col, width in zip("ABCDE", (70, 28, 28, 28, 14)):
            ws.column_dimensions[col].width = width
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in ws.iter_rows(min_row=2, max_row=1 + len(EXAMPLES)):
            for cell in row:
                cell.fill = PatternFill("solid", fgColor="DDDDDD")
        xw.sheets["Instructions"].column_dimensions["A"].width = 120
    return path


def load_sentences(path=HELDOUT_XLSX):
    """The volunteers' rows (example rows and empty rows left out)."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_excel(path, sheet_name="Sentences", dtype=str).fillna("")
    df = df[df["sentence"].str.strip().astype(bool) & (df["written_by"].str.strip().str.lower() != "example")]
    return df.reset_index(drop=True)


def _items(cell):
    return [normalize_term(x) for x in re.split(r"[,;/]| and ", str(cell)) if normalize_term(x)]


def _groups(term):
    """Allergen groups a word stands for (dairy -> milk, nuts -> peanut + tree_nut, shrimp -> shellfish...)."""
    return set(CATEGORY_GROUPS.get(term, set())) | set(tag_allergens(term))


def _found(term, parsed_terms, groups=None):
    """Did the parser find this item? Same word (either contains the other) or, for "never", the right groups."""
    if any(term == p or term in p or p in term for p in parsed_terms):
        return True
    return bool(groups is not None and _groups(term) and _groups(term) <= groups)


def score(df):
    """Per-sentence results and overall metrics."""
    rows = []
    for r in df.itertuples():
        p = parse_preferences(r.sentence)
        s = p.summary()
        never_terms = set(s["block"]) | set(s["block_diet_flags"])
        expected = {"never": _items(r.never), "less_often": _items(r.less_often), "more_often": _items(r.more_often)}
        got = {"never": never_terms, "less_often": set(s["avoid"]), "more_often": set(s["prefer"])}
        res = {"sentence": r.sentence, "written_by": r.written_by}
        for label, exp in expected.items():
            groups = p.block_groups if label == "never" else None
            hits = [t for t in exp if _found(t, got[label], groups)]
            extra = [t for t in got[label] if not any(_found(e, {t}) for e in exp)]
            res[f"{label}_expected"], res[f"{label}_found"], res[f"{label}_extra"] = len(exp), len(hits), len(extra)
            res[f"{label}_missed"] = ", ".join(t for t in exp if t not in hits)
        allergy = [t for t in expected["never"] if _groups(t) & set(ALLERGEN_GROUPS)]
        res["allergy_expected"] = len(allergy)
        res["allergy_blocked"] = sum(_groups(t) <= p.block_groups for t in allergy)
        res["understood"] = "; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in got.items() if v)
        rows.append(res)
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}
    metrics = {"sentences": len(out), "writers": out["written_by"].str.strip().str.lower().nunique()}
    for label in ("never", "less_often", "more_often"):
        e, f, x = out[f"{label}_expected"].sum(), out[f"{label}_found"].sum(), out[f"{label}_extra"].sum()
        metrics[f"{label}_recall"] = f / e if e else None
        metrics[f"{label}_precision"] = f / (f + x) if (f + x) else None
    a = out["allergy_expected"].sum()
    metrics["allergy_recall"] = out["allergy_blocked"].sum() / a if a else None
    metrics["allergy_items"] = int(a)
    return out, metrics


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    if "--template" in sys.argv:
        print(f"Saved {make_template()} - send it to people who did not build the app.")
    else:
        rows, m = score(load_sentences())
        if rows.empty:
            print(f"No sentences yet in {HELDOUT_XLSX} (run with --template to create the sheet).")
        else:
            print(m)
            print(rows[["sentence", "understood", "never_missed", "less_often_missed", "more_often_missed"]].to_string())
