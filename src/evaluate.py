"""Phase 8: evaluation - every number for the report, in one run.

  1. Macro adherence      - how close daily totals land to the targets (many random users)
  2. Allergy safety       - allergens that reach a plan for users with stated allergies (must be 0)
  3. Allergen tagger      - precision / recall on hand-labelled food descriptions
  4. Preference parser    - label accuracy on hand-labelled sentences; ablation NER vs dictionary vs both
  5. Embeddings           - neighbour agreement, raw USDA text vs cleaned text + category
  6. Feedback loop        - acceptance over simulated weeks, with vs without feedback
  7. Speed                - time to build a weekly plan

Run:  python -m src.evaluate            (about 3-5 minutes)
Writes data/evaluation/results.md plus CSV files.
"""
import time
import warnings

import numpy as np
import pandas as pd

from src.allergens import tag_allergens
from src.calculator import UserProfile
from src.config import DATA_DIR, FOODS_CLEAN
from src.embeddings import FoodEmbeddings, embed_text, encode, neighbor_agreement
from src.parser import parse_preferences
from src.planner import generate_week_plan

OUT = DATA_DIR / "evaluation"
ALLERGY_WORDS = {"milk": "dairy", "egg": "eggs", "peanut": "peanuts", "tree_nut": "tree nuts", "shellfish": "shellfish",
                 "fish": "fish", "soy": "soy", "wheat": "wheat", "sesame": "sesame"}
PREF_TEXTS = ["", "I love chicken and rice", "I hate fish", "I'm vegetarian", "I don't like beef, I enjoy lentils",
              "I'm halal and love lamb", "I prefer eggs and yogurt for breakfast"]


def random_profiles(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        sex = rng.choice(["male", "female"])
        h = rng.uniform(165, 195) if sex == "male" else rng.uniform(150, 180)
        bmi = rng.uniform(19, 32)
        out.append(UserProfile(f"U{i}", int(rng.integers(18, 70)), sex, round(bmi * (h / 100) ** 2, 1), round(h),
                               int(rng.integers(1, 4)), str(rng.choice(["maintain", "lose", "gain"])),
                               PREF_TEXTS[i % len(PREF_TEXTS)]))
    return out


# 1 ---------------------------------------------------------------------------------------------------------
def eval_adherence(foods, emb, n=30):
    rows, times = [], []
    for i, p in enumerate(random_profiles(n)):
        t0 = time.time()
        plan, targets, _ = generate_week_plan(p, foods=foods, embeddings=emb, seed=i)
        times.append(time.time() - t0)
        for day in plan.days:
            tot = day.totals()
            rows.append({"user": p.name, "goal": p.goal, "target_kcal": targets.calories, "day": day.day,
                         **{f"err_{m}": abs(tot[m] - plan.targets[m]) / plan.targets[m] * 100 for m in tot}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "adherence.csv", index=False)
    return df, times


# 2 ---------------------------------------------------------------------------------------------------------
def eval_allergy_safety(foods, emb, n=20):
    rng = np.random.default_rng(1)
    groups = list(ALLERGY_WORDS)
    rows = []
    for i, p in enumerate(random_profiles(n, seed=1)):
        chosen = list(rng.choice(groups, size=int(rng.integers(1, 4)), replace=False))
        p.preferences_text = "I'm allergic to " + " and ".join(ALLERGY_WORDS[g] for g in chosen)
        plan, _, prefs = generate_week_plan(p, foods=foods, embeddings=emb, seed=i)
        items = plan.to_dataframe().merge(foods[["fdc_id", "allergens"]], on="fdc_id")
        tags = items["allergens"].fillna("").str.split("|").map(set)
        violations = int(tags.map(lambda t: bool(t & set(chosen))).sum())
        rows.append({"allergies": ", ".join(chosen), "parsed_groups": ", ".join(sorted(prefs.block_groups)),
                     "all_detected": set(chosen) <= prefs.block_groups, "items_served": len(items), "violations": violations})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "allergy_safety.csv", index=False)
    return df


# 3 ---------------------------------------------------------------------------------------------------------
def eval_allergen_tagger():
    from tests.test_allergens import CASES
    tp = fn = fp = tn = 0
    for desc, cat, must, must_not in CASES:
        tags = set(tag_allergens(desc, cat))
        tp += sum(g in tags for g in must)
        fn += sum(g not in tags for g in must)
        fp += sum(g in tags for g in must_not)
        tn += sum(g not in tags for g in must_not)
    return {"cases": len(CASES), "decisions": tp + fn + fp + tn, "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "missed_allergens": fn, "false_alarms": fp}


# 4 ---------------------------------------------------------------------------------------------------------
def eval_parser():
    from tests.test_parser import CASES
    modes = {"NER + dictionary (final)": (True, True), "NER model only": (True, False), "Dictionary only": (False, True)}
    rows = []
    for mode, (use_ner, use_dict) in modes.items():
        correct = total = sentences_ok = group_hits = group_total = 0
        for text, labels, groups, flags in CASES:
            p = parse_preferences(text, use_ner=use_ner, use_dict=use_dict)
            got = {m.term: m.label for m in p.mentions if m.source != "diet" and m.cue != "safety-net"}
            ok = [got.get(t) == lab for t, lab in labels.items()]
            correct += sum(ok)
            total += len(ok)
            group_hits += len(groups & p.block_groups)
            group_total += len(groups)
            sentences_ok += all(ok) and groups <= p.block_groups and flags <= p.block_flags
        rows.append({"mode": mode, "label_accuracy": correct / total, "allergy_group_recall": group_hits / group_total,
                     "sentences_fully_correct": sentences_ok / len(CASES)})
    return pd.DataFrame(rows), len(CASES)


# 5 ---------------------------------------------------------------------------------------------------------
def eval_embeddings(foods, emb):
    raw = encode(foods["description"].tolist())
    cleaned = encode([embed_text(d) for d in foods["description"]])
    rows = []
    for name, v in [("Raw USDA description", raw), ("Cleaned text", cleaned), ("Cleaned text + category (final)", emb.vectors)]:
        rows.append({"embedded_text": name, "same_meal_role@5": neighbor_agreement(v, foods["meal_role"]),
                     "same_category@5": neighbor_agreement(v, foods["food_category"])})
    return pd.DataFrame(rows)


# 6 ---------------------------------------------------------------------------------------------------------
def eval_feedback():
    from src.simulate import experiment
    df = experiment(weeks=4, seeds=(0, 1, 2))
    df.to_csv(OUT / "feedback_simulation.csv", index=False)
    return df


# report ------------------------------------------------------------------------------------------------------
def pct(x):
    return f"{100 * x:.1f}%"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    foods = pd.read_csv(FOODS_CLEAN)
    emb = FoodEmbeddings(foods)
    parse_preferences("warm up the NER model")

    print("1/6 macro adherence...")
    adh, times = eval_adherence(foods, emb)
    print("2/6 allergy safety...")
    safety = eval_allergy_safety(foods, emb)
    print("3/6 allergen tagger...")
    tagger = eval_allergen_tagger()
    print("4/6 parser...")
    parser, n_sent = eval_parser()
    print("5/6 embeddings...")
    embq = eval_embeddings(foods, emb)
    print("6/6 feedback simulation...")
    fb = eval_feedback()

    m = ["calories", "protein_g", "carbs_g", "fat_g"]
    adh_table = pd.DataFrame({
        "Macro": ["Calories", "Protein", "Carbs", "Fat"],
        "Mean error": [f"{adh[f'err_{x}'].mean():.1f}%" for x in m],
        "Median error": [f"{adh[f'err_{x}'].median():.1f}%" for x in m],
        "95th percentile": [f"{adh[f'err_{x}'].quantile(.95):.1f}%" for x in m],
        "Days within ±10%": [pct((adh[f"err_{x}"] <= 10).mean()) for x in m],
    })
    by_goal = adh.groupby("goal")[[f"err_{x}" for x in m]].mean().round(1).reset_index()
    by_goal.columns = ["Goal", "Calories %", "Protein %", "Carbs %", "Fat %"]

    fb_table = (fb.groupby(["scenario", "feedback", "week"])["acceptance"].mean().unstack("week")
                .map(pct).reset_index())
    fb_table.columns = ["Hidden dislike", "Mode"] + [f"Week {w}" for w in fb_table.columns[2:]]
    dis_table = (fb.groupby(["scenario", "feedback", "week"])["disliked_items"].mean().unstack("week").round(1).reset_index())
    dis_table.columns = ["Hidden dislike", "Mode"] + [f"Week {w}" for w in dis_table.columns[2:]]

    parser_fmt = parser.assign(**{c: parser[c].map(pct) for c in parser.columns if c != "mode"})
    parser_fmt.columns = ["Mode", "Label accuracy", "Allergy group recall", "Sentences fully correct"]
    embq_fmt = embq.assign(**{c: embq[c].map(pct) for c in embq.columns if c != "embedded_text"})
    embq_fmt.columns = ["Text embedded", "Same meal role (5-NN)", "Same category (5-NN)"]

    report = f"""# Evaluation results

Generated by `python -m src.evaluate`. Food table: {len(foods):,} foods from USDA SR Legacy.

## 1. Macro adherence (30 random users x 7 days = {len(adh)} days)
Absolute % difference between each day's planned total and the user's target.

{md_table(adh_table)}

By goal (mean % error):

{md_table(by_goal)}

## 2. Allergy safety ({len(safety)} random users with 1-3 stated allergies)
- Allergies fully detected by the parser: **{pct(safety['all_detected'].mean())}**
- Food items served: {safety['items_served'].sum():,}
- Items containing a stated allergen: **{safety['violations'].sum()}**

## 3. Allergen tagger (deterministic dictionary, {tagger['cases']} hand-labelled foods, {tagger['decisions']} decisions)
- Precision: **{pct(tagger['precision'])}**   Recall: **{pct(tagger['recall'])}**
- Missed allergens: {tagger['missed_allergens']}   False alarms: {tagger['false_alarms']}
- Includes traps such as eggplant, peanut butter, coconut milk, water chestnut, gluten-free pasta, corn tortillas.

## 4. Preference parser ({n_sent} hand-labelled sentences) - ablation
{md_table(parser_fmt)}

## 5. Food embeddings (all-MiniLM-L6-v2) - share of each food's 5 nearest neighbours with the same label
{md_table(embq_fmt)}

## 6. Weekly feedback loop (simulated users with hidden dislikes, mean of 3 runs)
Share of meals eaten:

{md_table(fb_table)}

Disliked foods served per week:

{md_table(dis_table)}

## 7. Speed
- Weekly plan generation: mean **{np.mean(times):.2f} s**, max {np.max(times):.2f} s (after models are loaded).

## Notes and limitations
- The allergen-tagger and parser cases (sections 3-4) were used while developing the rules, so their scores are
  optimistic. A held-out set written by people who did not build the system gives a fairer estimate.
- Allergen tags come from food names; ingredients not named in a description cannot be detected.
- Parser test sentences were written by the developers; real users may phrase things differently (typos, "except").
- The feedback evaluation uses simulated users, not real people.
"""
    (OUT / "results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved -> {OUT / 'results.md'}")


if __name__ == "__main__":
    main()
