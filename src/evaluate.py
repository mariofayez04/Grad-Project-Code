"""Phase 8: evaluation - every number for the report, in one run.

  1. Macro adherence      - how close daily totals land to the targets (many random users)
  2. Allergy safety       - allergens that reach a plan for users with stated allergies (must be 0)
  3. Allergen tagger      - precision / recall on hand-labelled food descriptions
  4. Preference parser    - label accuracy on hand-labelled sentences; ablation NER vs dictionary vs both
  5. Embeddings           - neighbour agreement, raw USDA text vs cleaned text + category
  6. Feedback loop        - acceptance over simulated weeks, with vs without feedback
  7. Speed                - time to build a weekly plan
  8. Nutrients            - days on target / over the safety caps, nutrient-aware planner vs macros only (#7, #10)
  9. Diet modes           - sodium, sugar and carbs with the diabetes-friendly / low-salt modes (#10)
 10. Egyptian dishes      - dish meals per week for each setting, and accuracy (#9)
 11. Held-out sentences   - parser on sentences written by other people (data/evaluation/heldout_sentences.xlsx)
 12. Real-user pilot      - what real use of the app shows (data/app.db): meals eaten, swaps, choices, ratings

Run:  python -m src.evaluate            (about 10-15 minutes)
Writes data/evaluation/results.md plus CSV files.
"""
import time
import warnings

import numpy as np
import pandas as pd

from src.allergens import tag_allergens
from src.calculator import UserProfile, calculate_targets
from src.config import DATA_DIR, FOODS_CLEAN
from src.embeddings import FoodEmbeddings, embed_text, encode, neighbor_agreement
from src.nutrients import LABELS, MICROS, day_micros, filter_modes, status
from src.parser import apply_to_foods, parse_preferences
from src.planner import MealPlanner, generate_week_plan, taste_from_preferences

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
        items = plan.to_dataframe().merge(foods[["fdc_id", "allergens", "meal_role"]], on="fdc_id")
        tags = items["allergens"].fillna("").str.split("|").map(set)
        violations = int(tags.map(lambda t: bool(t & set(chosen))).sum())
        rows.append({"allergies": ", ".join(chosen), "parsed_groups": ", ".join(sorted(prefs.block_groups)),
                     "all_detected": set(chosen) <= prefs.block_groups, "items_served": len(items), "violations": violations,
                     "dishes_served": int((items["meal_role"] == "dish").sum())})
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


# 8 ---------------------------------------------------------------------------------------------------------
def eval_nutrients(foods, emb, n=12):
    """Days on target and over the caps, with the nutrient-aware planner vs the same planner on macros only."""
    rows = []
    for i, p in enumerate(random_profiles(n, seed=2)):
        targets = calculate_targets(p)
        prefs = parse_preferences(p.preferences_text)
        allowed, _ = apply_to_foods(foods, prefs)
        for mode, nutrition in (("nutrient-aware (final)", targets), ("macros only", None)):
            plan = MealPlanner(allowed, emb, taste_from_preferences(prefs), None, i, nutrition=nutrition).plan_week(targets)
            days = day_micros(plan.to_dataframe(), foods)
            for n_ in MICROS:
                t = targets.micros[n_]
                for v in days[n_]:
                    rows.append({"planner": mode, "nutrient": n_, "status": status(v, t),
                                 "over_cap": bool(t.get("upper")) and v > t["upper"], "has_cap": bool(t.get("upper")),
                                 "macro_err": np.mean(list(plan.adherence().values()))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "nutrients.csv", index=False)
    return df


# 9 ---------------------------------------------------------------------------------------------------------
MODES = {"none": "", "diabetes-friendly": "diabetes", "low-salt": "low_sodium", "both": "diabetes|low_sodium"}


def eval_modes(foods, emb, n=8):
    rows = []
    for name, modes in MODES.items():
        for i, p in enumerate(random_profiles(n, seed=3)):
            p.health_modes = modes
            plan, targets, _ = generate_week_plan(p, foods=foods, embeddings=emb, seed=i)
            items = plan.to_dataframe()
            days = day_micros(items, foods)
            served = foods[foods["fdc_id"].isin(items["fdc_id"])]
            daily = items.groupby("day_index")[["calories", "carbs_g"]].sum()
            rows.append({"mode": name, "sodium_mg": days["sodium_mg"].mean(), "sugar_g": days["sugar_g"].mean(),
                         "carbs_share": (4 * daily["carbs_g"] / daily["calories"]).mean(),
                         "sodium_ok": (days["sodium_mg"] <= targets.micros["sodium_mg"]["value"]).mean(),
                         "sugar_ok": ((days["sugar_g"] <= targets.micros["sugar_g"]["value"]).mean()
                                      if targets.micros["sugar_g"]["kind"] == "max" else np.nan),
                         "excluded_served": len(served) - len(filter_modes(served, modes)),
                         "macro_err": np.mean(list(plan.adherence().values()))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "diet_modes.csv", index=False)
    return df


# 10 --------------------------------------------------------------------------------------------------------
def eval_dishes(foods, emb, n=8):
    dish_ids = set(foods.loc[foods["meal_role"] == "dish", "fdc_id"])
    rows = []
    for setting in ("never", "sometimes", "often"):
        for i, p in enumerate(random_profiles(n, seed=4)):
            p.egyptian_dishes = setting
            plan, _, _ = generate_week_plan(p, foods=foods, embeddings=emb, seed=i)
            items = plan.to_dataframe()
            dishes = items[items["fdc_id"].isin(dish_ids)]
            rows.append({"setting": setting, "dish_meals": dishes.groupby(["day_index", "slot"]).ngroups,
                         "different_dishes": dishes["fdc_id"].nunique(), "meals": items.groupby(["day_index", "slot"]).ngroups,
                         "macro_err": np.mean(list(plan.adherence().values()))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dishes.csv", index=False)
    return df


# 11 --------------------------------------------------------------------------------------------------------
def eval_heldout():
    from src.heldout import load_sentences, score
    rows, metrics = score(load_sentences())
    if not rows.empty:
        rows.to_csv(OUT / "heldout_results.csv", index=False)
    return rows, metrics


# 12 --------------------------------------------------------------------------------------------------------
def eval_pilot(out=None):
    """Real use of the app (data/app.db), aggregated - no names or profiles in the report.
    out: folder for pilot_comments.csv (default data/evaluation)."""
    from src import db
    from src.feedback import meal_status
    data = db.pilot_data()
    items, surveys = data["items"], data["surveys"]
    weeks = []
    for (uid, week), g in items.groupby(["user_id", "week_number"]):
        weeks.append({"user": uid, "week": week, "eaten": meal_status(g)["eaten"].mean()})
    weeks = pd.DataFrame(weeks, columns=["user", "week", "eaten"])
    if not surveys.empty:
        surveys[surveys["comment"].str.strip().astype(bool)][["week_number", "comment"]].to_csv(
            (out or OUT) / "pilot_comments.csv", index=False)  # comments only, without who wrote them
    return data, weeks


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
    print("+ nutrients...")
    nut = eval_nutrients(foods, emb)
    print("+ diet modes...")
    modes = eval_modes(foods, emb)
    print("+ Egyptian dishes...")
    dishes = eval_dishes(foods, emb)
    print("+ held-out sentences and pilot...")
    held_rows, held = eval_heldout()
    pilot, pilot_weeks = eval_pilot()

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

    # 8: nutrients
    met = (nut.assign(on_target=nut["status"] == "ok").groupby(["nutrient", "planner"])["on_target"].mean()
           .unstack("planner"))
    nut_table = pd.DataFrame({"Nutrient": [LABELS[n_][0] for n_ in met.index],
                              "Days on target - macros only": met["macros only"].map(pct).to_numpy(),
                              "Days on target - nutrient-aware (final)": met["nutrient-aware (final)"].map(pct).to_numpy()})
    nut_table = nut_table[nut_table["Nutrient"] != "Sugars"]
    caps = nut[nut["has_cap"]].groupby(["nutrient", "planner"])["over_cap"].mean().unstack("planner")
    cap_table = pd.DataFrame({"Nutrient": [LABELS[n_][0] for n_ in caps.index],
                              "Days over cap - macros only": caps["macros only"].map(pct).to_numpy(),
                              "Days over cap - nutrient-aware (final)": caps["nutrient-aware (final)"].map(pct).to_numpy()})
    nut_err = nut.groupby("planner")["macro_err"].mean()
    # 9: diet modes
    mt = modes.groupby("mode", sort=False).agg(sodium=("sodium_mg", "mean"), sugar=("sugar_g", "mean"),
                                               carbs=("carbs_share", "mean"), sodium_ok=("sodium_ok", "mean"),
                                               sugar_ok=("sugar_ok", "mean"), excluded=("excluded_served", "sum"),
                                               err=("macro_err", "mean")).reset_index()
    mode_table = pd.DataFrame({"Mode": mt["mode"], "Sodium / day": mt["sodium"].map("{:,.0f} mg".format),
                               "Days under sodium limit": mt["sodium_ok"].map(pct), "Sugars / day": mt["sugar"].map("{:.0f} g".format),
                               "Days under sugar limit": mt["sugar_ok"].map(lambda v: "-" if pd.isna(v) else pct(v)),
                               "Carbs (% kcal)": mt["carbs"].map(pct), "Left-out foods served": mt["excluded"],
                               "Macro error": mt["err"].map("{:.1f}%".format)})
    # 10: dishes
    dt = dishes.groupby("setting", sort=False).agg(m=("dish_meals", "mean"), lo=("dish_meals", "min"), hi=("dish_meals", "max"),
                                                    d=("different_dishes", "mean"), meals=("meals", "mean"),
                                                    err=("macro_err", "mean")).reset_index()
    dish_table = pd.DataFrame({"Setting": dt["setting"], "Dish meals / week": [f"{m:.1f} (range {lo}-{hi})" for m, lo, hi in zip(dt.m, dt.lo, dt.hi)],
                               "Share of meals": (dt["m"] / dt["meals"]).map(pct), "Different dishes / week": dt["d"].map("{:.1f}".format),
                               "Macro error": dt["err"].map("{:.1f}%".format)})
    # 11: held-out
    if held:
        fmt = lambda v: "-" if v is None else pct(v)
        held_text = (f"{held['sentences']} sentences by {held['writers']} people who did not build the system.\n\n"
                     + md_table(pd.DataFrame({"Label": ["Never (allergies, diet)", "Less often (dislikes)", "More often (likes)"],
                                              "Recall": [fmt(held[f"{k}_recall"]) for k in ("never", "less_often", "more_often")],
                                              "Precision": [fmt(held[f"{k}_precision"]) for k in ("never", "less_often", "more_often")]}))
                     + f"\n\n- Allergy items blocked (safety): **{fmt(held['allergy_recall'])}** of {held['allergy_items']}"
                     + "\n- Per-sentence results: `data/evaluation/heldout_results.csv`")
    else:
        held_text = ("Not collected yet. Send `data/evaluation/heldout_sentences.xlsx` (make it with "
                     "`python -m src.heldout --template`) to 5-10 people who did not build the app, put the filled sheet "
                     "back in `data/evaluation/`, and run this script again.")
    # 12: pilot
    if pilot_weeks.empty:
        pilot_text = (f"No finished weeks yet ({pilot['users']} profiles, {pilot['plans']} plans in the database). Ask testers "
                      "to use the app for a week, tick what they eat, rate the week and press \"Finish week\".")
    else:
        by_week = pilot_weeks.groupby("week")["eaten"].agg(["mean", "size"]).reset_index()
        sv = pilot["surveys"]
        pilot_text = (f"- Profiles: {pilot['users']}, plans made: {pilot['plans']}, finished weeks: {len(pilot_weeks)} "
                      f"(by {pilot_weeks['user'].nunique()} " + ("person" if pilot_weeks['user'].nunique() == 1 else "people") + ")\n"
                      f"- Meals ticked as eaten, by week of use:\n\n"
                      + md_table(pd.DataFrame({"Week of use": by_week["week"], "People": by_week["size"],
                                               "Meals eaten": by_week["mean"].map(pct)}))
                      + f"\n\n- Swaps per plan: {pilot['swaps']['n'].sum() / max(pilot['plans'], 1):.1f}; "
                      f"New meal / day choices: {len(pilot['choices'])} "
                      f"({pct((pilot['choices']['kept'] == 'new').mean()) if len(pilot['choices']) else '-'} kept the new one)\n"
                      f"- People who logged their weight 2+ times: {(pilot['weights']['n'] >= 2).sum()}\n"
                      + (f"- Ratings ({len(sv)} weeks rated, 1-5): liked the meals **{sv['liked'].mean():.1f}**, realistic "
                         f"portions **{sv['portions'].mean():.1f}**, easy to follow **{sv['easy'].mean():.1f}**; would keep "
                         f"using it: " + ", ".join(f"{k} {pct(v)}" for k, v in sv['would_use'].value_counts(normalize=True).items())
                         + f"\n- Comments: {int(sv['comment'].str.strip().astype(bool).sum())} (in `pilot_comments.csv`, without names)"
                         if len(sv) else "- No ratings yet."))

    report = f"""# Evaluation results

Generated by `python -m src.evaluate`. Food table: {len(foods):,} foods from USDA SR Legacy, incl. {int((foods['meal_role'] == 'dish').sum())} Egyptian dishes built from USDA ingredients.
Plans use the final planner: nutrient-aware, Egyptian dishes "sometimes", no diet mode (unless stated).

## 1. Macro adherence (30 random users x 7 days = {len(adh)} days)
Absolute % difference between each day's planned total and the user's target.

{md_table(adh_table)}

By goal (mean % error):

{md_table(by_goal)}

## 2. Allergy safety ({len(safety)} random users with 1-3 stated allergies)
- Allergies fully detected by the parser: **{pct(safety['all_detected'].mean())}**
- Food items served: {safety['items_served'].sum():,}
- Items containing a stated allergen: **{safety['violations'].sum()}**
- Of which Egyptian dishes served: {safety['dishes_served'].sum()} (their allergens come from their ingredients)

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

## 8. Nutrients ({len(nut) // len(MICROS) // 2} planned days; same users, same seeds) - nutrient-aware planner vs macros only
"On target" = at least the daily minimum (fiber, potassium, calcium, iron) or at most the limit (saturated fat, sodium).

{md_table(nut_table)}

Days over a daily cap (iron 45 mg and calcium 2,500 / 2,000 mg = safe upper limits; fiber 2x its target; sodium and
saturated fat their limits):

{md_table(cap_table)}

Macro error: {nut_err['macros only']:.1f}% (macros only) vs {nut_err['nutrient-aware (final)']:.1f}% (nutrient-aware).

## 9. Diet modes ({len(modes) // len(MODES)} random users per mode, 1 week each)
{md_table(mode_table)}

Limits: sodium 2,300 mg (1,500 mg low-salt); sugars 10% of calories (diabetes-friendly); carbs <= 40% of calories
(diabetes-friendly). "Left-out foods served" counts foods the mode should exclude that still reached a plan.

## 10. Egyptian dishes ({len(dishes) // 3} random users per setting, 1 week each)
{md_table(dish_table)}

## 11. Held-out sentences (parser on wording it was not built with)
{held_text}

## 12. Real-user pilot (data/app.db, aggregated)
{pilot_text}

## Notes and limitations
- The allergen-tagger and parser cases (sections 3-4) were used while developing the rules, so their scores are
  optimistic. A held-out set written by people who did not build the system gives a fairer estimate.
- Allergen tags come from food names; ingredients not named in a description cannot be detected.
- Parser test sentences were written by the developers; real users may phrase things differently (typos, "except").
- The feedback evaluation uses simulated users, not real people (section 12 adds real use once testers finish weeks).
- Nutrient values: USDA gives total sugars (not added sugars); some missing values are estimated from similar foods;
  salt added while cooking is not counted.
- Egyptian dishes are home-style recipes estimated from USDA ingredients, not lab-measured dishes.
"""
    (OUT / "results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved -> {OUT / 'results.md'}")


if __name__ == "__main__":
    main()
