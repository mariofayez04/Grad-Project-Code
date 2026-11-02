"""Simulated testers: scripted users who go through the same steps as a person using the app.

These are NOT real people. Each tester has a profile, typed preferences, foods they dislike but never typed, and a
habit of eating or skipping meals (all invented here). They sign up with a PIN, get a plan, swap foods, ask for a new
meal, tick meals, log their weight and finish the week, for 4 weeks. The database writes, planner and feedback
functions are the app's own, on a throwaway database; the Streamlit screens are not involved. Every tester runs twice,
with the weekly learning on and off, so the two can be compared.

What this can show: plans stay accurate and safe over several weeks, and the system learns from behaviour.
What it can't show: whether real people like the meals, so no satisfaction scores are produced here.

Also scores the parser on three sets of sentences written by an AI assistant, not by volunteers:
    new_sentences.csv     used to find the problem; the fix was built by looking at its failures
    new_sentences_2.csv   written before the fix, but its failures were seen while building the fix
    new_sentences_3.csv   written after the rules were frozen and scored once: the fair number for new wording

Run:  python -m src.testers     (about 15 min)
"""
import os
import re
import sys
import tempfile
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.calculator import UserProfile, calculate_targets
from src.config import DATA_DIR, FOODS_CLEAN
from src.dishes import load_recipes
from src.feedback import acceptance_rate, food_penalties, next_week_inputs
from src.heldout import score
from src.nutrients import filter_modes
from src.parser import apply_to_foods, parse_preferences
from src.planner import DAYS, MACROS, MealPlanner, daily_targets, plan_slots, swap_options, taste_from_preferences
from src.simulate import SCENARIOS

OUT = DATA_DIR / "evaluation"
SENTENCE_SETS = {"Set 1 (used to build the fix)": ("new_sentences.csv", "18 / 27"),
                 "Set 2 (failures seen while fixing)": ("new_sentences_2.csv", "7 / 30"),
                 "Set 3 (kept apart, scored once)": ("new_sentences_3.csv", "not measured")}
WEEKS = 4
SWAP_RATE, NEW_MEAL_RATE = 0.30, 0.15   # chance of acting on a disliked food instead of just skipping the meal
SKIP_DISLIKED = 0.90                    # chance of skipping a meal that still has a disliked food
NOTHING = r"(?!x)x"
DISLIKES = {**SCENARIOS, "dislikes dairy": r"\b(?:cheese|yogurt|milk)\b",
            "dislikes beans & lentils": r"\b(?:beans?|lentils?|chickpeas?)\b", "no hidden dislike": NOTHING}

# age, sex, kg, cm, activity, goal, typed text, meals a day, diet modes, Egyptian dishes, hidden dislike, allergy
# groups, diet rule, how the text is worded ("clean" = usual wording, "messy" = typos or indirect wording)
TESTERS = [
    (24, "male", 82, 178, 2, "lose", "I'm allergic to shrimp and I hate mushrooms", 4, "", "sometimes", "dislikes fish & seafood", {"shellfish"}, "", "clean"),
    (31, "female", 64, 163, 1, "maintain", "I love chicken and rice", 4, "", "often", "dislikes beef & lamb", set(), "", "clean"),
    (45, "male", 95, 180, 1, "lose", "I don't like fish", 3, "diabetes", "sometimes", "dislikes eggs & cottage cheese", set(), "", "clean"),
    (52, "female", 78, 160, 1, "lose", "I'm allergic to peanuts", 4, "low_sodium", "sometimes", "dislikes dairy", {"peanut"}, "", "clean"),
    (19, "male", 62, 175, 3, "gain", "I love eggs and oats", 5, "", "sometimes", "dislikes beans & lentils", set(), "", "clean"),
    (28, "female", 55, 158, 2, "maintain", "I'm vegetarian and I love lentils", 4, "", "often", "dislikes dairy", set(), "vegetarian", "clean"),
    (36, "male", 75, 172, 2, "maintain", "I don't eat pork and I'm allergic to sesame", 4, "", "often", "dislikes fish & seafood", {"sesame"}, "halal", "clean"),
    (60, "female", 70, 158, 1, "maintain", "I'm allergic to milk", 3, "diabetes|low_sodium", "sometimes", "dislikes beef & lamb", {"milk"}, "", "clean"),
    (22, "female", 68, 170, 3, "lose", "I hate broccoli", 5, "", "never", "dislikes eggs & cottage cheese", set(), "", "clean"),
    (40, "male", 88, 183, 2, "lose", "I'm allergic to eggs and I love beef", 4, "", "sometimes", "dislikes beans & lentils", {"egg"}, "", "clean"),
    (27, "male", 70, 176, 3, "gain", "I'm allergic to tree nuts", 5, "", "often", "no hidden dislike", {"tree_nut"}, "", "clean"),
    (33, "female", 60, 165, 2, "maintain", "I'm vegan", 4, "", "never", "no hidden dislike", {"milk", "egg"}, "vegetarian", "clean"),
    (48, "male", 102, 176, 1, "lose", "I'm allergic to gluten", 4, "diabetes", "sometimes", "dislikes dairy", {"gluten"}, "", "clean"),
    (55, "female", 82, 165, 1, "lose", "I don't like beef", 3, "low_sodium", "often", "dislikes fish & seafood", set(), "", "clean"),
    (20, "female", 50, 160, 2, "gain", "I love pasta and yogurt", 4, "", "sometimes", "dislikes beef & lamb", set(), "", "clean"),
    (35, "male", 80, 175, 2, "maintain", "I'm allergic to soy and fish", 4, "", "sometimes", "dislikes eggs & cottage cheese", {"soy", "fish"}, "", "clean"),
    (29, "female", 72, 168, 2, "lose", "", 4, "", "often", "dislikes beans & lentils", set(), "", "clean"),
    (67, "male", 76, 170, 1, "maintain", "I don't eat pork", 3, "low_sodium", "sometimes", "no hidden dislike", set(), "halal", "clean"),
    (26, "male", 77, 180, 3, "gain", "", 5, "", "sometimes", "dislikes fish & seafood", set(), "", "clean"),
    (38, "female", 66, 162, 2, "lose", "I'm pescatarian", 4, "", "sometimes", "dislikes dairy", set(), "pescatarian", "clean"),
    # Same flow, but the allergy is worded the way people type: these show what happens when the parser misses one.
    (30, "male", 79, 177, 2, "maintain", "im alergic to peanuts", 4, "", "sometimes", "no hidden dislike", {"peanut"}, "", "messy"),
    (42, "female", 69, 164, 2, "maintain", "Shellfish gives me hives", 4, "", "sometimes", "no hidden dislike", {"shellfish"}, "", "messy"),
    (25, "female", 58, 161, 2, "maintain", "Milk upsets my stomach so I stay away from it", 4, "", "sometimes", "no hidden dislike", {"milk"}, "", "messy"),
    (34, "male", 84, 181, 2, "lose", "I have celiac disease", 4, "", "sometimes", "no hidden dislike", {"gluten"}, "", "messy"),
]


def food_texts(foods):
    """What a tester can read about each food: its name, and for an Egyptian dish also its ingredients."""
    text = dict(zip(foods["fdc_id"], foods["description"]))
    for dish, group in load_recipes().groupby("dish_fdc_id"):
        text[dish] = " ".join([str(text.get(dish, ""))] + [str(x) for x in group["description"]])
    return text


def rule_broken(items, foods, allergies, diet):
    """Foods of this week that break a stated allergy or diet: (allergen items, diet items)."""
    rows = items.merge(foods[["fdc_id", "allergens", "is_vegetarian", "contains_pork", "contains_alcohol", "food_category"]], on="fdc_id")
    tags = rows["allergens"].fillna("").str.split("|").map(set)
    allergen = int(tags.map(lambda t: bool(t & allergies)).sum())
    bad = pd.Series(False, index=rows.index)
    if diet == "vegetarian":
        bad = ~rows["is_vegetarian"].astype(bool)
    elif diet == "halal":
        bad = rows["contains_pork"].astype(bool) | rows["contains_alcohol"].astype(bool)
    elif diet == "pescatarian":
        fish = rows["food_category"].eq("Finfish and Shellfish Products")
        bad = ~rows["is_vegetarian"].astype(bool) & ~fish
    return allergen, int(bad.sum())


def run_week_actions(plan_id, plan, user, allowed, foods, emb, texts, rx, adherence, rng, learning):
    """One week of use: act on some disliked foods, then tick or skip each meal. Returns (swaps, new meals, kept new)."""
    swaps = new_meals = kept_new = 0
    base_of = dict(zip(foods["fdc_id"], foods["base_name"]))
    n_meals = int(plan["targets"]["meals_per_day"])
    disliked = lambda df: df[[bool(rx.search(str(texts.get(int(f), n)))) for f, n in zip(df["fdc_id"], df["name"])]]
    for day_index in range(len(DAYS)):
        for slot in plan_slots(db.plan_items(plan_id).query("day_index == @day_index")):
            items = db.plan_items(plan_id)
            meal = items[(items["day_index"] == day_index) & (items["slot"] == slot)]
            bad = disliked(meal)
            action = rng.random()
            if len(bad) and action < SWAP_RATE:
                r = bad.iloc[0]
                others = {base_of[f] for f in items.loc[items["day_index"] == day_index, "fdc_id"] if f != r["fdc_id"] and f in base_of}
                options = swap_options(int(r["fdc_id"]), slot, float(r["calories"]), allowed, emb, others)
                good = [o for o in options if not rx.search(str(texts.get(int(o.fdc_id), o.name)))]
                if good:
                    db.replace_item(int(r["id"]), good[0])
                    swaps += 1
            elif len(bad) and action < SWAP_RATE + NEW_MEAL_RATE:
                penalties = (food_penalties(db.user_history(user), foods, db.user_swaps(user), db.user_choices(user))
                             if learning else None)
                planner = MealPlanner(allowed, emb, plan["taste_vector"], penalties, int(rng.integers(1_000_000)), nutrition=plan["targets"])
                new = planner.replan_meal(items, day_index, slot, daily_targets(plan["targets"]), n_meals).items
                keep = not any(rx.search(str(texts.get(int(i.fdc_id), i.name))) for i in new)
                if keep:
                    db.replace_meal(plan_id, day_index, DAYS[day_index], slot, new)
                    kept_new += 1
                db.record_choice(plan_id, day_index, slot, "new" if keep else "current",
                                 list(zip(meal["fdc_id"], meal["name"])), [(i.fdc_id, i.name) for i in new])
                new_meals += 1
            meal = db.plan_items(plan_id).query("day_index == @day_index and slot == @slot")
            eat = rng.random() >= SKIP_DISLIKED if len(disliked(meal)) else rng.random() < adherence
            db.set_meal_eaten(plan_id, day_index, slot, bool(eat))
    return swaps, new_meals, kept_new


def build(profile, foods, emb, seed, taste=None, penalties=None):
    """The same steps as build_plan in app.py."""
    targets = calculate_targets(profile)
    prefs = parse_preferences(profile.preferences_text)
    allowed, _ = apply_to_foods(foods, prefs)
    taste = taste if taste is not None else taste_from_preferences(prefs)
    t0 = time.time()
    plan = MealPlanner(allowed, emb, taste, penalties, seed, nutrition=targets).plan_week(targets)
    return plan, targets, prefs, taste, time.time() - t0


def run_tester(i, spec, foods, emb, texts, learning, weeks=WEEKS):
    """One tester, start to finish. Returns one row per week."""
    age, sex, kg, cm, activity, goal, text, meals, modes, dishes, dislike, allergies, diet, wording = spec
    rng = np.random.default_rng(1000 + i)                    # same habits with and without learning
    adherence = rng.uniform(0.85, 0.98)
    rx = re.compile(DISLIKES[dislike], re.I)
    name = f"Tester {i:02d} {'learning' if learning else 'no learning'}"
    pin = f"{rng.integers(1000, 9999)}"                      # throwaway value, only in the temporary database
    profile = UserProfile(name, age, sex, kg, cm, activity, goal, text, meals, modes, dishes)
    uid = db.create_user(profile, pin)
    signed_in = db.verify_pin(name.lower(), pin) == ("ok", uid) and db.check_token(uid, db.session_token(uid))
    wrong_refused = db.verify_pin(name, "00000")[0] == "wrong"
    start = date.today() - timedelta(days=7 * weeks)
    db.log_weight(uid, kg, start)
    db.delete_weight(uid, date.today())
    weight, rows, taste, penalties = float(kg), [], None, None
    for week in range(1, weeks + 1):
        user = db.get_user(uid)
        profile = UserProfile(**{k: user[k] for k in UserProfile.__dataclass_fields__})
        plan, targets, prefs, taste, seconds = build(profile, foods, emb, 1000 * i + week, taste if learning else None,
                                                     penalties if learning else None)
        plan_id = db.save_plan(uid, plan, targets, prefs, 1000 * i + week, taste)
        planned = plan.to_dataframe()
        days = planned.groupby("day_index")[list(MACROS)].sum()
        errors = {m: float(((days[m] - getattr(targets, m)).abs() / getattr(targets, m) * 100).mean()) for m in MACROS}
        within = float(((days["calories"] - targets.calories).abs() <= 0.1 * targets.calories).mean())
        served_disliked = sum(bool(rx.search(str(texts.get(int(f), n)))) for f, n in zip(planned["fdc_id"], planned["name"]))
        saved = db.latest_plan(uid)
        allowed = filter_modes(apply_to_foods(foods, prefs)[0], modes)
        swaps, new_meals, kept = run_week_actions(plan_id, saved, uid, allowed, foods, emb, texts, rx, adherence, rng, learning)
        items = db.plan_items(plan_id)
        allergen, diet_broken = rule_broken(items, foods, allergies, diet)
        rows.append({"tester": i, "arm": "with learning" if learning else "no learning", "week": week, "wording": wording,
                     "dislike": dislike, "goal": goal, "meals": meals, "modes": modes or "none",
                     "allergy_found": allergies <= prefs.block_groups, "meals_eaten": acceptance_rate(items),
                     "disliked_served": served_disliked, "items": len(items), "allergen_items": allergen, "diet_items": diet_broken,
                     **{f"{m}_err": e for m, e in errors.items()}, "days_within_10": within, "plan_seconds": seconds,
                     "swaps": swaps, "new_meals": new_meals, "kept_new": kept, "target_kcal": targets.calories,
                     "signed_in": signed_in, "wrong_pin_refused": wrong_refused})
        # Finish the week. The weight moves as the plan predicts, plus noise, only so that the targets get updated.
        weight = round(weight + (targets.calories - targets.tdee) * 7 / 7700 + rng.normal(0, 0.2), 1)
        db.log_weight(uid, weight, start + timedelta(days=7 * week))
        db.mark_plan_finished(plan_id)
        if learning:
            taste, penalties = next_week_inputs(saved["taste_vector"], emb, items, db.user_history(uid), db.user_swaps(uid), db.user_choices(uid))
    return rows


def table(df):
    head = "| " + " | ".join(str(c) for c in df.columns) + " |\n|" + "---|" * len(df.columns)
    return "\n".join([head] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def sentence_report():
    """Parser results per set. "Asked" = allergy items not blocked, but the app shows a warning about that sentence."""
    from src.parser import parse_preferences
    rows, details = [], []
    for name, (file, before) in SENTENCE_SETS.items():
        df = pd.read_csv(OUT / file, dtype=str).fillna("")
        out, m = score(df)
        out["style"], out["set"] = df["style"], name
        out["warned"] = [bool(parse_preferences(t).unclear) for t in out["sentence"]]
        missed = out[out["allergy_blocked"] < out["allergy_expected"]]
        missing = (missed["allergy_expected"] - missed["allergy_blocked"])
        rows.append({"Sentences": name, "Number": len(out), "Allergy items blocked, before the fix": before,
                     "Allergy items blocked, after the fix": f"{out['allergy_blocked'].sum()} / {out['allergy_expected'].sum()}",
                     "Recall": f"{m['allergy_recall'] * 100:.1f}%", "Not blocked, but the app asks": f"{int(missing[missed['warned']].sum())} / {int(missing.sum())}",
                     "Dislikes found": f"{m['less_often_recall'] * 100:.0f}%", "Likes found": f"{m['more_often_recall'] * 100:.0f}%"})
        details.append(out)
    pd.concat(details).to_csv(OUT / "new_sentences_scored.csv", index=False)
    return table(pd.DataFrame(rows))


def report(df, errors, sentences):
    pc = lambda x: f"{x * 100:.1f}%"
    clean = df[df["wording"] == "clean"]
    with_dislike = clean[clean["dislike"] != "no hidden dislike"]
    eaten = with_dislike.groupby(["arm", "week"])["meals_eaten"].mean().unstack("week").map(pc).reset_index()
    served = with_dislike.groupby(["arm", "week"])["disliked_served"].mean().unstack("week").round(1).reset_index()
    eaten.columns = served.columns = ["Mode"] + [f"Week {w}" for w in eaten.columns[1:]]
    safety = df.groupby("wording").agg(testers=("tester", "nunique"), found=("allergy_found", "mean"), items=("items", "sum"),
                                       allergen=("allergen_items", "sum"), diet=("diet_items", "sum")).reset_index()
    safety["found"] = safety["found"].map(pc)
    safety.columns = ["Wording", "Testers", "Allergy understood", "Food items checked", "Items with a stated allergen", "Items breaking a diet rule"]
    messy = df[df["wording"] == "messy"].groupby("tester").agg(found=("allergy_found", "first"), allergen=("allergen_items", "sum")).reset_index()
    a = df[df["arm"] == "with learning"]
    per_account = df.groupby(["tester", "arm"])
    text = f"""# Simulated testers

**These are scripted users, not real people.** Their profiles, typed text, hidden dislikes and eating habits were
invented for this test (src/testers.py). The app's own database, planner and feedback functions ran for real on a
throwaway database. No satisfaction ratings exist here, because only real people can give them.

{df['tester'].nunique()} testers x {WEEKS} weeks x 2 modes (learning on and off) = {len(df)} planned weeks. Assumptions: a meal with a food the
tester dislikes is skipped {SKIP_DISLIKED:.0%} of the time, other meals are eaten 85-98% of the time, and on a disliked food they swap it
{SWAP_RATE:.0%} of the time and ask for a new meal {NEW_MEAL_RATE:.0%} of the time.

## 1. Learning from behaviour ({with_dislike['tester'].nunique()} testers with clean wording and a hidden dislike)
Share of meals eaten:

{table(eaten)}

Disliked foods served per week (before any swap):

{table(served)}

## 2. Safety
{table(safety)}

Messy-wording testers ({len(messy)}): allergy understood by {int(messy['found'].sum())}; stated allergen served in {int(messy['allergen'].sum())} food items over both modes.

## 3. Accuracy and speed (with learning)
- Calories {a['calories_err'].mean():.1f}%, protein {a['protein_g_err'].mean():.1f}%, carbs {a['carbs_g_err'].mean():.1f}%, fat {a['fat_g_err'].mean():.1f}% mean error; {pc(a['days_within_10'].mean())} of days within 10% of the calorie target.
- Plan generation: mean {df['plan_seconds'].mean():.2f} s, max {df['plan_seconds'].max():.2f} s.

## 4. Use of the features (both modes)
- Swaps: {int(df['swaps'].sum())}. New meal requests: {int(df['new_meals'].sum())}, new version kept: {int(df['kept_new'].sum())}.
- Sign-in with the right PIN worked for {pc(per_account['signed_in'].first().mean())} of accounts; a wrong PIN was refused for {pc(per_account['wrong_pin_refused'].first().mean())}.
- Testers that crashed: {len(errors)}{''.join(chr(10) + '  - ' + e for e in errors)}

## 5. Parser on new wording (written by an AI assistant, not by volunteers)
{sentences}

"Before the fix" is commit e75a8e4. Sets 1 and 2 went into building the fix, so their scores after it are optimistic. Set 3
is the fair number: written after the rules were frozen and scored once. The asking rule was added after seeing set 3, so
that column is not an independent result either. It is one general rule, not new phrases, and it doesn't change the recall.
"""
    (OUT / "simulated_testers.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    from src.embeddings import FoodEmbeddings
    foods = pd.read_csv(FOODS_CLEAN)
    emb = FoodEmbeddings(foods)
    texts = food_texts(foods)
    sentences = sentence_report()
    os.environ.pop("DATABASE_URL", None)
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["NUTRI_DB_PATH"] = str(Path(tmp) / "testers.db")   # never data/app.db
        db.init_db()
        rows, errors = [], []
        for learning in (True, False):
            for i, spec in enumerate(TESTERS, 1):
                try:
                    rows += run_tester(i, spec, foods, emb, texts, learning)
                except Exception as e:
                    errors.append(f"Tester {i} ({'learning' if learning else 'no learning'}): {type(e).__name__}: {e}")
                    print(errors[-1])
                print(f"tester {i} learning={learning} done", flush=True)
        os.environ.pop("NUTRI_DB_PATH")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "simulated_testers.csv", index=False)
    report(df, errors, sentences)


if __name__ == "__main__":
    main()
