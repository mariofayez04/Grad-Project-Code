"""How realistic are the planned portions? Compares plans with everyday serving sizes.

A plan can hit its numbers with meals nobody would eat, so this measures something the accuracy tests can't. The limits
below are a judgment about a generous single serving of cooked or as-served food, not official values.

Run:  python -m src.portions                (results)
      python -m src.portions --sheet        (real meals for a qualified person to rate)
"""
import sys
import warnings

import numpy as np
import pandas as pd

from src.config import DATA_DIR, FOODS_CLEAN
from src.evaluate import random_profiles
from src.planner import (EVERYDAY, HAS_STARCH_RE, MACROS, MealPlanner, RAW_NOT_SALAD, combo_allowed, daily_targets, is_everyday,
                         taste_from_preferences)

OUT = DATA_DIR / "evaluation"
# Generous single servings, in grams, by the food's role in a meal.
SERVING_LIMITS = {"protein": 220, "legume": 250, "carb": 280, "vegetable": 220, "fruit": 220, "dairy": 300, "fat": 30, "snack": 120,
                  "mixed": 350, "dish": 500}
LIMIT_FOR_MEAL = 800     # total grams of a main meal (lunch or dinner) that one person would finish


def measure(foods, emb, users=30, seed=0):
    """One week for each random user. Returns (food rows, summary)."""
    from src.calculator import calculate_targets
    from src.parser import apply_to_foods, parse_preferences
    roles = dict(zip(foods["fdc_id"], foods["meal_role"]))
    rows, errors, within, kinds = [], [], [], []
    for i, profile in enumerate(random_profiles(users, seed=seed)):
        targets = calculate_targets(profile)
        prefs = parse_preferences(profile.preferences_text)
        allowed, _ = apply_to_foods(foods, prefs)
        plan = MealPlanner(allowed, emb, taste_from_preferences(prefs), None, i, nutrition=targets).plan_week(targets)
        items = plan.to_dataframe()
        items["role"] = items["fdc_id"].map(roles)
        items["user"] = i
        items["target_kcal"] = targets.calories
        rows.append(items)
        days = items.groupby("day_index")[MACROS].sum()
        target = daily_targets(targets)
        errors.append({m: float(((days[m] - target[m]).abs() / target[m]).mean() * 100) for m in MACROS})
        within.append(float(((days["calories"] - target["calories"]).abs() <= 0.1 * target["calories"]).mean()))
        kinds.append(items["fdc_id"].nunique())
    items = pd.concat(rows, ignore_index=True)
    items["over"] = items["grams"] > items["role"].map(SERVING_LIMITS)
    meals = items.groupby(["user", "day_index", "slot"])["grams"].sum().reset_index()
    main = meals[meals["slot"].isin(["lunch", "dinner"])]
    err = pd.DataFrame(errors).mean()
    summary = {"Items over a generous serving": f"{items['over'].mean() * 100:.1f}%",
               "Largest single food (not a dish), g": f"{items.loc[items['role'] != 'dish', 'grams'].max():.0f}",
               "95th percentile of a food's portion, g": f"{items.loc[items['role'] != 'dish', 'grams'].quantile(.95):.0f}",
               "Main meals over 800 g": f"{(main['grams'] > LIMIT_FOR_MEAL).mean() * 100:.1f}%",
               "Heaviest main meal, g": f"{main['grams'].max():.0f}",
               "Calorie error": f"{err['calories']:.1f}%", "Protein error": f"{err['protein_g']:.1f}%",
               "Carbohydrate error": f"{err['carbs_g']:.1f}%", "Fat error": f"{err['fat_g']:.1f}%",
               "Days within 10% of calories": f"{np.mean(within) * 100:.1f}%",
               "Different foods per week": f"{np.mean(kinds):.0f}"}
    return items, summary


NOT_SERVABLE = r"hulled|as purchased|\\bpeel\\b|lecithin|eggnog|tofu yogurt|muscadine|stalks|guava sauce|puree, |dried.*stewed"


def meal_quality(items, foods):
    """Measures the rules themselves: everyday foods, nonsense items and foods that don't go together. It can't say
    whether people would like the meals, which only people can."""
    desc = dict(zip(foods["fdc_id"], foods["description"]))
    roles = dict(zip(foods["fdc_id"], foods["meal_role"]))
    df = items.assign(d=items["fdc_id"].map(desc), r=items["fdc_id"].map(roles))
    listed = df[df["r"].isin(EVERYDAY)]
    everyday = pd.Series([is_everyday(d, r) for d, r in zip(listed["d"], listed["r"])], index=listed.index)
    raw_bad = df[df["r"] == "vegetable"]["d"].str.contains(RAW_NOT_SALAD, case=False, regex=True)
    meals = df.groupby(["user", "day_index", "slot"])
    clash = sum(not combo_allowed(list(zip(g["d"], g["r"])), slot) for (_, _, slot), g in meals)
    starch = sum(bool(g["r"].eq("dish").any() and g.loc[g["r"].eq("dish"), "d"].str.match(HAS_STARCH_RE).any() and g["r"].eq("carb").any())
                 for _, g in meals)
    by_role = {r: float(pd.Series(everyday[listed["r"] == r]).mean() * 100) for r in EVERYDAY}
    out = {"Everyday foods": f"{everyday.mean() * 100:.1f}%"}
    out.update({f"Everyday {r}": f"{v:.1f}%" for r, v in by_role.items()})
    out.update({"Items nobody should be served": str(int(df["d"].str.contains(NOT_SERVABLE, case=False, regex=True).sum() + raw_bad.sum())),
                "Meals with two bean or soy foods, or cereal without milk": f"{clash} of {meals.ngroups}",
                "Dishes with their own starch plus another carbohydrate": str(starch)})
    return out


def meal_rating_sheet(foods, emb, path=OUT / "meal_rating_sheet.xlsx", meals=20):
    """Real planned meals for someone qualified to rate. Nothing is filled in: the scores can only come from that person."""
    items, _ = measure(foods, emb, users=6, seed=5)
    keys = items[["user", "day_index", "slot"]].drop_duplicates().sample(meals, random_state=7).sort_values(["user", "day_index"])
    rows = []
    for n, k in enumerate(keys.itertuples(), 1):
        meal = items[(items["user"] == k.user) & (items["day_index"] == k.day_index) & (items["slot"] == k.slot)]
        rows.append({"Meal": n, "Type": k.slot.replace("_", " "), "Foods": "; ".join(f"{i.name}: {i.grams:.0f} g" for i in meal.itertuples()),
                     "Calories": round(meal["calories"].sum()), "Would a person eat this? (1-5)": "", "Portions realistic? (1-5)": "",
                     "Foods go together? (1-5)": "", "Comment": ""})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Meals", index=False)
        pd.DataFrame({"Instructions": [
            "These meals were planned automatically for a student project. Please rate each one from 1 (no) to 5 (yes).",
            "Please write your job or qualification here so it can be reported accurately: ______________",
            "This is about how realistic the meals are. It is not a medical review of any person's diet."]}).to_excel(xw, sheet_name="Instructions", index=False)
        ws = xw.sheets["Meals"]
        for col, width in zip("ABCDEFGH", (7, 14, 90, 9, 16, 16, 16, 30)):
            ws.column_dimensions[col].width = width
        xw.sheets["Instructions"].column_dimensions["A"].width = 110
    return path


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")
    from src.embeddings import FoodEmbeddings
    foods = pd.read_csv(FOODS_CLEAN)
    if "--sheet" in sys.argv:
        print("Saved", meal_rating_sheet(foods, FoodEmbeddings(foods)))
        sys.exit()
    items, summary = measure(foods, FoodEmbeddings(foods))
    for k, v in summary.items():
        print(f"{k}: {v}")
    by_role = items.groupby("role")["over"].mean().mul(100).round(1).to_dict()
    print("Over a serving, by role:", by_role)
    big = items[items["target_kcal"] >= 2800]
    print("Over a serving, targets of 2,800 kcal or more:", round(big["over"].mean() * 100, 1), "%; people:", big["user"].nunique())
    usual = items[items["target_kcal"] < 2800]
    print("Over a serving, targets under 2,800 kcal:", round(usual["over"].mean() * 100, 1), "%; people:", usual["user"].nunique())
    quality = meal_quality(items, foods)
    for k, v in quality.items():
        print(f"{k}: {v}")
    if len(sys.argv) > 1:
        pd.Series({**summary, **quality}).to_csv(OUT / sys.argv[1], header=False)
