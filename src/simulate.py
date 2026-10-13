"""Phase 7 evaluation: does the feedback loop learn a user's hidden dislikes?

A simulated user has dislikes they never typed (e.g. fish). Each week they skip every meal containing a
disliked food and eat the rest. We compare:
   with feedback:     taste vector + food penalties updated every week (Phase 7)
   without feedback:  a fresh plan every week that ignores the checkboxes (baseline)

Run:  python -m src.simulate
"""
import re

import numpy as np
import pandas as pd

from src.calculator import UserProfile, calculate_targets
from src.config import FOODS_CLEAN
from src.feedback import acceptance_rate, food_penalties, split_foods, update_taste
from src.parser import apply_to_foods, parse_preferences
from src.planner import MealPlanner, taste_from_preferences

# Hidden dislikes for the experiment (regex on the food name).
SCENARIOS = {
    "dislikes fish & seafood": r"\b(fish|salmon|tuna|tilapia|cod|sardines?|mackerel|trout|crustaceans|mollusks|shrimp)\b",
    "dislikes beef & lamb": r"\b(beef|lamb|veal)\b",
    "dislikes eggs & cottage cheese": r"\b(egg|eggs|cottage)\b",
}


def simulate_ticks(items, dislike_re):
    """The simulated user skips any meal that contains a disliked food."""
    items = items.copy()
    bad = items["name"].str.contains(dislike_re, case=False, regex=True)
    skipped_meals = set(items.loc[bad, ["day_index", "slot"]].itertuples(index=False, name=None))
    items["eaten"] = [0 if (d, s) in skipped_meals else 1 for d, s in zip(items["day_index"], items["slot"])]
    return items


def run(profile, dislike_re, embeddings, foods, weeks=4, feedback=True, seed=0):
    """Returns one dict per week: acceptance rate and number of disliked items served."""
    targets = calculate_targets(profile)
    prefs = parse_preferences(profile.preferences_text)
    allowed, _ = apply_to_foods(foods, prefs)
    taste = taste_from_preferences(prefs)
    history, results = [], []
    rx = re.compile(dislike_re, re.I)
    for week in range(1, weeks + 1):
        penalties = food_penalties(pd.concat(history), foods) if (feedback and history) else None
        plan = MealPlanner(allowed, embeddings, taste if feedback else taste_from_preferences(prefs),
                           penalties, seed=seed + week).plan_week(targets)
        items = simulate_ticks(plan.to_dataframe(), rx)
        results.append({"week": week, "acceptance": round(acceptance_rate(items), 3),
                        "disliked_items": int(items["name"].str.contains(rx).sum())})
        if feedback:
            eaten, skipped = split_foods(items)
            taste = update_taste(taste, embeddings, eaten, skipped)
            history.append(items.assign(week_number=week))
    return results


def experiment(weeks=4, seeds=(0, 1, 2)):
    from src.embeddings import FoodEmbeddings
    foods = pd.read_csv(FOODS_CLEAN)
    emb = FoodEmbeddings(foods)
    user = UserProfile("Sim", 30, "male", 75, 175, 2, "maintain", "")
    rows = []
    for name, rx in SCENARIOS.items():
        for fb in (False, True):
            for seed in seeds:
                for r in run(user, rx, emb, foods, weeks, fb, seed * 100):
                    rows.append({"scenario": name, "feedback": "with feedback" if fb else "no feedback", "seed": seed, **r})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    df = experiment()
    table = df.groupby(["scenario", "feedback", "week"])[["acceptance", "disliked_items"]].mean().unstack("week")
    pd.set_option("display.width", 200)
    print("Mean over 3 runs. acceptance = share of meals eaten; disliked_items = disliked foods served that week\n")
    print(table.round(2).to_string())
