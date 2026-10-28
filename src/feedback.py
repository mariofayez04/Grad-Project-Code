"""Phase 7: weekly learning loop.

At the end of a week, the eaten / not-eaten checkboxes drive two updates for next week's plan:

 1. Taste vector (embedding space, no training needed):
        taste_new = normalize(taste_old + ALPHA * mean(eaten food vectors) - BETA * mean(skipped food vectors))
    The planner ranks candidates partly by cosine(food, taste), so foods *similar* to what you ate rise
    and foods similar to what you skipped fall - even ones never shown before.

 2. Food penalties (exact foods):
        penalty(food) = sum over past weeks of DECAY^age * (SKIP_PENALTY * times skipped - EAT_BONUS * times eaten)
    Recent weeks count more (DECAY), so the profile follows changing tastes.

 3. Food-family penalty: a family (e.g. all "beef" foods) skipped more often than the user's average gets
        FAMILY_PENALTY * (family skip rate - average skip rate) * confidence
    An Egyptian dish also belongs to the families of its main ingredients (>= MAIN_INGREDIENT_SHARE of the
    serving, not oils / broth / spices): bamya is "bamya with rice" + beef + okra + rice. So skipping beef also
    pushes down the beef dishes, and skipping a dish counts against its ingredients' families.

An unticked meal counts as skipped. Every food in a skipped meal is penalised (we don't know which item was
the problem); the taste vector spreads that signal to similar foods.

A food the user swapped away ("Swap a food" in the app) is a clearer signal: it counts as skipped once, in the
taste vector, the food penalties and the family penalties, even if the rest of the meal was eaten.

"New meal" / "New day" in the app shows the current and a new version side by side and the user keeps one.
That choice is a softer signal (both versions may be fine): foods of the version not kept count as
CHOICE_WEIGHT of a skip, foods of the version kept as CHOICE_WEIGHT of an eaten meal.
"""
import re

import numpy as np
import pandas as pd

ALPHA = 0.3          # pull towards eaten foods
BETA = 0.15          # push away from skipped foods
SKIP_PENALTY = 0.8   # per skipped meal containing the food
EAT_BONUS = 0.2      # per eaten meal containing the food (negative penalty = boost)
DECAY = 0.7          # weight of a week that is one week older
FAMILY_PENALTY = 2.0  # max extra penalty for a food family skipped far more than average
NEUTRAL_RE = r"^(?:Oil|Butter|Ghee)\b"  # cooking fat added by the planner, not a taste signal
CHOICE_WEIGHT = 0.5  # a before / after choice counts as half a meal eaten (kept) or skipped (not kept)
MAIN_INGREDIENT_SHARE = 0.10  # a dish ingredient this big a share of the serving gives the dish its family
NEUTRAL_CATEGORIES = {"Fats and Oils", "Soups, Sauces, and Gravies", "Spices and Herbs", "Beverages"}


def food_families(foods, recipes=None):
    """{fdc_id: set of family names}: a food's own base name (e.g. "beef"); an Egyptian dish also the base names
    of its main ingredients (e.g. bamya -> {"bamya with rice", "beef", "okra", "rice"})."""
    fams = {int(f): {b} for f, b in zip(foods["fdc_id"], foods["base_name"])}
    if recipes is None:
        from src.dishes import load_recipes
        recipes = load_recipes()
    if recipes.empty:
        return fams
    main = recipes[(recipes["grams"] / recipes["serving_g"] >= MAIN_INGREDIENT_SHARE)
                   & ~recipes["food_category"].isin(NEUTRAL_CATEGORIES)]
    for dish, desc in zip(main["dish_fdc_id"], main["description"]):
        if int(dish) in fams:
            fams[int(dish)].add(desc.split(",")[0].strip().lower())  # same rule as base_name in clean_data
    return fams


def meal_status(items):
    """One row per meal: eaten (True/False). Unanswered (NULL) counts as skipped."""
    meals = items.groupby(["day_index", "slot"])["eaten"].max().fillna(0).astype(bool)
    return meals.rename("eaten").reset_index()


def acceptance_rate(items):
    """Share of the week's meals that were eaten."""
    meals = meal_status(items)
    return float(meals["eaten"].mean()) if len(meals) else 0.0


def split_foods(items):
    """fdc_ids of foods in eaten meals and in skipped meals (one entry per meal).
    Cooking fat is left out: it is in almost every lunch and dinner, so it says nothing about taste."""
    status = items.merge(meal_status(items), on=["day_index", "slot"], suffixes=("_raw", ""))
    status = status[~status["name"].str.contains(NEUTRAL_RE, regex=True)]
    return status.loc[status["eaten"], "fdc_id"].tolist(), status.loc[~status["eaten"], "fdc_id"].tolist()


def update_taste(old, embeddings, eaten_ids, skipped_ids, alpha=ALPHA, beta=BETA):
    """taste_new = normalize(taste_old + alpha * mean(eaten) - beta * mean(skipped))."""
    v = np.zeros(embeddings.vectors.shape[1], dtype=np.float32) if old is None else np.asarray(old, dtype=np.float32).copy()
    eaten_ids = [i for i in eaten_ids if i in embeddings.index]
    skipped_ids = [i for i in skipped_ids if i in embeddings.index]
    if eaten_ids:
        v += alpha * embeddings.vectors[[embeddings.index[i] for i in eaten_ids]].mean(axis=0)
    if skipped_ids:
        v -= beta * embeddings.vectors[[embeddings.index[i] for i in skipped_ids]].mean(axis=0)
    norm = np.linalg.norm(v)
    return None if norm == 0 else v / norm


def choice_foods(choices):
    """(kept fdc_ids, not-kept fdc_ids) from rows of db.user_choices, cooking fat left out."""
    pick = lambda col: [f["fdc_id"] for foods in choices[col] for f in foods if not re.match(NEUTRAL_RE, f["name"])]
    return pick("chosen"), pick("rejected")


def food_penalties(history, foods=None, swaps=None, choices=None, recipes=None):
    """history: plan items of all finished weeks with a `week_number` column. Returns {fdc_id: penalty}.
    swaps: rows of db.user_swaps (old_fdc_id, week_number); each swapped-away food counts as one skip.
    choices: rows of db.user_choices; foods not kept count as CHOICE_WEIGHT of a skip, kept ones of an eaten meal.

    If `foods` (the food table) is given, a food-family penalty is added too: a family (e.g. "beef") that is
    skipped more often than the user's average skip rate is penalised for *every* food of that family, so a
    different beef cut doesn't take its place - nor a dish made with beef (food_families). Families skipped no
    more than average (e.g. rice that just sat next to a disliked fish) get nothing.
    recipes: the dish recipes (default: dish_ingredients.csv).
    """
    if history.empty:
        return {}
    latest = history["week_number"].max()
    penalties, fam_skip, fam_eat = {}, {}, {}
    fams = food_families(foods, recipes) if foods is not None else {}

    def count(fid, weight, eaten):
        penalties[fid] = penalties.get(fid, 0.0) + (-weight * EAT_BONUS if eaten else weight * SKIP_PENALTY)
        tally = fam_eat if eaten else fam_skip
        for fam in fams.get(int(fid), ()):
            tally[fam] = tally.get(fam, 0.0) + weight

    for week, items in history.groupby("week_number"):
        weight = DECAY ** (latest - week)
        eaten, skipped = split_foods(items)
        for fid in skipped:
            count(fid, weight, eaten=False)
        for fid in eaten:
            count(fid, weight, eaten=True)
    if swaps is not None:
        for fid, week in zip(swaps["old_fdc_id"], swaps["week_number"]):
            count(fid, DECAY ** (latest - week), eaten=False)
    if choices is not None:
        for week, rows in choices.groupby("week_number"):
            weight = CHOICE_WEIGHT * DECAY ** (latest - week)
            kept, not_kept = choice_foods(rows)
            for fid in not_kept:
                count(fid, weight, eaten=False)
            for fid in kept:
                count(fid, weight, eaten=True)

    if fams:
        members = {}
        for fid, fs in fams.items():
            for fam in fs:
                members.setdefault(fam, []).append(fid)
        total_skip, total = sum(fam_skip.values()), sum(fam_skip.values()) + sum(fam_eat.values())
        avg_rate = total_skip / total if total else 0.0
        for fam in fam_skip:
            n = fam_skip[fam] + fam_eat.get(fam, 0.0)
            excess = fam_skip[fam] / n - avg_rate           # skipped more than usual?
            confidence = min(1.0, n / 3)                     # need a few observations
            if excess > 0:
                for fid in members.get(fam, []):
                    penalties[fid] = penalties.get(fid, 0.0) + FAMILY_PENALTY * excess * confidence
    return {int(k): round(v, 3) for k, v in penalties.items() if v != 0}


def week_summary(items, top=5, swaps=None, choices=None):
    """What the system learned from one week, for display.
    swaps / choices: that week's rows of db.user_swaps / db.user_choices."""
    eaten, skipped = split_foods(items)
    names = dict(zip(items["fdc_id"], items["name"]))
    count = lambda ids: pd.Series(ids, dtype="int64").value_counts().head(top)
    swapped = [] if swaps is None or swaps.empty else swaps["old_name"].value_counts().head(top).items()
    return {
        "acceptance": acceptance_rate(items),
        "most_skipped": [(names[i], int(n)) for i, n in count(skipped).items()],
        "most_eaten": [(names[i], int(n)) for i, n in count(eaten).items()],
        "swapped_away": [(name, int(n)) for name, n in swapped],
        "not_chosen": _choice_names(choices, "rejected", top),
        "chosen": _choice_names(choices, "chosen", top),
    }


def _choice_names(choices, col, top):
    if choices is None or choices.empty:
        return []
    names = [f["name"] for foods in choices[col] for f in foods if not re.match(NEUTRAL_RE, f["name"])]
    return [(n, int(k)) for n, k in pd.Series(names, dtype="object").value_counts().head(top).items()]


def next_week_inputs(old_taste, embeddings, last_week_items, history, swaps=None, choices=None):
    """Everything the planner needs for next week: (taste_vector, food_penalties).
    swaps / choices: all of the user's finished-week swaps and before / after choices (db.user_swaps,
    db.user_choices); last week's ones also move the taste vector."""
    eaten, skipped = split_foods(last_week_items)
    last_week = history["week_number"].max() if not history.empty else None
    if swaps is not None and not swaps.empty and last_week is not None:
        skipped += swaps.loc[swaps["week_number"] == last_week, "old_fdc_id"].astype(int).tolist()
    if choices is not None and not choices.empty and last_week is not None:
        kept, not_kept = choice_foods(choices[choices["week_number"] == last_week])
        eaten, skipped = eaten + kept, skipped + not_kept
    return (update_taste(old_taste, embeddings, eaten, skipped),
            food_penalties(history, embeddings.foods, swaps, choices))
