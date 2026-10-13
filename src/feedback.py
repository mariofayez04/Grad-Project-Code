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

An unticked meal counts as skipped. Every food in a skipped meal is penalised (we don't know which item was
the problem); the taste vector spreads that signal to similar foods.
"""
import numpy as np
import pandas as pd

ALPHA = 0.3          # pull towards eaten foods
BETA = 0.15          # push away from skipped foods
SKIP_PENALTY = 0.8   # per skipped meal containing the food
EAT_BONUS = 0.2      # per eaten meal containing the food (negative penalty = boost)
DECAY = 0.7          # weight of a week that is one week older
FAMILY_PENALTY = 2.0  # max extra penalty for a food family skipped far more than average
NEUTRAL_RE = r"^(?:Oil|Butter|Ghee)\b"  # cooking fat added by the planner, not a taste signal


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


def food_penalties(history, foods=None):
    """history: plan items of all finished weeks with a `week_number` column. Returns {fdc_id: penalty}.

    If `foods` (the food table) is given, a food-family penalty is added too: a base food (e.g. "beef") that is
    skipped more often than the user's average skip rate is penalised for *every* food of that family, so a
    different beef cut doesn't take its place. Families skipped no more than average (e.g. rice that just sat
    next to a disliked fish) get nothing.
    """
    if history.empty:
        return {}
    latest = history["week_number"].max()
    penalties, fam_skip, fam_eat = {}, {}, {}
    base_of = dict(zip(foods["fdc_id"], foods["base_name"])) if foods is not None else {}
    for week, items in history.groupby("week_number"):
        weight = DECAY ** (latest - week)
        eaten, skipped = split_foods(items)
        for fid in skipped:
            penalties[fid] = penalties.get(fid, 0.0) + weight * SKIP_PENALTY
            if fid in base_of:
                fam_skip[base_of[fid]] = fam_skip.get(base_of[fid], 0.0) + weight
        for fid in eaten:
            penalties[fid] = penalties.get(fid, 0.0) - weight * EAT_BONUS
            if fid in base_of:
                fam_eat[base_of[fid]] = fam_eat.get(base_of[fid], 0.0) + weight

    if base_of:
        total_skip, total = sum(fam_skip.values()), sum(fam_skip.values()) + sum(fam_eat.values())
        avg_rate = total_skip / total if total else 0.0
        for base in fam_skip:
            n = fam_skip[base] + fam_eat.get(base, 0.0)
            excess = fam_skip[base] / n - avg_rate          # skipped more than usual?
            confidence = min(1.0, n / 3)                     # need a few observations
            if excess > 0:
                fam_pen = FAMILY_PENALTY * excess * confidence
                for fid, b in base_of.items():
                    if b == base:
                        penalties[fid] = penalties.get(fid, 0.0) + fam_pen
    return {int(k): round(v, 3) for k, v in penalties.items() if v != 0}


def week_summary(items, top=5):
    """What the system learned from one week, for display."""
    eaten, skipped = split_foods(items)
    names = dict(zip(items["fdc_id"], items["name"]))
    count = lambda ids: pd.Series(ids, dtype="int64").value_counts().head(top)
    return {
        "acceptance": acceptance_rate(items),
        "most_skipped": [(names[i], int(n)) for i, n in count(skipped).items()],
        "most_eaten": [(names[i], int(n)) for i, n in count(eaten).items()],
    }


def next_week_inputs(old_taste, embeddings, last_week_items, history):
    """Everything the planner needs for next week: (taste_vector, food_penalties)."""
    eaten, skipped = split_foods(last_week_items)
    return update_taste(old_taste, embeddings, eaten, skipped), food_penalties(history, embeddings.foods)
