"""Feature #7: extra nutrients - fiber, sugars, saturated fat, sodium, potassium, calcium, iron.

Daily reference values (adults; cite these in the report):
  fiber          >= 14 g per 1,000 kcal                     Dietary Guidelines for Americans 2020-2025
  saturated fat  <  10% of calories                          Dietary Guidelines for Americans 2020-2025, WHO
  sodium         <= 2,300 mg                                 NASEM 2019 (Chronic Disease Risk Reduction intake)
  potassium      >= 3,400 mg men / 2,600 mg women            NASEM 2019 (Adequate Intake)
  calcium        >= 1,000 mg; 1,200 mg women 51+, men 71+    NASEM 2011 (RDA)
  iron           >= 8 mg; 18 mg women 19-50                  NASEM 2001 (RDA)
  sugars         shown only: USDA lists total sugars, which include the natural sugar of fruit and milk, while
                 the limits (WHO, DGA) are for added / free sugars - so no pass / fail is given.

Daily caps the planner keeps under (target["upper"]): iron 45 mg and calcium 2,500 mg (2,000 mg 51+) - the
NASEM Tolerable Upper Intake Levels - fiber at twice its target (comfort; no official limit), and every "max"
target (sodium, saturated fat, sugars in the diabetes mode). Fortified cereals and breads can otherwise give one day
60+ mg of iron.

Food values are per 100 g in foods_clean.csv (Phase 1); a plan's amounts are computed from the grams.

Feature #10 - nutrient-aware planning and diet modes:
  Every plan: each food gets a nutrient score (nutrient_score) that the planner adds to its ranking - a bonus for
  foods rich in what is often short (fiber, calcium, iron, potassium) and a penalty for foods with more sodium or
  saturated fat than their "fair share" of the day's limit.
  Diet modes (a user can pick both):
    diabetes    carbs <= 40% of calories (calculator), total sugars <= 10% of calories, stronger fiber bonus,
                sugar penalty, sugary foods left out (dried / sweetened / flavoured foods, syrups...)
                (ADA Standards of Care: fewer refined carbs and added sugars, more fiber; no single ideal split)
    low_sodium  sodium <= 1,500 mg (AHA / DASH), stronger potassium and calcium bonus, sodium penalty,
                very salty foods left out (> 600 mg sodium per 100 g, cured / pickled / salted foods)
  These are general guidance, not a treatment: the app says so and asks people to follow their doctor.
"""
import re

import pandas as pd

from src.config import MICRO_IDS

MICROS = list(MICRO_IDS)  # fiber_g, sugar_g, sat_fat_g, sodium_mg, potassium_mg, calcium_mg, iron_mg
LABELS = {"fiber_g": ("Fiber", "g"), "sugar_g": ("Sugars", "g"), "sat_fat_g": ("Saturated fat", "g"),
          "sodium_mg": ("Sodium", "mg"), "potassium_mg": ("Potassium", "mg"), "calcium_mg": ("Calcium", "mg"),
          "iron_mg": ("Iron", "mg")}
CLOSE_ENOUGH = 0.9  # a "min" target counts as nearly met at 90%


HEALTH_MODES = {
    "diabetes": "Diabetes-friendly (less sugar, more fiber, carbs ≤ 40%)",
    "low_sodium": "Low-salt / blood pressure (sodium ≤ 1,500 mg, DASH-style)",
}
MODE_SODIUM_MG = 1500     # low_sodium daily limit (AHA ideal, DASH)
MODE_SUGAR_SHARE = 0.10   # diabetes: total sugars <= 10% of calories
SALTY_MG_PER_100G = 600   # low_sodium: foods above this are left out
IRON_UL_MG, CALCIUM_UL_MG, CALCIUM_UL_MG_51 = 45, 2500, 2000  # NASEM Tolerable Upper Intake Levels (adults)
FIBER_CAP_TIMES = 2       # fiber: at most twice the target (more upsets the stomach)
SALTY_RE = re.compile(r"\b(cured|pickled|salted|brine|corned|cheese spread|soy sauce|bouillon|jerky)\b", re.I)
SUGARY_RE = re.compile(r"\b(dried|dehydrated|sweetened|syrup|candied|honey|jam|jelly|frosted|glazed|chocolate|"
                       r"vanilla|strawberry|fruit|flavored|cinnamon|raisins?|granola|sugar|pan dulce|sweet yeast|"
                       r"sweet rolls?|danish|pastry|cakes?|cookies?|doughnuts?|muffins?|pies?)\b", re.I)

# Planner weights (per food, added to its score). "fair share" = the daily target spread evenly over the calories.
ENCOURAGE = {"fiber_g": 1.0, "calcium_mg": 1.0, "iron_mg": 0.6, "potassium_mg": 0.5}
LIMIT = {"sodium_mg": 0.6, "sat_fat_g": 0.5}
MODE_ENCOURAGE = {"diabetes": {"fiber_g": 1.0}, "low_sodium": {"potassium_mg": 0.8, "calcium_mg": 0.5}}
MODE_LIMIT = {"diabetes": {"sugar_g": 1.5}, "low_sodium": {"sodium_mg": 1.6}}


def parse_modes(modes):
    """'diabetes|low_sodium' / list / None -> set of known modes."""
    if not modes:
        return set()
    items = modes.split("|") if isinstance(modes, str) else modes
    return {m for m in items if m in HEALTH_MODES}


def micro_targets(sex, age, calories, modes=()):
    """{nutrient: {"value": number or None, "kind": "min" / "max" / "info"}} for one person."""
    female = str(sex).lower().startswith("f")
    modes = parse_modes(modes)
    targets = {
        "fiber_g": {"value": round(14 * calories / 1000), "kind": "min"},
        "sugar_g": {"value": None, "kind": "info"},
        "sat_fat_g": {"value": round(0.10 * calories / 9), "kind": "max"},
        "sodium_mg": {"value": 2300, "kind": "max"},
        "potassium_mg": {"value": 2600 if female else 3400, "kind": "min"},
        "calcium_mg": {"value": 1200 if (female and age > 50) or (not female and age > 70) else 1000, "kind": "min"},
        "iron_mg": {"value": 18 if female and age <= 50 else 8, "kind": "min"},
    }
    if "low_sodium" in modes:
        targets["sodium_mg"] = {"value": MODE_SODIUM_MG, "kind": "max"}
    if "diabetes" in modes:
        targets["sugar_g"] = {"value": round(MODE_SUGAR_SHARE * calories / 4), "kind": "max"}
    upper = {"iron_mg": IRON_UL_MG, "calcium_mg": CALCIUM_UL_MG if age <= 50 else CALCIUM_UL_MG_51,
             "fiber_g": FIBER_CAP_TIMES * targets["fiber_g"]["value"]}
    for n, t in targets.items():
        t["upper"] = t["value"] if t["kind"] == "max" else upper.get(n)
    return targets


def filter_modes(foods, modes):
    """Leave out foods that don't fit a diet mode (very salty / sugary). Returns the remaining foods."""
    modes = parse_modes(modes)
    keep = pd.Series(True, index=foods.index)
    desc = foods["description"]
    if "low_sodium" in modes:
        keep &= ~((foods["sodium_mg"] > SALTY_MG_PER_100G) | desc.str.contains(SALTY_RE))
    if "diabetes" in modes:
        sugary_text = desc.str.contains(SUGARY_RE) & (foods["sugar_g"] > 5)
        mostly_sugar = (foods["sugar_g"] > 15) & (foods["meal_role"] != "fruit")
        keep &= ~(sugary_text | mostly_sugar)
    return foods[keep]


def weights(modes=()):
    """(encourage, limit): {nutrient: weight} for the planner, including the diet modes' extra weights."""
    encourage, limit = dict(ENCOURAGE), dict(LIMIT)
    for m in parse_modes(modes):
        for n, w in MODE_ENCOURAGE.get(m, {}).items():
            encourage[n] = encourage.get(n, 0) + w
        for n, w in MODE_LIMIT.get(m, {}).items():
            limit[n] = limit.get(n, 0) + w
    return encourage, limit


def nutrient_parts(foods, targets, calories, modes=()):
    """One column per nutrient: + for nutrients a food brings in more than their fair share, - for limits it would
    push over. fair share = daily target / daily calories (per kcal); ratio = the food's amount per kcal / fair share.
    Kept apart so the planner can stop rewarding a nutrient once the day already has enough (pacing)."""
    kcal = foods["calories"].clip(lower=20)  # very low-calorie foods (lettuce...) would get huge ratios
    encourage, limit = weights(modes)
    parts = {}
    for n, w in encourage.items():
        if has_target(targets[n]):
            ratio = foods[n] / kcal / (targets[n]["value"] / calories)
            parts[n] = w * ratio.clip(0, 2) / 2
    for n, w in limit.items():
        if has_target(targets[n]):
            ratio = foods[n] / kcal / (targets[n]["value"] / calories)
            parts[n] = parts.get(n, 0) - w * (ratio - 1).clip(0, 2) / 2
    return pd.DataFrame(parts, index=foods.index)


def nutrient_score(foods, targets, calories, modes=()):
    """Total nutrient score of each food (sum of nutrient_parts)."""
    return nutrient_parts(foods, targets, calories, modes).sum(axis=1)


def add_micros(items, foods):
    """Plan items (fdc_id, grams) + the food table -> the same items with the amount of each nutrient."""
    per100 = foods.set_index("fdc_id")[MICROS]
    out = items.copy()
    amounts = per100.reindex(out["fdc_id"]).to_numpy() * out["grams"].to_numpy()[:, None] / 100
    out[MICROS] = pd.DataFrame(amounts, index=out.index, columns=MICROS).fillna(0.0)
    return out


def day_micros(items, foods):
    """One row per day_index with the day's total of each nutrient."""
    return add_micros(items, foods).groupby("day_index")[MICROS].sum()


def has_target(target):
    return target["kind"] != "info" and target["value"] is not None and not pd.isna(target["value"])


def status(value, target):
    """'ok' / 'close' / 'low' / 'high' / 'info' for one nutrient of one day."""
    if not has_target(target):
        return "info"
    if target["kind"] == "max":
        return "ok" if value <= target["value"] else "high"
    if value >= target["value"]:
        return "ok"
    return "close" if value >= CLOSE_ENOUGH * target["value"] else "low"


def week_summary(items, foods, targets):
    """Per nutrient: daily average, target, and on how many days the target was met (for the app and report)."""
    days = day_micros(items, foods)
    rows = []
    for m in MICROS:
        t = targets[m]
        met = sum(status(v, t) == "ok" for v in days[m]) if t["kind"] != "info" else None
        rows.append({"nutrient": m, "label": LABELS[m][0], "unit": LABELS[m][1], "average": float(days[m].mean()),
                     "target": t["value"], "kind": t["kind"], "days_met": met, "days": len(days)})
    return pd.DataFrame(rows)


def fmt(value, unit):
    return f"{value:,.0f} {unit}" if unit == "mg" else f"{value:.0f} {unit}"


def target_text(target, unit):
    """'≥ 35 g' / '≤ 2,300 mg' / '' for display."""
    if not has_target(target):
        return ""
    return ("≥ " if target["kind"] == "min" else "≤ ") + fmt(target["value"], unit)
