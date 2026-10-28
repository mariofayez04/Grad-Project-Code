"""Feature #3: weekly shopping list.

Adds up every food of the week's plan and turns it into what to buy:
 - grouped by shop section (meat & poultry, fish, eggs & dairy, grains, vegetables...)
 - cooked weights converted to the approximate raw / dry weight to buy: cooked rice weighs about 3x the dry
   rice, cooked beans about 2.5x the dry beans, and meat loses about a quarter of its raw weight when cooked
 - eggs also counted in pieces
 - an Egyptian dish (Feature #9) is split into its ingredients, scaled to the portion
The plate weights stay in the plan itself; the list is only a shopping aid, so the factors are rounded averages.

Run a demo:  python -m src.shopping
"""
import re

import pandas as pd

from src.embeddings import embed_text

SECTIONS = ["🥩 Meat & poultry", "🐟 Fish & seafood", "🥚 Eggs & dairy", "🍞 Bread & cereals", "🍚 Rice, pasta & grains",
            "🫘 Beans & lentils", "🥦 Vegetables", "🍎 Fruits", "🥜 Nuts & seeds", "🫒 Oils & fats",
            "🧂 Herbs, spices & sauces", "🛒 Other"]
CATEGORY_SECTION = {
    "Beef Products": 0, "Pork Products": 0, "Poultry Products": 0, "Lamb, Veal, and Game Products": 0,
    "Sausages and Luncheon Meats": 0, "Finfish and Shellfish Products": 1, "Dairy and Egg Products": 2,
    "Baked Products": 3, "Breakfast Cereals": 3, "Cereal Grains and Pasta": 4, "Legumes and Legume Products": 5,
    "Vegetables and Vegetable Products": 6, "Fruits and Fruit Juices": 7, "Nut and Seed Products": 8, "Fats and Oils": 9,
    "Spices and Herbs": 10, "Soups, Sauces, and Gravies": 10, "Beverages": 10, "Sweets": 11,
}
MEAT_CATEGORIES = {"Beef Products", "Pork Products", "Poultry Products", "Lamb, Veal, and Game Products",
                   "Finfish and Shellfish Products"}

COOKED_RE = re.compile(r"\b(cooked|boiled|roasted|grilled|broiled|baked|braised|stewed|simmered|steamed|microwaved|"
                       r"pan-fried|pan-broiled|fried|poached|scrambled|hard-boiled|rotisserie|moist heat|dry heat|"
                       r"heated|crumbles|with water|prepared with water)\b", re.I)
READY_RE = re.compile(r"\b(canned|cured|smoked|pickled|salted|surimi|luncheon|frozen dinner|with broth)\b", re.I)
RAW_MEAT_FACTOR = 1.35       # raw weight to buy per gram of cooked meat or fish
# (description pattern, food category or None, bought grams per plate gram, label) - first match wins.
DRY_YIELD = [
    (r"^pasta, fresh", "Cereal Grains and Pasta", 0.8, "fresh"),
    (r"^(rice|wild rice)\b", "Cereal Grains and Pasta", 0.35, "dry"),
    (r"^(pasta|spaghetti|macaroni|noodles)\b", "Cereal Grains and Pasta", 0.45, "dry"),
    (r"^(bulgur|couscous|barley|quinoa|freekeh|millet|buckwheat)\b", "Cereal Grains and Pasta", 0.4, "dry"),
    (r"^cereals, oats\b", "Breakfast Cereals", 0.17, "dry"),
    (r".", "Legumes and Legume Products", 0.4, "dry"),
]
EGG_RE = re.compile(r"^egg\b", re.I)
EGG_GRAMS = 50               # one large egg without shell
OIL_DENSITY = 0.92           # g per ml
# USDA names -> what you would ask for in a shop (mostly ingredients of the Egyptian dishes).
SHOP_NAMES = {"Broadbeans": "Fava beans (ful)", "Spices, cumin seed": "Cumin", "Spices, cinnamon": "Cinnamon",
              "Falafel, home": "Ta'ameya (falafel) mix", "Seeds, sesame butter": "Tahini", "Spearmint": "Mint",
              "Dill weed": "Dill", "Vinegar, distilled": "Vinegar", "Jute, potherb": "Molokhia leaves",
              "Soup, chicken broth": "Chicken broth", "Butter oil, anhydrous": "Ghee", "Coriander leaves": "Fresh coriander",
              "Tomato products, canned": "Tomato sauce / puree (canned)", "Sugars, granulated": "Sugar",
              "Bread, pita": "Baladi / pita bread", "Squash, summer": "Zucchini", "Pasta, without added salt": "Pasta",
              "Lemon juice": "Lemons (for juice)", "Tomatoes, red": "Tomatoes", "Peppers, sweet": "Green peppers",
              "Cucumber, with peel": "Cucumber", "Potatoes, flesh": "Potatoes"}

# Words that describe the plate, not the thing to buy ("Asparagus, cooked, boiled" -> "Asparagus").
PLATE_WORDS_RE = re.compile(COOKED_RE.pattern[:-3] + r"|raw|toasted|drained|as purchased|retail parts|from whole bird|"
                            r"whole bird|all classes|separable lean)\b", re.I)


def shop_name(description, category):
    """The name on the shopping list: same food + cut / type, without plate words, so variants merge.
    'Chicken, broilers or fryers, breast, meat only, cooked, roasted' -> 'Chicken, breast'."""
    if EGG_RE.search(description):
        return "Eggs"
    if category == "Fats and Oils" and re.match(r"oil\b", description, re.I):
        return "Cooking oil"
    parts = [p.strip() for p in embed_text(description).split(",")]
    parts = [parts[0]] + [p.lower() for p in parts[1:] if p and not PLATE_WORDS_RE.search(p)]  # "Swiss" = "swiss"
    name = " ".join(", ".join(parts[:3 if category in MEAT_CATEGORIES else 2]).split())  # also "Coriander  leaves"
    return SHOP_NAMES.get(name, name)


def purchase(description, category):
    """What to buy for one plan food: (name, bought grams per plate gram, label).
    'Rice, white, long-grain, regular, enriched, cooked' -> ('Rice, white', 0.35, 'dry')."""
    name = shop_name(description, category)
    if not COOKED_RE.search(description) or READY_RE.search(description):
        return name, 1.0, ""
    if category in MEAT_CATEGORIES:
        return name, RAW_MEAT_FACTOR, "raw"
    for pattern, cat, factor, label in DRY_YIELD:
        if cat == category and re.search(pattern, description, re.I):
            return name, factor, label
    return name, 1.0, ""


def format_amount(grams):
    if grams >= 1000:
        return f"{grams / 1000:.1f} kg"
    return f"{max(5, round(grams / 10) * 10 if grams >= 50 else round(grams / 5) * 5):.0f} g"


def expand_dishes(items, recipes=None):
    """Replace each Egyptian dish in the plan items by its ingredients, scaled to the portion served."""
    from src.dishes import load_recipes
    recipes = load_recipes() if recipes is None else recipes
    is_dish = items["fdc_id"].isin(recipes["dish_fdc_id"])
    if not is_dish.any():
        return items
    parts = items[is_dish].drop(columns=[c for c in ("name",) if c in items]).merge(
        recipes[["dish_fdc_id", "fdc_id", "grams", "serving_g"]].rename(columns={"fdc_id": "ing_id", "grams": "ing_g"}),
        left_on="fdc_id", right_on="dish_fdc_id")
    parts["grams"] = parts["grams"] * parts["ing_g"] / parts["serving_g"]
    parts["fdc_id"] = parts["ing_id"]
    return pd.concat([items[~is_dish], parts[[c for c in items.columns if c in parts.columns]]], ignore_index=True)


def shopping_list(items, foods, skip_eaten=False):
    """items: plan items (fdc_id, grams, day_index, slot, eaten). Returns one row per thing to buy:
    section, item, label, grams (to buy), plate_g (as served), meals (how many meals use it), note."""
    if skip_eaten and "eaten" in items:
        items = items[items["eaten"].fillna(0) != 1]
    cols = ["section", "item", "label", "grams", "plate_g", "meals", "note"]
    if items.empty:
        return pd.DataFrame(columns=cols)
    from src.dishes import load_recipes
    recipes = load_recipes()
    items = expand_dishes(items, recipes)
    info = pd.concat([foods[["fdc_id", "description", "food_category"]],
                      recipes[["fdc_id", "description", "food_category"]]]).drop_duplicates("fdc_id").set_index("fdc_id")
    df = items.join(info, on="fdc_id")
    bought = [purchase(d, c) for d, c in zip(df["description"], df["food_category"])]
    df["item"] = [b[0] for b in bought]
    df["label"] = [b[2] for b in bought]
    df["plate_g"] = df["grams"]
    df["grams"] = df["grams"] * [b[1] for b in bought]
    df["section_i"] = df["food_category"].map(CATEGORY_SECTION).fillna(len(SECTIONS) - 1).astype(int)
    df["meal"] = df["day_index"].astype(str) + df["slot"]

    out = (df.groupby(["section_i", "item", "label"], as_index=False)
             .agg(grams=("grams", "sum"), plate_g=("plate_g", "sum"), meals=("meal", "nunique")))
    out["section"] = out["section_i"].map(lambda i: SECTIONS[i])
    oils = df.loc[df["item"] == "Cooking oil", "description"].str.split(",").str[1].str.strip().value_counts().index[:3]
    out["note"] = [f"≈ {max(1, round(g / EGG_GRAMS))} eggs" if item == "Eggs" else
                   f"≈ {round(g / OIL_DENSITY / 10) * 10:.0f} ml, any kind (e.g. {', '.join(oils)})" if item == "Cooking oil" else ""
                   for g, item in zip(out["grams"], out["item"])]
    out = out.sort_values(["section_i", "grams"], ascending=[True, False]).reset_index(drop=True)
    return out[cols]


def to_text(shop, title="Shopping list"):
    """Plain-text list with tick boxes, for printing or sending to a phone."""
    lines = [title, "=" * len(title), "Amounts for the whole week. Rice, pasta, beans and meat: approximate raw / dry weight.", ""]
    for section, rows in shop.groupby("section", sort=False):
        lines.append(section)
        for r in rows.itertuples():
            extra = " ".join(x for x in (f"({r.label})" if r.label else "", r.note) if x)
            lines.append(f"  [ ] {r.item} - {format_amount(r.grams)}" + (f" {extra}" if extra else ""))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    sys.stdout.reconfigure(encoding="utf-8")  # section emoji on the Windows console
    from src.calculator import UserProfile
    from src.config import FOODS_CLEAN
    from src.planner import generate_week_plan
    foods = pd.read_csv(FOODS_CLEAN)
    plan, _, _ = generate_week_plan(UserProfile("Demo", 30, "male", 75, 175, 2, "maintain"), foods=foods, seed=1)
    print(to_text(shopping_list(plan.to_dataframe(), foods)))
