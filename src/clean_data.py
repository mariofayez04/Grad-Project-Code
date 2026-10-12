"""Phase 1: build data/processed/foods_clean.csv from the raw USDA SR Legacy tables.

Steps
 1. Join food + food_category, pivot the 4 nutrients (per 100 g) into columns.
 2. Pick one realistic portion per food from food_portion (modifier + gram_weight).
 3. Remove foods that don't belong in a normal meal plan (categories, brands, raw meat, organs, mixes...).
 4. Tag allergens and diet flags (deterministic, see allergens.py).
 5. Assign a meal_role used by the planner (protein / carb / vegetable / ...).

Run:  python -m src.clean_data
"""
import re

import pandas as pd

from src.allergens import ALLERGEN_GROUPS, diet_flags, tag_allergens
from src.config import FOODS_CLEAN, NUTRIENT_IDS, PROCESSED_DIR, USDA_DIR

DROP_CATEGORIES = {
    "Baby Foods", "Fast Foods", "Restaurant Foods", "American Indian/Alaska Native Foods",
    "Spices and Herbs", "Beverages",
}

# Categories where a raw/dry/uncooked item should be replaced by its cooked version.
COOK_FIRST_CATEGORIES = {
    "Beef Products", "Pork Products", "Poultry Products", "Lamb, Veal, and Game Products",
    "Finfish and Shellfish Products", "Sausages and Luncheon Meats", "Cereal Grains and Pasta",
    "Legumes and Legume Products",
}
RAW_RE = re.compile(r"\braw\b|\buncooked\b|\bunheated\b|\bdry\b(?![- ]?roasted)", re.I)
RAW_OK_RE = re.compile(r"\b(?:tofu|tempeh|hummus|soymilk|peanuts?|smoked|cured|canned|kippered|pickled)\b", re.I)

DROP_ANYWHERE = [
    # processing states that aren't eaten as-is
    r"powder(?:ed)?", r"dehydrated", r"freeze-dried", r"unprepared", r"condensed", r"concentrate",
    r"industrial", r"imitation", r"formula", r"infant", r"babyfood", r"toddler",
    r"with salt", r"mechanically separated", r"separable fat", r"separable lean and fat", r"skin only",
    r"meat and skin", r"variety meats", r"by-products", r"leavening", r"yeast extract", r"shortening",
    r"baking chocolate", r"sugars?,", r"syrups?", r"lard", r"tallow", r"suet",
    # organs (kidney/turtle are not listed: they would remove kidney beans and black turtle beans)
    r"liver", r"heart", r"brains?", r"tongue", r"tripe", r"gizzards?", r"giblets", r"spleen",
    r"lungs?", r"pancreas", r"thymus", r"sweetbreads?", r"feet", r"tail", r"ears", r"chitterlings", r"blood",
    r"testes", r"marrow", r"cracklings",
    # game / unusual meats
    r"deer", r"venison", r"elk", r"bison", r"buffalo", r"moose", r"caribou", r"bear", r"rabbit",
    r"squirrel", r"beaver", r"muskrat", r"opossum", r"raccoon", r"horse", r"frog legs", r"whale",
    r"seal", r"emu", r"ostrich", r"pheasant", r"antelope", r"boar", r"dove", r"squab",
]
DROP_ANYWHERE_RE = re.compile(r"\b(?:" + "|".join(DROP_ANYWHERE) + r")\b", re.I)
# A dry mix is only kept once it has been prepared (e.g. "Puddings, dry mix, prepared with 2% milk").
DRY_MIX_RE = re.compile(r"\b(?:dry mix|mix,? dry)\b", re.I)

# Grain ingredients that aren't a dish on their own.
GRAIN_INGREDIENT_RE = re.compile(r"\b(?:flour|bran|starch|germ|gluten|meal|semolina|masa)\b", re.I)

# Brands: SR Legacy writes them in CAPS (QUAKER, KFC, ...) or as known mixed-case names.
CAPS_WHITELIST = {"USDA", "DHA", "ARA", "BBQ", "USA", "NFS", "RTE", "UHT", "NLEA", "II", "III"}
CAPS_RE = re.compile(r"\b[A-Z][A-Z'&.]{2,}\b")
BRANDS_RE = re.compile(
    r"\b(Pillsbury|Kraft|Kellogg|Keebler|Nabisco|Pepperidge|Archway|Goya|Gamesa|La Moderna|George Weston|"
    r"Thomas|Martha White|Mission|Heinz|Campbell|Hormel|Oscar Mayer|Tyson|Banquet|Stouffer|Lean Cuisine|"
    r"Healthy Choice|Swanson|Hunt|Del Monte|Dole|Ocean Spray|Sara Lee|Entenmann|Hostess|Little Debbie|"
    r"Mrs\.? Smith|Marie Callender|Kashi|Quaker|General Mills|Post|Nestle|Hershey|Mars|Frito|Lay's|Ruffles|"
    r"Tostitos|Doritos|Cheetos|Snyder|Rold Gold|Sunshine|Murray|Mother's|Nature Valley|Fiber One|"
    r"Glutino|Sage Valley|Schar|Canyon Bakehouse|Kinnikinnick|Bob's Red Mill|Ener-G|Gardenburger|Morningstar|"
    r"Boca|Amy's|Annie's|Udi's|Rudi's|Van's)\b"
    r"|^[A-Z][\w.&-]*(?: [A-Z][\w.&-]*)*'s,"  # "Udi's, ..." style brand prefix
)

STARCHY_VEG_RE = re.compile(r"\b(potato(es)?|sweet potato(es)?|yams?|corn|plantains?|cassava|taro|yautia|breadfruit|parsnips?)\b", re.I)
EGG_RE = re.compile(r"^egg|\begg(s)?,", re.I)
HIGH_PROTEIN_LEGUME_RE = re.compile(r"\b(tofu|tempeh|seitan|soy protein|textured vegetable protein)\b", re.I)
WEIGHT_UNITS = {"oz", "lb", "g", "kg", "fl oz"}
MAX_PORTION_G = 350


def load(name, **kw):
    return pd.read_csv(USDA_DIR / f"{name}.csv", low_memory=False, **kw)


def build_base():
    """Steps 1 + 2: one row per food with macros per 100 g and a default portion."""
    food = load("food", usecols=["fdc_id", "description", "food_category_id"])
    cat = load("food_category", usecols=["id", "description"]).rename(columns={"id": "food_category_id", "description": "food_category"})
    df = food.merge(cat, on="food_category_id", how="left").drop(columns="food_category_id")

    fn = load("food_nutrient", usecols=["fdc_id", "nutrient_id", "amount"])
    fn = fn[fn["nutrient_id"].isin(NUTRIENT_IDS.values())]
    macros = fn.pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first")
    macros = macros.rename(columns={v: k for k, v in NUTRIENT_IDS.items()}).reset_index()
    df = df.merge(macros, on="fdc_id", how="left")

    portions = load("food_portion", usecols=["fdc_id", "seq_num", "amount", "modifier", "gram_weight"])
    portions = portions[portions["gram_weight"] > 0].copy()
    portions["modifier"] = portions["modifier"].fillna("").str.strip()
    portions["is_weight"] = portions["modifier"].str.lower().isin(WEIGHT_UNITS)
    portions["too_big"] = portions["gram_weight"] > MAX_PORTION_G
    # Prefer a normal-sized household measure (e.g. "1 cup", "1 slice") over a whole roast/pie or a bare "1 oz".
    portions = portions.sort_values(["fdc_id", "too_big", "is_weight", "seq_num"]).drop_duplicates("fdc_id")
    portions["portion_desc"] = portions["amount"].map("{:g}".format) + " " + portions["modifier"]
    df = df.merge(portions[["fdc_id", "portion_desc", "gram_weight"]].rename(columns={"gram_weight": "portion_g"}), on="fdc_id", how="left")
    df["portion_desc"] = df["portion_desc"].fillna("100 g")
    df["portion_g"] = df["portion_g"].fillna(100.0).round(1)
    return df


def is_brand(desc):
    caps = [t for t in CAPS_RE.findall(desc) if t.strip("'.") not in CAPS_WHITELIST]
    return bool(caps) or bool(BRANDS_RE.search(desc))


def filter_foods(df, log):
    """Step 3: keep only foods that make sense in a normal meal plan."""
    def drop(mask, reason):
        nonlocal df
        log.append(f"  - {reason:<55} {int(mask.sum()):>5} dropped")
        df = df[~mask]

    log.append(f"Start: {len(df)} foods")
    drop(df[["calories", "protein_g", "carbs_g", "fat_g"]].isna().any(axis=1), "missing calories or a macro")
    drop(df["calories"] <= 0, "zero calories")
    drop(df["food_category"].isin(DROP_CATEGORIES), "category not used for meals")
    drop(df["description"].map(is_brand), "brand-name product")
    drop(df["description"].str.contains(DROP_ANYWHERE_RE), "processed form / organ / game / with salt")
    drop(df["description"].str.contains(DRY_MIX_RE) & ~df["description"].str.contains(r"\bprepared\b"), "dry mix not prepared")
    cook_first = df["food_category"].isin(COOK_FIRST_CATEGORIES) | df["description"].str.match(EGG_RE)
    drop(cook_first & df["description"].str.contains(RAW_RE) & ~df["description"].str.contains(RAW_OK_RE), "raw/dry version of food eaten cooked")
    drop((df["food_category"] == "Cereal Grains and Pasta") & df["description"].str.contains(GRAIN_INGREDIENT_RE), "grain ingredient (flour, bran, starch...)")
    drop((df["food_category"] == "Soups, Sauces, and Gravies") & ~df["description"].str.match(r"Soup\b"), "sauce / gravy")
    drop(df["description"].str.contains(r"\bjuice\b|\bnectar\b|\bdrink\b", case=False), "juice / drink")

    atwater = 4 * df["protein_g"] + 4 * df["carbs_g"] + 9 * df["fat_g"]
    drop((df["calories"] > 50) & ((atwater - df["calories"]).abs() / df["calories"] > 0.35), "calories inconsistent with macros (>35%)")
    log.append(f"End: {len(df)} foods")
    return df


def meal_role(row):
    """Step 5: which slot of a meal this food can fill."""
    cat, desc = row["food_category"], row["description"]
    protein_share = 4 * row["protein_g"] / row["calories"]
    if cat in {"Beef Products", "Pork Products", "Poultry Products", "Lamb, Veal, and Game Products",
               "Finfish and Shellfish Products", "Sausages and Luncheon Meats"}:
        return "protein"
    if cat == "Dairy and Egg Products":
        return "protein" if EGG_RE.search(desc) else "dairy"
    if cat == "Legumes and Legume Products":
        return "protein" if HIGH_PROTEIN_LEGUME_RE.search(desc) or protein_share >= 0.35 else "legume"
    if cat in {"Cereal Grains and Pasta", "Baked Products", "Breakfast Cereals"}:
        return "carb"
    if cat == "Vegetables and Vegetable Products":
        return "carb" if STARCHY_VEG_RE.search(desc) else "vegetable"
    if cat == "Fruits and Fruit Juices":
        return "fruit"
    if cat in {"Fats and Oils", "Nut and Seed Products"}:
        return "fat"
    if cat in {"Snacks", "Sweets"}:
        return "snack"
    return "mixed"  # Meals/Entrees, Soups


def main():
    log = []
    df = build_base()
    df = filter_foods(df, log)

    df["allergens"] = [tag_allergens(d, c) for d, c in zip(df["description"], df["food_category"])]
    df["is_allergen_free"] = df["allergens"].map(lambda a: len(a) == 0)
    flags = [diet_flags(d, c) for d, c in zip(df["description"], df["food_category"])]
    df["is_vegetarian"], df["contains_pork"], df["contains_alcohol"] = zip(*flags)
    df["allergens"] = df["allergens"].map("|".join)
    df["meal_role"] = df.apply(meal_role, axis=1)
    df["base_name"] = df["description"].str.split(",").str[0].str.strip().str.lower()

    cols = ["fdc_id", "description", "food_category", "meal_role", "base_name", "calories", "protein_g", "carbs_g",
            "fat_g", "portion_desc", "portion_g", "allergens", "is_allergen_free", "is_vegetarian", "contains_pork",
            "contains_alcohol"]
    df = df[cols].sort_values(["food_category", "description"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FOODS_CLEAN, index=False)

    log.append("\nMeal roles:\n" + df["meal_role"].value_counts().to_string())
    log.append("\nCategories kept:\n" + df["food_category"].value_counts().to_string())
    counts = {g: int(df["allergens"].str.split("|").map(lambda a: g in a).sum()) for g in ALLERGEN_GROUPS}
    log.append("\nAllergen tags: " + ", ".join(f"{g}={n}" for g, n in counts.items()))
    log.append(f"Allergen-free foods: {int(df['is_allergen_free'].sum())}")
    log.append(f"Vegetarian: {int(df['is_vegetarian'].sum())}, contains pork: {int(df['contains_pork'].sum())}, "
               f"contains alcohol: {int(df['contains_alcohol'].sum())}")
    log.append(f"Foods using a real household portion: {int((df['portion_desc'] != '100 g').sum())}")
    report = "\n".join(log)
    (PROCESSED_DIR / "cleaning_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved {len(df)} foods -> {FOODS_CLEAN}")


if __name__ == "__main__":
    main()
