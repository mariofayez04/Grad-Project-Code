"""Feature #9: Egyptian dishes, built as recipes from USDA SR Legacy ingredients.

USDA has no Egyptian dishes, so each dish is one serving "as served": a list of USDA ingredients (by their exact
USDA description) with grams. A dish's nutrients per 100 g are the gram-weighted sum of its ingredients; its
allergens and diet flags come from the ingredients (deterministic, as for every food). Dishes are added to
foods_clean.csv by clean_data.py (category "Egyptian Dishes", meal_role "dish") and the recipes are saved in
dish_ingredients.csv, which the shopping list uses to buy the ingredients instead of "a koshari".

Notes for the report:
 - Amounts follow common home recipes, without added salt (the sodium of salt added when cooking is not counted).
 - USDA's falafel is chickpea-based; Egyptian ta'ameya uses fava beans - the nutrients are close.
 - Desserts (basbousa, konafa, om ali) are left out on purpose; roz bel laban is the one sweet snack.
"""
import pandas as pd

DISH_ID_START = 9_000_001   # synthetic fdc_ids, far from USDA's
CATEGORY = "Egyptian Dishes"

# Ingredients (exact USDA SR Legacy descriptions)
FAVA = "Broadbeans (fava beans), mature seeds, cooked, boiled, without salt"
FALAFEL = "Falafel, home-prepared"
LENTILS = "Lentils, mature seeds, cooked, boiled, without salt"
CHICKPEAS = "Chickpeas (garbanzo beans, bengal gram), mature seeds, cooked, boiled, without salt"
WHITE_BEANS = "Beans, white, mature seeds, cooked, boiled, without salt"
RICE = "Rice, white, long-grain, regular, enriched, cooked"
RICE_SHORT = "Rice, white, short-grain, enriched, cooked"
PASTA = "Pasta, cooked, enriched, without added salt"
PITA = "Bread, pita, white, enriched"
PITA_WW = "Bread, pita, whole-wheat"
TOMATO_SAUCE = "Tomato products, canned, sauce"
TOMATO_PUREE = "Tomato products, canned, puree, without salt added"
TOMATO = "Tomatoes, red, ripe, raw, year round average"
ONION = "Onions, raw"
ONION_C = "Onions, cooked, boiled, drained, without salt"
GARLIC = "Garlic, raw"
CUMIN = "Spices, cumin seed"
CINNAMON = "Spices, cinnamon, ground"
CORIANDER = "Coriander (cilantro) leaves, raw"
PARSLEY = "Parsley, fresh"
DILL = "Dill weed, fresh"
MINT = "Spearmint, fresh"
LEMON = "Lemon juice, raw"
VINEGAR = "Vinegar, distilled"
JUTE = "Jute, potherb, cooked, boiled, drained, without salt"
BROTH = "Soup, chicken broth, ready-to-serve"
CHICKEN = "Chicken, broilers or fryers, breast, meat only, cooked, roasted"
CHICKEN_THIGH = "Chicken, broilers or fryers, thigh, meat only, cooked, roasted"
BEEF = 'Beef, chuck, arm pot roast, separable lean only, trimmed to 1/8" fat, all grades, cooked, braised'
BEEF_MINCE = "Beef, ground, 85% lean meat / 15% fat, patty, cooked, broiled"
BEEF_MINCE_LEAN = "Beef, ground, 90% lean meat / 10% fat, patty, cooked, broiled"
TILAPIA = "Fish, tilapia, cooked, dry heat"
EGG = "Egg, whole, raw, fresh"
COTTAGE = "Cheese, cottage, lowfat, 1% milkfat"
MILK = "Milk, whole, 3.25% milkfat, with added vitamin D"
TAHINI = "Seeds, sesame butter, tahini, type of kernels unspecified"
OLIVE_OIL = "Oil, olive, salad or cooking"
SUN_OIL = "Oil, sunflower, linoleic (less than 60%)"
GHEE = "Butter oil, anhydrous"
SUGAR = "Sugars, granulated"
ZUCCHINI = "Squash, summer, zucchini, includes skin, cooked, boiled, drained, without salt"
PEPPER = "Peppers, sweet, green, raw"
PEPPER_C = "Peppers, sweet, green, cooked, boiled, drained, without salt"
EGGPLANT = "Eggplant, cooked, boiled, drained, without salt"
OKRA = "Okra, cooked, boiled, drained, without salt"
POTATO = "Potatoes, boiled, cooked without skin, flesh, without salt"
CARROT = "Carrots, cooked, boiled, drained, without salt"
PEAS = "Peas, green, cooked, boiled, drained, without salt"
SPINACH = "Spinach, cooked, boiled, drained, without salt"
CUCUMBER = "Cucumber, with peel, raw"

# Short names shown in the recipe line of a dish
LABELS = {FAVA: "fava beans", FALAFEL: "falafel", LENTILS: "lentils", CHICKPEAS: "chickpeas", WHITE_BEANS: "white beans",
          RICE: "rice", RICE_SHORT: "short-grain rice", PASTA: "pasta", PITA: "baladi bread", PITA_WW: "whole-wheat baladi bread",
          TOMATO_SAUCE: "tomato sauce", TOMATO_PUREE: "tomato puree", TOMATO: "tomato", ONION: "onion", ONION_C: "onion",
          GARLIC: "garlic", CUMIN: "cumin", CINNAMON: "cinnamon", CORIANDER: "fresh coriander", PARSLEY: "parsley",
          DILL: "dill", MINT: "mint", LEMON: "lemon juice", VINEGAR: "vinegar", JUTE: "molokhia leaves",
          BROTH: "chicken broth", CHICKEN: "chicken breast", CHICKEN_THIGH: "chicken thigh", BEEF: "beef",
          BEEF_MINCE: "minced beef", BEEF_MINCE_LEAN: "lean minced beef", TILAPIA: "tilapia", EGG: "eggs",
          COTTAGE: "cottage cheese (gebna)", MILK: "milk", TAHINI: "tahini", OLIVE_OIL: "olive oil", SUN_OIL: "sunflower oil",
          GHEE: "ghee", SUGAR: "sugar", ZUCCHINI: "zucchini", PEPPER: "green pepper", PEPPER_C: "green pepper",
          EGGPLANT: "eggplant", OKRA: "okra", POTATO: "potatoes", CARROT: "carrots", PEAS: "peas", SPINACH: "spinach",
          CUCUMBER: "cucumber"}

# name, Arabic name, short description, slots, ingredients (USDA description, grams per serving)
DISHES = [
    ("Ful medames", "فول مدمس", "fava beans with olive oil, cumin and lemon", ["breakfast", "dinner"],
     [(FAVA, 200), (OLIVE_OIL, 10), (LEMON, 10), (TOMATO, 40), (PARSLEY, 5), (GARLIC, 3), (CUMIN, 1)]),
    ("Taameya", "طعمية", "Egyptian falafel with tahini, tomato and cucumber", ["breakfast", "dinner"],
     [(FALAFEL, 120), (TAHINI, 15), (TOMATO, 40), (CUCUMBER, 40)]),
    ("Shakshuka", "شكشوكة", "eggs cooked in tomato and green pepper sauce", ["breakfast", "dinner"],
     [(EGG, 100), (TOMATO, 150), (PEPPER, 50), (ONION, 40), (OLIVE_OIL, 10), (CUMIN, 1)]),
    ("Gebna with tomatoes", "جبنة بالطماطم", "cottage cheese salad with cucumber, mint and olive oil", ["breakfast"],
     [(COTTAGE, 150), (TOMATO, 80), (CUCUMBER, 60), (OLIVE_OIL, 7), (MINT, 2)]),
    ("Bessara", "بصارة", "fava bean and herb puree with crispy onions", ["breakfast", "lunch", "dinner"],
     [(FAVA, 180), (SPINACH, 40), (PARSLEY, 15), (DILL, 10), (ONION, 30), (OLIVE_OIL, 12), (GARLIC, 4), (CUMIN, 1)]),
    ("Koshari", "كشري", "rice, lentils and pasta with spicy tomato sauce, chickpeas and crispy onions", ["lunch", "dinner"],
     [(RICE, 120), (LENTILS, 80), (PASTA, 80), (CHICKPEAS, 30), (TOMATO_SAUCE, 80), (ONION, 40), (SUN_OIL, 15),
      (GARLIC, 3), (VINEGAR, 5), (CUMIN, 1)]),
    ("Molokhia with rice and chicken", "ملوخية بالأرز والفراخ", "jute leaf stew with garlic and coriander", ["lunch", "dinner"],
     [(JUTE, 200), (BROTH, 150), (RICE, 150), (CHICKEN, 100), (GHEE, 8), (GARLIC, 6), (CORIANDER, 3)]),
    ("Mahshi", "محشي", "zucchini and peppers stuffed with herbed rice in tomato sauce", ["lunch", "dinner"],
     [(ZUCCHINI, 120), (PEPPER_C, 60), (RICE_SHORT, 130), (TOMATO_PUREE, 60), (ONION, 30), (SUN_OIL, 12), (PARSLEY, 8),
      (DILL, 5), (MINT, 2)]),
    ("Fattah", "فتة", "rice and toasted bread with beef in garlic and vinegar tomato sauce", ["lunch"],
     [(RICE, 150), (PITA, 50), (BEEF, 110), (BROTH, 100), (TOMATO_PUREE, 60), (GHEE, 10), (GARLIC, 6), (VINEGAR, 8)]),
    ("Bamya with rice", "بامية بالأرز", "okra and beef stew in tomato sauce", ["lunch", "dinner"],
     [(OKRA, 180), (BEEF, 90), (TOMATO_PUREE, 80), (RICE, 150), (ONION, 30), (SUN_OIL, 10), (GARLIC, 4), (CORIANDER, 3)]),
    ("Kofta", "كفتة", "grilled spiced minced beef with tahini, tomato and cucumber", ["lunch", "dinner"],
     [(BEEF_MINCE_LEAN, 150), (ONION, 20), (PARSLEY, 8), (CUMIN, 1), (TAHINI, 15), (TOMATO, 60), (CUCUMBER, 60)]),
    ("Hawawshi", "حواوشي", "baladi bread stuffed with spiced minced beef, onion and pepper", ["lunch", "dinner"],
     [(PITA_WW, 110), (BEEF_MINCE, 110), (ONION, 30), (PEPPER, 25), (SUN_OIL, 5)]),
    ("Sayadiya", "صيادية", "fish with caramelised onion rice", ["lunch", "dinner"],
     [(TILAPIA, 150), (RICE, 180), (ONION, 50), (SUN_OIL, 12), (TOMATO_PUREE, 30), (CUMIN, 1)]),
    ("Fasolia with rice", "فاصوليا بالأرز", "white bean and beef stew in tomato sauce", ["lunch", "dinner"],
     [(WHITE_BEANS, 150), (BEEF, 70), (TOMATO_PUREE, 80), (RICE, 130), (ONION, 30), (SUN_OIL, 8), (GARLIC, 3)]),
    ("Shorbet ads", "شوربة عدس", "lentil soup with carrots, cumin and lemon", ["lunch", "dinner"],
     [(LENTILS, 220), (BROTH, 150), (CARROT, 40), (ONION_C, 40), (SUN_OIL, 8), (LEMON, 10), (CUMIN, 2)]),
    ("Mesaqaa", "مسقعة", "Egyptian moussaka, eggplant baked with minced beef, pepper and tomato", ["lunch", "dinner"],
     [(EGGPLANT, 200), (BEEF_MINCE, 60), (TOMATO_PUREE, 100), (PEPPER_C, 40), (ONION, 30), (SUN_OIL, 15), (GARLIC, 4)]),
    ("Torly", "تورلي", "mixed vegetable and beef stew with potatoes, zucchini, carrots and peas", ["lunch", "dinner"],
     [(POTATO, 100), (ZUCCHINI, 70), (CARROT, 50), (PEAS, 50), (BEEF, 70), (TOMATO_PUREE, 80), (ONION, 30), (SUN_OIL, 10)]),
    ("Ferakh bel roz", "فراخ بالأرز", "roast chicken with vermicelli rice", ["lunch", "dinner"],
     [(CHICKEN_THIGH, 130), (RICE, 170), (PASTA, 20), (GHEE, 6)]),
    ("Halabessa", "حلبسة", "hot chickpeas in spiced broth with lemon and cumin", ["snack", "breakfast"],
     [(CHICKPEAS, 160), (BROTH, 120), (LEMON, 10), (CUMIN, 1)]),
    ("Roz bel laban", "رز بلبن", "Egyptian rice pudding with milk and cinnamon", ["snack"],
     [(MILK, 150), (RICE_SHORT, 60), (SUGAR, 15), (CINNAMON, 1)]),
]
NUTRIENT_COLS = ["calories", "protein_g", "carbs_g", "fat_g"]


def dish_description(name, details):
    return f"{name}, {details}"


def dishes_for(slot):
    """Names of the dishes eaten at a meal slot (the morning snack uses the snack dishes)."""
    slot = "snack" if slot == "morning_snack" else slot
    return [d[0] for d in DISHES if slot in d[3]]


ARABIC = {d[0].lower(): d[1] for d in DISHES}       # base_name -> Arabic name (for display)


def load_recipes():
    """The saved recipes (dish_ingredients.csv): one row per ingredient of each dish; empty if not built yet."""
    from src.config import DISH_INGREDIENTS
    if not DISH_INGREDIENTS.exists():
        return pd.DataFrame(columns=["dish_fdc_id", "dish", "fdc_id", "description", "label", "food_category",
                                     "grams", "serving_g"])
    return pd.read_csv(DISH_INGREDIENTS)


def recipe_line(recipes, dish_fdc_id, grams, min_grams=5):
    """'rice 130 g, lentils 90 g, ...' for a portion of a dish (tiny amounts like spices left out)."""
    r = recipes[recipes["dish_fdc_id"] == dish_fdc_id]
    if r.empty:
        return ""
    scaled = r.assign(g=r["grams"] * grams / r["serving_g"])
    return ", ".join(f"{x.label} {round(x.g / 5) * 5:.0f} g" for x in scaled.itertuples() if x.g >= min_grams)


def build_dishes(base, micro_cols, tag_allergens, diet_flags):
    """Dishes as food rows (same columns as foods_clean) + their recipes.
    base: every USDA food with macros and (estimated) extra nutrients per 100 g - incl. spices and juices that
    foods_clean leaves out. Returns (dish rows, recipe rows)."""
    by_desc = base.drop_duplicates("description").set_index("description")
    rows, recipe = [], []
    for i, (name, _arabic, details, _slots, ingredients) in enumerate(DISHES):
        fdc_id = DISH_ID_START + i
        missing = [d for d, _ in ingredients if d not in by_desc.index]
        if missing:
            raise KeyError(f"{name}: ingredient(s) not in USDA: {missing}")
        ing = by_desc.loc[[d for d, _ in ingredients]].reset_index()
        grams = pd.Series([g for _, g in ingredients], index=ing.index, dtype=float)
        total = grams.sum()
        per100 = {c: float((ing[c] * grams).sum() / total) for c in NUTRIENT_COLS + list(micro_cols)}
        description = dish_description(name, details)
        allergens = set(tag_allergens(description))
        flags = []
        for d, c in zip(ing["description"], ing["food_category"]):
            allergens |= set(tag_allergens(d, c))
            flags.append(diet_flags(d, c))
        estimated = sorted({e for est in ing["estimated"].fillna("") for e in est.split("|") if e})
        rows.append({"fdc_id": fdc_id, "description": description, "food_category": CATEGORY, "meal_role": "dish",
                     "base_name": name.lower(), **per100, "estimated": "|".join(estimated),
                     "portion_desc": "1 serving", "portion_g": total, "allergens": "|".join(sorted(allergens)),
                     "is_allergen_free": not allergens, "is_vegetarian": all(f[0] for f in flags),
                     "contains_pork": any(f[1] for f in flags), "contains_alcohol": any(f[2] for f in flags)})
        recipe += [{"dish_fdc_id": fdc_id, "dish": name, "fdc_id": int(r.fdc_id), "description": r.description,
                    "label": LABELS.get(r.description, r.description.split(",")[0].lower()),
                    "food_category": r.food_category, "grams": float(g), "serving_g": float(total)}
                   for r, g in zip(ing.itertuples(), grams)]
    return pd.DataFrame(rows), pd.DataFrame(recipe)
