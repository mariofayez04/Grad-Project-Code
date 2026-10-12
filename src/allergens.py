"""Deterministic allergen and diet tagging.

This is the safety net for allergies: it never relies on an ML model.
Design rule: over-flagging is acceptable, missing an allergen is not.
Used by clean_data.py (Phase 1) and by the preference parser (Phase 3).
"""
import re

# The 9 major allergen groups, plus `gluten` (wheat + barley/rye/malt).
ALLERGEN_TRIGGERS = {
    "milk": [
        "milk", "milkfat", "buttermilk", "cheese", "cheeseburger", "butter", "buttered", "cream", "creamed",
        "yogurt", "yoghurt", "whey", "casein", "caseinate", "lactose", "ghee", "curd", "custard", "kefir",
        "paneer", "ricotta", "mozzarella", "parmesan", "cheddar", "feta", "pudding", "eggnog", "alfredo", "queso",
        "chocolate", "fudge", "caramel", "toffee", "latte", "cappuccino", "mocha",
    ],
    "egg": [
        "egg", "eggnog", "mayonnaise", "mayo", "meringue", "albumen", "albumin", "omelet", "omelette", "custard",
    ],
    "peanut": ["peanut", "groundnut", "arachis"],
    "tree_nut": [
        "nut", "almond", "walnut", "cashew", "pistachio", "pecan", "hazelnut", "filbert", "macadamia",
        "pine nut", "pinyon", "chestnut", "brazilnut", "praline", "marzipan", "nougat", "pesto", "nutella",
    ],
    "shellfish": [
        "shellfish", "shrimp", "prawn", "crab", "lobster", "crayfish", "crawfish", "clam", "mussel", "oyster",
        "scallop", "squid", "calamari", "octopus", "abalone", "conch", "whelk", "cuttlefish",
    ],
    "fish": [
        "fish", "salmon", "tuna", "cod", "tilapia", "sardine", "anchovy", "anchovies", "mackerel", "trout",
        "herring", "pollock", "haddock", "halibut", "catfish", "bass", "perch", "pike", "carp", "whiting",
        "flounder", "sole", "snapper", "grouper", "mahimahi", "swordfish", "roe", "caviar", "surimi", "bluefish",
        "mullet", "sturgeon", "whitefish", "smelt", "pompano", "croaker", "eel", "shad", "roughy", "burbot",
    ],
    "soy": ["soy", "soya", "soybean", "soymilk", "tofu", "tempeh", "miso", "edamame", "natto", "okara"],
    "wheat": [
        "wheat", "flour", "bread", "breaded", "breading", "breadcrumbs", "breadsticks", "frybread", "cornbread",
        "flatbread", "shortbread", "gingerbread", "pasta", "spaghetti", "macaroni", "lasagna", "noodle", "couscous",
        "semolina", "bulgur", "durum", "farina", "seitan", "spelt", "cracker", "bagel", "biscuit", "muffin",
        "croissant", "pretzel", "pita", "tortilla", "cake", "cupcake", "cookie", "pie", "pancake", "waffle",
        "doughnut", "donut", "pastry", "strudel", "bun", "crouton", "stuffing", "dumpling", "gnocchi",
        "ravioli", "tortellini", "pizza", "matzo", "matzoh", "wonton", "graham", "batter", "battered", "pierogi",
        "brownie", "burrito", "sandwich", "corn dog", "corndog", "empanada", "calzone", "turnover", "knish",
    ],
    "sesame": ["sesame", "tahini", "tahina", "halva", "halvah", "hummus"],
}

# Gluten = every wheat trigger plus other gluten grains.
GLUTEN_EXTRA = ["barley", "rye", "malt", "malted", "triticale", "pumpernickel"]

# Phrases removed before matching a group, so they don't cause false positives.
# Only clearly safe cases are listed; when in doubt, keep the flag.
NOT_TRIGGERS = {
    "milk": [
        r"(peanut|almond|cashew|sesame|sunflower(\s+seed)?|apple|cocoa|nut|seed|soy|hazelnut|pistachio|macadamia)\s+butter",
        r"(coconut|almond|soy|rice|oat|cashew|hemp)\s+(milk|cream|beverage)",
        r"cream\s+of\s+(tartar|wheat|rice)", r"milk[- ]free", r"dairy[- ]free", r"butter\s*beans?",
        r"peanut butter", r"milkfish",
    ],
    "egg": [r"egg[- ]free", r"egg\s*plant"],
    "tree_nut": [r"water\s*chestnuts?", r"nutmeg", r"coconut", r"nut[- ]free", r"butternut", r"doughnuts?", r"peanuts?"],
    "wheat": [r"buckwheat", r"rice\s+(noodles?|cakes?|crackers?|flour)", r"corn\s+(flour|cakes?)", r"cracker\s*jack"],
    "fish": [r"fish[- ]free", r"cuttlefish", r"shellfish", r"crayfish", r"goldfish"],
}

# Sweet baked goods usually contain milk and egg even when the name doesn't say so (incl. gluten-free ones).
BAKED_SWEET_RE = re.compile(
    r"\b(cookies?|cakes?|cupcakes?|pancakes?|waffles?|muffins?|pastr(y|ies)|doughnuts?|donuts?|pies?|croissants?|"
    r"brownies?|danish|sweet rolls?|cinnamon rolls?|french toast|crepes?|eclairs?|cream puffs?)\b", re.I)

# Whole descriptions that are explicitly wheat-free (e.g. "Pasta, gluten-free, corn" or corn tortillas).
WHEAT_FREE_RE = re.compile(r"gluten[- ]free|wheat[- ]free|^(tortillas?|taco shells?|tostada shells?)\b.*\bcorn\b|\bcorn tortillas?\b", re.I)

MEAT_WORDS = [
    "beef", "veal", "pork", "ham", "bacon", "lamb", "mutton", "goat", "chicken", "turkey", "duck", "goose",
    "quail", "pheasant", "squab", "meat", "meatball", "meatloaf", "sausage", "frankfurter", "hotdog", "bologna",
    "salami", "pepperoni", "pastrami", "prosciutto", "chorizo", "gelatin", "lard", "broth", "bouillon", "jerky",
    "hamburger", "cheeseburger", "burger", "steak", "liver",
]
PORK_WORDS = ["pork", "ham", "bacon", "lard", "prosciutto", "salami", "pepperoni", "chorizo", "pancetta", "chitterlings", "frankfurter", "bologna"]
ALCOHOL_WORDS = ["alcohol", "alcoholic", "beer", "wine", "liqueur", "rum", "vodka", "whiskey", "gin", "sake", "brandy"]
MEAT_CATEGORIES = {
    "Beef Products", "Pork Products", "Poultry Products", "Lamb, Veal, and Game Products",
    "Sausages and Luncheon Meats", "Finfish and Shellfish Products",
}


def _words_regex(words):
    """Whole-word match, allowing simple plurals (egg -> eggs, peach -> peaches)."""
    alts = "|".join(sorted((re.escape(w).replace(r"\ ", r"\s+") for w in words), key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})(?:s|es)?\b", re.IGNORECASE)


_TRIGGER_RE = {g: _words_regex(w) for g, w in ALLERGEN_TRIGGERS.items()}
_TRIGGER_RE["gluten"] = _words_regex(ALLERGEN_TRIGGERS["wheat"] + GLUTEN_EXTRA)
_NOT_RE = {g: re.compile("|".join(p), re.IGNORECASE) for g, p in NOT_TRIGGERS.items()}
_NOT_RE["gluten"] = _NOT_RE["wheat"]
_MEAT_RE, _PORK_RE, _ALCOHOL_RE = _words_regex(MEAT_WORDS), _words_regex(PORK_WORDS), _words_regex(ALCOHOL_WORDS)
_NOT_MEAT_RE = re.compile(r"(coconut|nut)\s+meat|meatless|vegetable\s+(broth|bouillon)|vegetarian|ham\s*burger\s+bun", re.I)

ALLERGEN_GROUPS = list(ALLERGEN_TRIGGERS) + ["gluten"]


def tag_allergens(text, category=None):
    """Return the sorted list of allergen groups found in a food description (or any text)."""
    tags = set()
    for group, rx in _TRIGGER_RE.items():
        cleaned = _NOT_RE[group].sub(" ", text) if group in _NOT_RE else text
        if rx.search(cleaned):
            tags.add(group)

    if category == "Finfish and Shellfish Products" and not tags & {"fish", "shellfish"}:
        tags.add("fish")
    if category == "Dairy and Egg Products" and not tags & {"milk", "egg"}:
        tags.add("milk")
    if category in ("Baked Products", "Breakfast Cereals") and not _NOT_RE["wheat"].search(text):
        tags.add("wheat")
    if category in ("Baked Products", "Sweets", "Snacks") and BAKED_SWEET_RE.search(text):
        tags |= {"milk", "egg"}

    if WHEAT_FREE_RE.search(text):
        tags -= {"wheat", "gluten"}
    elif "wheat" in tags:
        tags.add("gluten")
    return sorted(tags)


def diet_flags(text, category=None):
    """Return (is_vegetarian, contains_pork, contains_alcohol) for a description."""
    has_meat = category in MEAT_CATEGORIES or bool(_MEAT_RE.search(_NOT_MEAT_RE.sub(" ", text)))
    pork = category == "Pork Products" or bool(_PORK_RE.search(text))
    alcohol = bool(_ALCOHOL_RE.search(text))
    return (not has_meat, pork, alcohol)
