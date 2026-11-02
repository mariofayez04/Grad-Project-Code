"""Phase 5: build a 7-day meal plan that hits the user's calorie and macro targets.

SR Legacy foods are ingredients, not meals, so every meal is a combination of 2-4 foods
(e.g. protein + carb + vegetable + optional cooking oil) with gram amounts solved to hit that meal's targets.

A food the user doesn't want can be swapped afterwards (swap_options): the replacement fills the same meal
component, is close in embedding and macro profile, and gets the grams that keep the meal's calories.

For each meal slot:
 1. Candidates = foods allowed by the parser (allergies / diet already removed) that fit the slot
    (e.g. breakfast carb = bread / oats / cereal, not pasta).
 2. Each candidate gets a score:
        score = W_FIT * role_fit + W_TASTE * cosine(food, taste_vector) + W_PREF * pref_score
                + W_NUT * nutrient_score - variety_penalty
    role_fit:      how well the food does its job (protein share for a protein, low kcal for a vegetable...)
    taste_vector:  embedding of the user's liked foods (updated weekly in Phase 7)
    pref_score:    +1 per liked word, -1 per disliked word (from the parser)
    nutrient_score: bonus for fiber / calcium / iron / potassium, penalty for too much sodium / saturated fat,
                   stronger in the diet modes (Feature #10, src/nutrients.py)
    variety:       foods already used this week are penalised, same base food twice in a day is not allowed
 3. The best few candidates per component are combined; for every combination the grams are solved with
    bounded least squares (scipy lsq_linear) and the combination with the best fit + score wins.
 4. Each slot's target is taken from what is still left of the day, so the last meal corrects earlier errors.

Egyptian dishes (Feature #9): a meal can also be one dish (koshari, molokhia...) + optional sides (bread / rice,
salad, extra protein) - DISH_TEMPLATES. For each meal both kinds are tried and the better one is kept; the
"Egyptian dishes" setting (often / sometimes / never) adds DISH_BONUS to the dish kind or switches it off.

One meal or one day can be planned again later (replan_meal / replan_day): the rest of the week counts for
variety, and the current foods are left out so the alternative is really different.

Run a demo:  python -m src.planner
"""
import itertools
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from src.calculator import DEFAULT_MEALS, meal_split
from src.dishes import dishes_for
from src.config import FOODS_CLEAN
from src.embeddings import embed_text, encode

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MACROS = ["calories", "protein_g", "carbs_g", "fat_g"]

W_FIT, W_TASTE, W_PREF = 1.0, 1.0, 1.0
W_NUT = 1.0                # weight of the nutrient score (Feature #10)
W_CAP = 20.0               # penalty per "cap" of excess when a meal would push the day over a nutrient cap
CAP_MARGIN = 0.95          # plan to 95% of each cap: portions are rounded to 5 g afterwards
CAP_SLACK = 0.35           # a meal may use its share of a daily cap + 35% of it (so breakfast can't use it all)
W_ERROR = 4.0              # weight of the meal's macro error when comparing combinations
TOP_K = 5                  # candidates kept per component before trying combinations
REPEAT_PENALTY = 0.6       # per earlier use of the same food this week
BASE_REPEAT_PENALTY = 0.25  # per earlier use of the same base food (e.g. "chicken") this week
NOISE = 0.15               # randomness so two plans are not identical
PORTION_BASE_KCAL = 2400   # daily target above which maximum portions start to grow
SWAP_LIST_SIZE = 30        # ranked replacements kept for 'Swap a food' (shown 3 at a time)
SEARCH_WORD_BONUS = 0.3    # typed search: bonus when every typed word is in the food name

# Food names whose first word is a family ("Beans, kidney"): the title reads "Kidney beans".
NAME_FIRST_SECOND = {"oil", "beans", "rice", "cheese", "yogurt", "milk", "peppers", "squash", "melons", "peas", "pasta"}
NAME_SECOND_ONLY = {"nuts", "seeds", "cereals", "cereals ready-to-eat"}

DESSERT_RE = r"cake|pie\b|cookie|doughnut|danish|pastr|brownie|sweet roll|frosting|icing|crust|cobbler|eclair|cream puff|strudel|tart\b"
NOT_A_SIDE_RE = (r"chips|fries|french fried|hash brown|stuffing|crouton|cracker|pretzel|puffs|flour|starch|syrup|sauce|"
                 r"cornmeal|corn grain|\bdry\b|dry mix|\bmix\b|au gratin|\braw\b|bran\b|germ\b|sprouted")

# Common, everyday foods (with Egyptian staples) so plans look like real meals, not a botany list.
COMMON_FRUIT = (r"^(apples?|bananas?|oranges?|grapes?|strawberries|watermelon|melons, cantaloupe|mangos?|peach(es)?|pears?|guavas?|"
                r"pomegranates?|dates?|figs?|apricots?|kiwifruit|pineapple|plums?|cherries|tangerines?|grapefruit|blueberries|"
                r"raspberries|blackberries|nectarines?|clementines?)\b")
COMMON_VEG = (r"^(broccoli|carrots?|spinach|squash, summer|beans, snap|peas, green|tomatoes|cucumber|lettuce|cabbage|cauliflower|"
              r"peppers, sweet|eggplant|okra|mixed vegetables|vegetables, mixed|jute, potherb|asparagus|beets|kale|artichokes?|"
              r"brussels sprouts|mushrooms|celery|zucchini|green beans|peas and carrots|leeks|chard|collards)\b")
WEIRD_PROTEIN_RE = (r"New Zealand|imported|headcheese|souse|scrapple|loaf|pate|spread|roll\b|canned.*(pork|beef)|meatless|"
                    r"cured|water added|unheated|dried|powder|\bmixture\b|substitute|whole, raw|ground, raw")


@dataclass
class Component:
    name: str
    roles: tuple
    include: str = None      # regex the description must match
    exclude: str = None      # regex the description must not match
    grams: tuple = (30, 300)
    top_k: int = TOP_K       # candidates tried in the combination search


MAIN_PROTEIN = Component(
    "protein", ("protein",),
    include=r"^(chicken|turkey|beef|lamb|veal|pork|fish|salmon|tuna|tilapia|cod|sardines?|mackerel|trout|crustaceans|mollusks|"
            r"egg|tofu|tempeh|goat)",
    exclude=r"breaded|batter|fried|sausage|frankfurter|bologna|salami|luncheon|bacon|with skin|\bskin\b(?!less)|giblets|neck|"
            r"back\b|wing|smoked|caviar|roe|fermented|" + WEIRD_PROTEIN_RE,
    grams=(80, 300))
MAIN_CARB = Component(
    "carb", ("carb", "legume"),
    include=r"^(rice|pasta|spaghetti|macaroni|noodles|potatoes|sweet potato|couscous|bulgur|quinoa|bread|pita|beans|lentils|"
            r"chickpeas|corn|barley|freekeh)\b",
    exclude=DESSERT_RE + "|" + NOT_A_SIDE_RE + r"|baked beans|refried|cereal|salad|scalloped|rice cake|crumbs|with pork|"
                                              r"chow mein|fried|canned|flakes|pudding|fritter|candied|glazed|skin\b|leaves",
    grams=(60, 350))
VEGETABLE = Component("vegetable", ("vegetable",), include=COMMON_VEG,
                      exclude=r"pickled|dehydrated|juice|sauce|cream|soup|frozen.*butter|canned.*(cream|sauce)|onion rings|batter|"
                              r"fried|kimchi|sauerkraut|dried",
                      grams=(80, 250))
FRUIT = Component("fruit", ("fruit",), include=COMMON_FRUIT, exclude=r"canned|syrup|pie filling|candied|juice|frozen.*sweetened",
                  grams=(80, 250))
# Cooking fat: lean meals need it to reach the fat target. Lower bound 0 = "no oil needed".
COOKING_FAT = Component("fat", ("fat",),
                        include=r"^(oil, (olive|sunflower|canola|corn|soybean|peanut|sesame)|butter, (salted|without salt)|"
                                r"butter oil|ghee)",
                        exclude=r"industrial|hydrogenated|whipped|light|blend|high oleic", grams=(0, 30), top_k=2)

SLOTS = {
    "breakfast": [
        Component("main", ("protein", "dairy", "legume"),
                  include=r"^(egg|eggs|yogurt|cheese|milk|beans, (fava|broad|black|kidney|white)|broadbeans|hummus|chickpeas)\b",
                  exclude=r"sauce|spread|product|food|powder|dessert|ice cream|pudding|shake|eggnog|substitute|dried|frozen|"
                          r"imitation|goat|sheep|buffalo|chocolate|human|canned|condensed|evaporated|dry|filled|raw|"
                          r"parmesan|romano|grated|pasteurized process|low sodium|substitute",
                  grams=(50, 300)),
        Component("carb", ("carb",),
                  include=r"^(bread|cereals, oats|oats|bagels?|english muffins|pita|tortillas|cereals ready-to-eat|granola|rolls)\b",
                  exclude=DESSERT_RE + "|" + NOT_A_SIDE_RE + r"|french toast|pancake|waffle|hamburger|hot dog|stuffing|"
                                                              r"crumbs|frosted|sugar|chocolate|marshmallow",
                  grams=(30, 200)),
        FRUIT,
    ],
    "lunch": [MAIN_PROTEIN, MAIN_CARB, VEGETABLE, COOKING_FAT],
    "dinner": [MAIN_PROTEIN, MAIN_CARB, VEGETABLE, COOKING_FAT],
    "snack": [
        FRUIT,
        Component("extra", ("dairy", "fat", "legume"),
                  include=r"^(yogurt|cheese, cottage|milk|nuts, (almonds|walnuts|pistachio|cashew|hazelnuts|pecans|mixed)|"
                          r"peanuts|peanut butter|seeds, (sunflower|pumpkin))",
                  exclude=r"oil|cream|ice cream|pudding|dessert|shake|powder|meal|flour|paste|chocolate|honey|sweetened|"
                          r"goat|sheep|buffalo|human|canned|condensed|evaporated|dry\b|dried|imitation|filled|with vegetables",
                  grams=(15, 250)),
    ],
}
SLOTS["morning_snack"] = SLOTS["snack"]


def _dish(slot, grams):
    names = "|".join(re.escape(n) for n in dishes_for(slot))
    return Component("dish", ("dish",), include=rf"^(?:{names}),", grams=grams, top_k=4)


# Optional sides of a dish (lower bound 0 g = "not needed"): the solver adds them only to reach the targets.
SIDE_CARB = Component("side_carb", ("carb", "legume"), include=MAIN_CARB.include, exclude=MAIN_CARB.exclude,
                      grams=(0, 250), top_k=2)
SIDE_VEG = Component("side_veg", ("vegetable",), include=COMMON_VEG, exclude=VEGETABLE.exclude, grams=(0, 200), top_k=2)
SIDE_PROTEIN = Component("side_protein", ("protein", "dairy"),
                         include=r"^(?:chicken|turkey|fish|egg|yogurt, (?:plain|greek)|cheese, cottage)",
                         exclude=MAIN_PROTEIN.exclude + r"|sweetened|vanilla|fruit|chocolate|with vegetables",
                         grams=(0, 200), top_k=2)
SIDE_BREAD = Component("side_bread", ("carb",), include=r"^(?:bread, pita|bread, whole-wheat|bread, white|rolls)",
                       exclude=DESSERT_RE + "|" + NOT_A_SIDE_RE, grams=(0, 120), top_k=2)
SIDE_FRUIT = Component("side_fruit", ("fruit",), include=COMMON_FRUIT, exclude=FRUIT.exclude, grams=(0, 200), top_k=2)
DISH_TEMPLATES = {
    "breakfast": [_dish("breakfast", (120, 350)), SIDE_BREAD, SIDE_FRUIT],  # room left for baladi bread
    "lunch": [_dish("lunch", (150, 650)), SIDE_CARB, SIDE_VEG, SIDE_PROTEIN],
    "dinner": [_dish("dinner", (150, 650)), SIDE_CARB, SIDE_VEG, SIDE_PROTEIN],
    "snack": [_dish("snack", (120, 300)), SIDE_FRUIT],
    "morning_snack": [_dish("morning_snack", (120, 300)), SIDE_FRUIT],
}
EGYPTIAN_OPTIONS = {"often": "Often", "sometimes": "Sometimes", "never": "Never"}
# Added to a dish meal's value; "never" = no dish meals. Dish meals are flexible (optional sides), so without a
# handicap they win most meals (~16 of 28 a week). Tuned: -0.7 -> ~7 dish meals a week, -0.3 -> ~12.
DISH_BONUS = {"often": -0.3, "sometimes": -0.7}
SLOT_ORDER = ["breakfast", "morning_snack", "lunch", "snack", "dinner"]  # every slot, in time order
SLOT_ICONS = {"breakfast": "🍳", "morning_snack": "🍌", "lunch": "🍲", "snack": "🍎", "dinner": "🍽️"}


def slot_label(slot, meals_per_day=DEFAULT_MEALS):
    """'morning_snack' -> 'Morning snack'; with 5 meals the other snack is the afternoon one."""
    if slot == "snack" and int(meals_per_day or DEFAULT_MEALS) == 5:
        return "Afternoon snack"
    return slot.replace("_", " ").capitalize()


def plan_slots(items):
    """Slots present in saved plan items, in time order."""
    present = set(items["slot"])
    return [s for s in SLOT_ORDER if s in present] + sorted(present - set(SLOT_ORDER))


# Calorie-dense foods get smaller portions (only for these roles, so "beans, immature seeds" is not affected).
DENSE_GRAMS = [
    (("dairy", "protein"), r"^cheese(?!, cottage)", (20, 60)),
    (("fat", "legume"), r"^(nuts|peanuts|peanut butter|seeds)", (15, 40)),
]


# --- result objects ----------------------------------------------------------------------------------
@dataclass
class MealItem:
    fdc_id: int
    name: str
    grams: float
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


@dataclass
class Meal:
    slot: str
    items: list
    target: dict

    def totals(self):
        return {m: round(sum(getattr(i, m) for i in self.items), 1) for m in MACROS}


@dataclass
class DayPlan:
    day: str
    meals: list = field(default_factory=list)

    def totals(self):
        return {m: round(sum(meal.totals()[m] for meal in self.meals), 1) for m in MACROS}


@dataclass
class WeekPlan:
    days: list
    targets: dict

    def to_dataframe(self):
        """One row per food item - the format stored in SQLite and shown in the web app."""
        rows = []
        for d_idx, day in enumerate(self.days):
            for meal in day.meals:
                for item in meal.items:
                    rows.append({"day_index": d_idx, "day": day.day, "slot": meal.slot, **asdict(item)})
        return pd.DataFrame(rows)

    def adherence(self):
        """Mean absolute % error of daily totals vs targets, per macro."""
        errs = {m: [] for m in MACROS}
        for day in self.days:
            t = day.totals()
            for m in MACROS:
                errs[m].append(abs(t[m] - self.targets[m]) / self.targets[m] * 100)
        return {m: round(float(np.mean(v)), 1) for m, v in errs.items()}


# --- helpers -----------------------------------------------------------------------------------------
def display_name(description):
    return embed_text(description)


def split_name(name):
    """'Beans, kidney, cooked, boiled' -> ('Kidney beans', 'cooked, boiled'): a short title and the details."""
    parts = [p.strip() for p in name.split(",") if p.strip()]
    first = parts[0]
    if len(parts) > 1 and first.lower() in NAME_SECOND_ONLY:
        return parts[1][:1].upper() + parts[1][1:], ", ".join(parts[2:])
    if len(parts) > 1 and first.lower() in NAME_FIRST_SECOND:
        return f"{parts[1][:1].upper() + parts[1][1:]} {first.lower()}", ", ".join(parts[2:])
    return first, ", ".join(parts[1:])


def taste_from_preferences(prefs):
    """Initial taste vector: embedding of liked foods minus half the disliked ones (None if nothing stated)."""
    if not prefs.prefer_terms and not prefs.avoid_terms:
        return None
    v = np.zeros(384, dtype=np.float32)
    if prefs.prefer_terms:
        v += encode(sorted(prefs.prefer_terms)).mean(axis=0)
    if prefs.avoid_terms:
        v -= 0.5 * encode(sorted(prefs.avoid_terms)).mean(axis=0)
    return v / (np.linalg.norm(v) or 1.0)


def role_fit(df, component):
    """0..1: how well each food does its job in the meal."""
    kcal = df["calories"].clip(lower=1)
    protein_share = (4 * df["protein_g"] / kcal).clip(0, 1)
    carb_share = (4 * df["carbs_g"] / kcal).clip(0, 1)
    if component.name in ("protein", "main", "side_protein"):
        return protein_share
    if component.name in ("carb", "side_carb", "side_bread"):
        return carb_share * 0.7 + (df["protein_g"] / 20).clip(0, 0.3)  # slight bonus for whole grains / legumes
    if component.name in ("vegetable", "side_veg"):
        return (1 - df["calories"] / 120).clip(0, 1)
    if component.name == "extra":
        return protein_share.clip(0, 0.5) + 0.5
    return pd.Series(0.5, index=df.index)


def grams_bounds(description, role, component):
    for roles, pattern, bounds in DENSE_GRAMS:
        if role in roles and re.search(pattern, description, re.I):
            return bounds
    return component.grams


def component_foods(df, component):
    """Foods that can fill a meal component: right meal role and passing its include / exclude rules."""
    df = df[df["meal_role"].isin(component.roles)]
    if component.include:
        df = df[df["description"].str.contains(component.include, case=False, regex=True)]
    if component.exclude:
        df = df[~df["description"].str.contains(component.exclude, case=False, regex=True)]
    return df


def component_of(food_row, slot):
    """The component of a meal slot that a food fills (None if it fits none, e.g. an old plan)."""
    one = food_row.to_frame().T
    comps = SLOTS[slot] + DISH_TEMPLATES.get(slot, [])
    return next((c for c in comps if not component_foods(one, c).empty), None)


def solve_grams(per100, lo, hi, target):
    """Bounded least squares: grams of each food so the meal's kcal/protein/carbs/fat are close to target.
    per100: n x 4 array (kcal, protein, carbs, fat per 100 g)."""
    a = per100.T / 100.0                                        # 4 x n, per gram
    b = np.array([target[m] for m in MACROS], dtype=float)
    scale = np.array([2.0, 1.5, 1.0, 1.0]) / np.maximum(b, 1)   # relative error, calories matter most
    res = lsq_linear(a * scale[:, None], b * scale, bounds=(lo, hi + 1e-9), method="bvls")  # bvls: ~10x faster for tiny problems
    hi5 = np.maximum(np.floor(hi / 5) * 5, lo)                  # a maximum portion is a multiple of 5 g too
    grams = np.clip(np.round(res.x / 5) * 5, lo, hi5)            # round to 5 g
    error = float(np.sqrt(np.mean(((a @ grams - b) * scale) ** 2)))
    return grams, error


class MealPlanner:
    def __init__(self, allowed_foods, embeddings=None, taste_vector=None, food_penalties=None, seed=None,
                 nutrition=None):
        """
        allowed_foods:  output of parser.apply_to_foods (blocked foods already removed, has `pref_score`)
        embeddings:     FoodEmbeddings (optional; enables the taste term)
        taste_vector:   384-d user taste (Phase 7), or None
        food_penalties: {fdc_id: penalty} from feedback (Phase 7), e.g. skipped foods
        nutrition:      the user's NutritionTargets (or saved targets dict): enables the nutrient score and the
                        diet-mode food filters (Feature #10). None = macros only, as before.
        """
        df = allowed_foods.copy()
        self.nut_cols, self.nut_targets, self.encouraged, self.per100, self.caps = [], {}, set(), {}, {}
        if nutrition is not None:
            from src.nutrients import MICROS, filter_modes, nutrient_parts, weights
            get = nutrition.get if isinstance(nutrition, dict) else lambda k, d=None: getattr(nutrition, k, d)
            modes, micros = get("health_modes", ""), get("micros", None)
            df = filter_modes(df, modes)
            if micros and set(MICROS) <= set(df.columns):
                parts = nutrient_parts(df, micros, get("calories"), modes)
                for n in parts:
                    df[f"nut_{n}"] = parts[n]
                self.nut_cols, self.nut_targets = list(parts), micros
                self.encouraged = set(weights(modes)[0]) & set(parts)
                self.per100 = dict(zip(df["fdc_id"], df[MICROS].to_dict("records")))
                self.caps = {n: t["upper"] for n, t in micros.items() if t.get("upper")}  # daily caps
        if "pref_score" not in df:
            df["pref_score"] = 0.0
        df["taste"] = 0.0
        if embeddings is not None and taste_vector is not None:
            sims = embeddings.vectors @ taste_vector
            df["taste"] = df["fdc_id"].map(dict(zip(embeddings.ids, sims))).fillna(0.0)
        df["penalty"] = df["fdc_id"].map(food_penalties or {}).fillna(0.0)
        self.foods = df
        self.base_of = dict(zip(df["fdc_id"], df["base_name"]))
        self.rng = np.random.default_rng(seed)
        self.portion_scale = 1.0
        setting = "sometimes"
        if nutrition is not None:
            setting = (nutrition.get("egyptian_dishes") if isinstance(nutrition, dict)
                       else getattr(nutrition, "egyptian_dishes", None)) or "sometimes"
        self.dish_bonus = DISH_BONUS.get(setting)  # None = no dish meals
        self.pools = {(slot, c.name): self._pool(c) for slot, comps in SLOTS.items() for c in comps}
        if self.dish_bonus is not None:
            self.pools.update({(slot, c.name): self._pool(c) for slot, comps in DISH_TEMPLATES.items() for c in comps})

    def _pool(self, component):
        df = component_foods(self.foods, component).copy()
        bounds = [grams_bounds(d, r, component) for d, r in zip(df["description"], df["meal_role"])]
        df["_lo"] = [b[0] for b in bounds]
        df["_hi"] = [b[1] for b in bounds]
        df["_score"] = W_FIT * role_fit(df, component) + W_TASTE * df["taste"] + W_PREF * df["pref_score"] - df["penalty"]
        return df

    def _candidates(self, slot, component, used_ids, used_bases_week, used_bases_today, exclude_ids=frozenset(),
                    pace=None):
        pool = self.pools[(slot, component.name)]
        if component.name != "fat":  # cooking oil at lunch and dinner is fine
            pool = pool[~pool["base_name"].isin(used_bases_today)]
        if exclude_ids:
            pool = pool[~pool["fdc_id"].isin(exclude_ids)]
        if pool.empty:
            return pool
        score = pool["_score"].to_numpy().copy()
        for n in self.nut_cols:  # nutrient score, without the nutrients the day already has enough of
            if (pace or {}).get(n, 1.0):
                score += W_NUT * pool[f"nut_{n}"].to_numpy()
        score -= REPEAT_PENALTY * pool["fdc_id"].map(used_ids).fillna(0).to_numpy()
        score -= BASE_REPEAT_PENALTY * pool["base_name"].map(used_bases_week).fillna(0).to_numpy()
        score += self.rng.gumbel(0, NOISE, len(score))
        top = np.argsort(-score)[:component.top_k]
        return pool.iloc[top].assign(_final=score[top])

    def plan_meal(self, slot, target, used_ids, used_bases_week, used_bases_today, exclude_ids=frozenset(), pace=None,
                  allowance=None):
        """allowance: {nutrient: amount the day can still take} - combinations going over get W_CAP penalty.
        Tries the usual meal (SLOTS) and, unless switched off, an Egyptian dish meal (DISH_TEMPLATES)."""
        templates = [(SLOTS[slot], 0.0)]
        if self.dish_bonus is not None and slot in DISH_TEMPLATES:
            templates.append((DISH_TEMPLATES[slot], self.dish_bonus))
        best = None
        for components, bonus in templates:
            found = self._best_combo(slot, components, target, used_ids, used_bases_week, used_bases_today,
                                     exclude_ids, pace, allowance, bonus)
            if found and (best is None or found[0] > best[0]):
                best = found
        if best is None:
            return Meal(slot, [], target)
        _, options, combo, grams = best

        items = []
        for (i, j), g in zip(enumerate(combo), grams):
            if g <= 0:  # optional component not needed (e.g. no cooking oil, no side)
                continue
            r = options[i].iloc[j]
            f = g / 100.0
            items.append(MealItem(int(r["fdc_id"]), display_name(r["description"]), float(g),
                                  round(r["calories"] * f, 1), round(r["protein_g"] * f, 1),
                                  round(r["carbs_g"] * f, 1), round(r["fat_g"] * f, 1)))
        return Meal(slot, items, target)

    def _best_combo(self, slot, components, target, used_ids, used_bases_week, used_bases_today, exclude_ids, pace,
                    allowance, bonus):
        """Best (value, options, combo, grams) for one meal template, or None."""
        caps = [n for n in (allowance or {}) if n in self.caps]
        options = []
        for comp in components:
            cands = self._candidates(slot, comp, used_ids, used_bases_week, used_bases_today, exclude_ids, pace)
            if not cands.empty:
                options.append(cands)
            elif comp.name == "dish":  # no dish allowed today (allergies, already eaten...)
                return None
        if not options:
            return None

        arrays = [(o[MACROS].to_numpy(float), o["_lo"].to_numpy(float), o["_hi"].to_numpy(float) * self.portion_scale,
                   o["_final"].to_numpy(float), o["base_name"].to_numpy(), o[caps].to_numpy(float)) for o in options]
        best = None
        for combo in itertools.product(*[range(len(o)) for o in options]):
            bases = [arrays[i][4][j] for i, j in enumerate(combo)]
            if len(set(bases)) < len(bases):
                continue
            per100 = np.array([arrays[i][0][j] for i, j in enumerate(combo)])
            lo = np.array([arrays[i][1][j] for i, j in enumerate(combo)])
            hi = np.array([arrays[i][2][j] for i, j in enumerate(combo)])
            grams, error = solve_grams(per100, lo, hi, target)
            value = np.mean([arrays[i][3][j] for i, j in enumerate(combo)]) - W_ERROR * error
            if caps:  # nutrient caps: how far this meal would push the day over each cap
                amounts = np.array([arrays[i][5][j] for i, j in enumerate(combo)]).T @ grams / 100
                value -= W_CAP * sum(max(0.0, a - allowance[n]) / self.caps[n] for n, a in zip(caps, amounts))
            if best is None or value > best[0]:
                best = (value, combo, grams)
        if best is None:
            return None
        return best[0] + bonus, options, best[1], best[2]

    def _set_portion_scale(self, daily):
        # Bigger eaters get bigger maximum portions (up to +60%) instead of an impossible target.
        self.portion_scale = float(np.clip(daily["calories"] / PORTION_BASE_KCAL, 1.0, 1.6))

    def _use(self, items, used_ids, used_bases_week, used_today=None):
        """Count foods as used (for the variety penalties)."""
        for fid in items:
            used_ids[fid] = used_ids.get(fid, 0) + 1
            base = self.base_of.get(fid)
            if base is not None:
                used_bases_week[base] = used_bases_week.get(base, 0) + 1
                if used_today is not None:
                    used_today.add(base)

    def _nutrients_of(self, fdc_ids, grams):
        """Running nutrient totals of some foods (for pacing and caps)."""
        got = {n: 0.0 for n in self.encouraged | set(self.caps)}
        for f, g in zip(fdc_ids, grams):
            for n in got:
                got[n] += self.per100.get(f, {}).get(n, 0.0) * g / 100
        return got

    def _allowance(self, got, share_by_end=1.0):
        """What this meal can still take of each capped nutrient: the cap paced over the day (its share so far
        + CAP_SLACK), minus what the day already has."""
        part = min(1.0, share_by_end + CAP_SLACK)
        return {n: CAP_MARGIN * cap * part - got.get(n, 0.0) for n, cap in self.caps.items()}

    def _pace(self, got, share_by_end):
        """Nutrient pacing: 0 (stop rewarding) for an encouraged nutrient the day already has enough of for the
        meals planned so far plus this one, 1 otherwise. Keeps fiber / iron near the target instead of far above."""
        return {n: 0.0 if got[n] >= self.nut_targets[n]["value"] * share_by_end else 1.0 for n in self.encouraged}

    def _plan_day(self, day_name, daily, used_ids, used_bases_week, exclude_ids=frozenset(), avoid_bases=(),
                  meals_per_day=DEFAULT_MEALS):
        """avoid_bases: base foods treated as already eaten today (not served again, except cooking fat)."""
        day = DayPlan(day_name)
        remaining = dict(daily)
        remaining_share, done_share = 1.0, 0.0
        used_today = set(avoid_bases)
        got = {n: 0.0 for n in self.encouraged | set(self.caps)}
        for slot, share in meal_split(meals_per_day).items():
            # This slot's share of whatever is still left of the day.
            target = {m: max(remaining[m] * share / remaining_share, 1.0) for m in MACROS}
            meal = self.plan_meal(slot, target, used_ids, used_bases_week, used_today, exclude_ids,
                                  self._pace(got, done_share + share), self._allowance(got, done_share + share))
            day.meals.append(meal)
            self._use([i.fdc_id for i in meal.items], used_ids, used_bases_week, used_today)
            for n, v in self._nutrients_of([i.fdc_id for i in meal.items], [i.grams for i in meal.items]).items():
                got[n] += v
            for m in MACROS:
                remaining[m] -= meal.totals()[m]
            remaining_share -= share
            done_share += share
        return day

    def plan_week(self, targets):
        """targets: calculator.NutritionTargets"""
        daily = daily_targets(targets)
        self._set_portion_scale(daily)
        used_ids, used_bases_week = {}, {}
        n = getattr(targets, "meals_per_day", DEFAULT_MEALS)
        days = [self._plan_day(day_name, daily, used_ids, used_bases_week, meals_per_day=n) for day_name in DAYS]
        return WeekPlan(days, daily)

    def replan_meal(self, items, day_index, slot, daily, meals_per_day=DEFAULT_MEALS):
        """A different meal for one slot of a saved week.
        items: the week's plan items (day_index, slot, fdc_id + macros); daily: daily targets dict.
        Target = the day's target minus the day's other meals, so the day total stays on target."""
        self._set_portion_scale(daily)
        this = (items["day_index"] == day_index) & (items["slot"] == slot)
        used_ids, used_bases_week = {}, {}
        self._use(items.loc[~this, "fdc_id"], used_ids, used_bases_week)
        same_day = items[(items["day_index"] == day_index) & ~this]
        current = items.loc[this, "fdc_id"].tolist()
        used_today = {self.base_of[f] for f in same_day["fdc_id"].tolist() + current if f in self.base_of}
        nominal = {m: daily[m] * meal_split(meals_per_day)[slot] for m in MACROS}
        target = {m: max(daily[m] - same_day[m].sum(), 0.5 * nominal[m], 1.0) for m in MACROS}
        got = self._nutrients_of(same_day["fdc_id"], same_day["grams"])  # the rest of the day counts
        return self.plan_meal(slot, target, used_ids, used_bases_week, used_today, frozenset(current),
                              self._pace(got, 1.0), self._allowance(got))

    def replan_day(self, items, day_index, daily, meals_per_day=DEFAULT_MEALS):
        """A different day: planned around the rest of the week, without that day's current foods
        (nor other versions of them, e.g. another cheese)."""
        self._set_portion_scale(daily)
        this = items["day_index"] == day_index
        used_ids, used_bases_week = {}, {}
        self._use(items.loc[~this, "fdc_id"], used_ids, used_bases_week)
        current = items.loc[this, "fdc_id"].tolist()
        bases = {self.base_of[f] for f in current if f in self.base_of}
        return self._plan_day(DAYS[day_index], daily, used_ids, used_bases_week, frozenset(current), bases, meals_per_day)


def daily_targets(targets):
    """calculator.NutritionTargets (or its saved dict) -> {calories, protein_g, carbs_g, fat_g}."""
    get = targets.get if isinstance(targets, dict) else lambda k: getattr(targets, k)
    return {m: get(m) for m in MACROS}


def suggest_substitutes(fdc_id, allowed_foods, embeddings, k=5, slot=None, exclude_bases=()):
    """Replacement for a rejected food: same meal role, close in meaning (embedding) and in macro profile.
    With `slot`, candidates must fill the same component of that meal (e.g. a lunch protein, not a sausage).
    `exclude_bases`: base foods already in the meal. k=None returns every candidate, best first."""
    foods = allowed_foods.reset_index(drop=True)
    row = embeddings.foods.loc[embeddings.foods["fdc_id"] == fdc_id].iloc[0]
    foods = foods[(foods["fdc_id"] != fdc_id) & (foods["base_name"] != row["base_name"]) & ~foods["base_name"].isin(exclude_bases)]
    component = component_of(row, slot) if slot else None
    cands = (component_foods(foods, component) if component else foods[foods["meal_role"] == row["meal_role"]]).copy()
    if cands.empty:
        return cands.assign(similarity=[])
    sem = embeddings.similarity(embeddings.vector(fdc_id), cands["fdc_id"].tolist())

    def profile(df):
        kcal = np.maximum(df["calories"].to_numpy(dtype=float), 1)[:, None]
        p = np.column_stack([4 * df["protein_g"], 4 * df["carbs_g"], 9 * df["fat_g"]]).astype(float) / kcal
        return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)

    macro = profile(cands) @ profile(row.to_frame().T)[0]
    cands["similarity"] = 0.6 * sem + 0.4 * macro
    cands = cands.sort_values("similarity", ascending=False)
    return cands if k is None else cands.head(k)


def _diverse(df, limit):
    """Keep the best food of each base food and drop foods whose shown name repeats (e.g. enriched /
    unenriched pasta both display as 'Pasta, cooked'), so every suggestion is a real choice."""
    names = df["description"].map(display_name)
    return df[~df["base_name"].duplicated() & ~names.duplicated()].head(limit)


def _sized_item(r, calories, component):
    """MealItem for food row `r` with the grams that give about `calories` kcal (within portion limits)."""
    lo, hi = grams_bounds(r.description, r.meal_role, component)
    g = float(np.clip(round(calories / max(r.calories, 1) * 100 / 5) * 5, max(lo, 5), hi * 1.6))
    f = g / 100.0
    return MealItem(int(r.fdc_id), display_name(r.description), g, round(r.calories * f, 1),
                    round(r.protein_g * f, 1), round(r.carbs_g * f, 1), round(r.fat_g * f, 1))


def _swap_context(fdc_id, slot, embeddings):
    row = embeddings.foods.loc[embeddings.foods["fdc_id"] == fdc_id].iloc[0]
    return row, component_of(row, slot) or Component("any", (row["meal_role"],))


def swap_options(fdc_id, slot, calories, allowed_foods, embeddings, exclude_bases=(), k=SWAP_LIST_SIZE):
    """Ranked replacements (MealItems) for one food of a meal, one per base food, sized to keep its calories.
    The app shows them a few at a time ("Show more options")."""
    subs = suggest_substitutes(fdc_id, allowed_foods, embeddings, None, slot, exclude_bases)
    _, component = _swap_context(fdc_id, slot, embeddings)
    return [_sized_item(r, calories, component) for r in _diverse(subs, k).itertuples()]


def servable_foods(foods):
    """Foods the planner could serve in some meal (pass at least one component's rules) - no flour, no organ meats."""
    ids = set()
    for comps in list(SLOTS.values()) + list(DISH_TEMPLATES.values()):
        for c in comps:
            ids |= set(component_foods(foods, c)["fdc_id"])
    return foods[foods["fdc_id"].isin(ids)]


def search_swap(query, fdc_id, slot, calories, allowed_foods, embeddings, exclude_bases=(), k=3):
    """Replacements matching what the user typed (e.g. 'brown rice'), ranked by embedding similarity to the text.
    Only foods of the same kind (meal role) are searched, so the meal keeps its shape and macros stay close.
    Returns (items, best_similarity)."""
    from src.embeddings import encode
    from src.parser import normalize_term, term_regex

    row, component = _swap_context(fdc_id, slot, embeddings)
    cands = servable_foods(allowed_foods)
    cands = cands[cands["meal_role"].isin(component.roles) & (cands["fdc_id"] != fdc_id)
                  & ~cands["base_name"].isin(exclude_bases)].copy()
    term = normalize_term(query)
    if cands.empty or not term:
        return [], 0.0
    sims = embeddings.similarity(encode([query])[0], cands["fdc_id"].tolist())
    words = cands["description"].str.contains(term_regex(term)).to_numpy()  # every typed word appears in the name
    cands["similarity"] = sims + SEARCH_WORD_BONUS * words
    cands = cands.sort_values("similarity", ascending=False)
    return [_sized_item(r, calories, component) for r in _diverse(cands, k).itertuples()], float(cands["similarity"].iloc[0])


def generate_week_plan(profile, foods=None, embeddings=None, taste_vector=None, food_penalties=None, seed=None):
    """Convenience: profile -> targets -> parse preferences -> filter foods -> plan."""
    from src.calculator import calculate_targets
    from src.parser import apply_to_foods, parse_preferences

    foods = foods if foods is not None else pd.read_csv(FOODS_CLEAN)
    targets = calculate_targets(profile)
    prefs = parse_preferences(profile.preferences_text, allergies=profile.allergies)
    allowed, _ = apply_to_foods(foods, prefs)
    if taste_vector is None:
        taste_vector = taste_from_preferences(prefs)
    planner = MealPlanner(allowed, embeddings, taste_vector, food_penalties, seed, nutrition=targets)
    return planner.plan_week(targets), targets, prefs


if __name__ == "__main__":
    import time
    import warnings
    warnings.filterwarnings("ignore")
    from src.calculator import UserProfile
    from src.embeddings import FoodEmbeddings

    user = UserProfile("Ahmed", 25, "male", 80, 180, 2, "lose",
                       "I'm allergic to shrimp, I hate mushrooms and I love grilled chicken and rice")
    t0 = time.time()
    plan, targets, prefs = generate_week_plan(user, embeddings=FoodEmbeddings(), seed=42)
    print(f"Target: {targets.calories} kcal | P {targets.protein_g} g | C {targets.carbs_g} g | F {targets.fat_g} g "
          f"(built in {time.time() - t0:.1f}s)")
    for day in plan.days:
        t = day.totals()
        print(f"\n{day.day}: {t['calories']:.0f} kcal | P {t['protein_g']:.0f} | C {t['carbs_g']:.0f} | F {t['fat_g']:.0f}")
        for meal in day.meals:
            print(f"  {meal.slot:<9} " + " + ".join(f"{i.name} ({i.grams:.0f} g)" for i in meal.items))
    print("\nMean daily error %:", plan.adherence())
