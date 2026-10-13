"""Phase 5: build a 7-day meal plan that hits the user's calorie and macro targets.

SR Legacy foods are ingredients, not meals, so every meal is a combination of 2-3 foods
(e.g. protein + carb + vegetable) with gram amounts solved to hit that meal's targets.

For each meal slot:
 1. Candidates = foods allowed by the parser (allergies / diet already removed) that fit the slot
    (e.g. breakfast carb = bread / oats / cereal, not pasta).
 2. Each candidate gets a score:
        score = W_FIT * role_fit + W_TASTE * cosine(food, taste_vector) + W_PREF * pref_score - variety_penalty
    role_fit:      how well the food does its job (protein share for a protein, low kcal for a vegetable...)
    taste_vector:  embedding of the user's liked foods (updated weekly in Phase 7)
    pref_score:    +1 per liked word, -1 per disliked word (from the parser)
    variety:       foods already used this week are penalised, same base food twice in a day is not allowed
 3. The best few candidates per component are combined; for every combination the grams are solved with
    bounded least squares (scipy lsq_linear) and the combination with the best fit + score wins.
 4. Each slot's target is taken from what is still left of the day, so the last meal corrects earlier errors.

Run a demo:  python -m src.planner
"""
import itertools
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from src.calculator import MEAL_SPLIT
from src.config import FOODS_CLEAN
from src.embeddings import embed_text, encode

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MACROS = ["calories", "protein_g", "carbs_g", "fat_g"]

W_FIT, W_TASTE, W_PREF = 1.0, 1.0, 1.0
W_ERROR = 4.0              # weight of the meal's macro error when comparing combinations
TOP_K = 5                  # candidates kept per component before trying combinations
REPEAT_PENALTY = 0.6       # per earlier use of the same food this week
BASE_REPEAT_PENALTY = 0.25  # per earlier use of the same base food (e.g. "chicken") this week
NOISE = 0.15               # randomness so two plans are not identical
PORTION_BASE_KCAL = 2400   # daily target above which maximum portions start to grow

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
                                              r"chow mein|fried|canned|flakes|pudding|fritter|candied|glazed|skin\b",
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
    if component.name in ("protein", "main"):
        return protein_share
    if component.name == "carb":
        return carb_share * 0.7 + (df["protein_g"] / 20).clip(0, 0.3)  # slight bonus for whole grains / legumes
    if component.name == "vegetable":
        return (1 - df["calories"] / 120).clip(0, 1)
    if component.name == "extra":
        return protein_share.clip(0, 0.5) + 0.5
    return pd.Series(0.5, index=df.index)


def grams_bounds(description, role, component):
    for roles, pattern, bounds in DENSE_GRAMS:
        if role in roles and re.search(pattern, description, re.I):
            return bounds
    return component.grams


def solve_grams(per100, lo, hi, target):
    """Bounded least squares: grams of each food so the meal's kcal/protein/carbs/fat are close to target.
    per100: n x 4 array (kcal, protein, carbs, fat per 100 g)."""
    a = per100.T / 100.0                                        # 4 x n, per gram
    b = np.array([target[m] for m in MACROS], dtype=float)
    scale = np.array([2.0, 1.5, 1.0, 1.0]) / np.maximum(b, 1)   # relative error, calories matter most
    res = lsq_linear(a * scale[:, None], b * scale, bounds=(lo, hi + 1e-9), method="bvls")  # bvls: ~10x faster for tiny problems
    grams = np.clip(np.round(res.x / 5) * 5, lo, hi)             # round to 5 g
    error = float(np.sqrt(np.mean(((a @ grams - b) * scale) ** 2)))
    return grams, error


class MealPlanner:
    def __init__(self, allowed_foods, embeddings=None, taste_vector=None, food_penalties=None, seed=None):
        """
        allowed_foods:  output of parser.apply_to_foods (blocked foods already removed, has `pref_score`)
        embeddings:     FoodEmbeddings (optional; enables the taste term)
        taste_vector:   384-d user taste (Phase 7), or None
        food_penalties: {fdc_id: penalty} from feedback (Phase 7), e.g. skipped foods
        """
        df = allowed_foods.copy()
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
        self.pools = {(slot, c.name): self._pool(c) for slot, comps in SLOTS.items() for c in comps}

    def _pool(self, component):
        df = self.foods[self.foods["meal_role"].isin(component.roles)]
        if component.include:
            df = df[df["description"].str.contains(component.include, case=False, regex=True)]
        if component.exclude:
            df = df[~df["description"].str.contains(component.exclude, case=False, regex=True)]
        df = df.copy()
        bounds = [grams_bounds(d, r, component) for d, r in zip(df["description"], df["meal_role"])]
        df["_lo"] = [b[0] for b in bounds]
        df["_hi"] = [b[1] for b in bounds]
        df["_score"] = W_FIT * role_fit(df, component) + W_TASTE * df["taste"] + W_PREF * df["pref_score"] - df["penalty"]
        return df

    def _candidates(self, slot, component, used_ids, used_bases_week, used_bases_today):
        pool = self.pools[(slot, component.name)]
        if component.name != "fat":  # cooking oil at lunch and dinner is fine
            pool = pool[~pool["base_name"].isin(used_bases_today)]
        if pool.empty:
            return pool
        score = pool["_score"].to_numpy().copy()
        score -= REPEAT_PENALTY * pool["fdc_id"].map(used_ids).fillna(0).to_numpy()
        score -= BASE_REPEAT_PENALTY * pool["base_name"].map(used_bases_week).fillna(0).to_numpy()
        score += self.rng.gumbel(0, NOISE, len(score))
        top = np.argsort(-score)[:component.top_k]
        return pool.iloc[top].assign(_final=score[top])

    def plan_meal(self, slot, target, used_ids, used_bases_week, used_bases_today):
        options = []
        for comp in SLOTS[slot]:
            cands = self._candidates(slot, comp, used_ids, used_bases_week, used_bases_today)
            if not cands.empty:
                options.append(cands)
        if not options:
            return Meal(slot, [], target)

        arrays = [(o[MACROS].to_numpy(float), o["_lo"].to_numpy(float), o["_hi"].to_numpy(float) * self.portion_scale,
                   o["_final"].to_numpy(float), o["base_name"].to_numpy()) for o in options]
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
            if best is None or value > best[0]:
                best = (value, combo, grams)
        if best is None:
            return Meal(slot, [], target)
        _, combo, grams = best

        items = []
        for (i, j), g in zip(enumerate(combo), grams):
            if g <= 0:  # optional component not needed (e.g. no cooking oil)
                continue
            r = options[i].iloc[j]
            f = g / 100.0
            items.append(MealItem(int(r["fdc_id"]), display_name(r["description"]), float(g),
                                  round(r["calories"] * f, 1), round(r["protein_g"] * f, 1),
                                  round(r["carbs_g"] * f, 1), round(r["fat_g"] * f, 1)))
        return Meal(slot, items, target)

    def plan_week(self, targets):
        """targets: calculator.NutritionTargets"""
        daily = {"calories": targets.calories, "protein_g": targets.protein_g, "carbs_g": targets.carbs_g, "fat_g": targets.fat_g}
        # Bigger eaters get bigger maximum portions (up to +60%) instead of an impossible target.
        self.portion_scale = float(np.clip(targets.calories / PORTION_BASE_KCAL, 1.0, 1.6))
        used_ids, used_bases_week, days = {}, {}, []
        for day_name in DAYS:
            day = DayPlan(day_name)
            remaining = dict(daily)
            remaining_share = 1.0
            used_today = set()
            for slot, share in MEAL_SPLIT.items():
                # This slot's share of whatever is still left of the day.
                target = {m: max(remaining[m] * share / remaining_share, 1.0) for m in MACROS}
                meal = self.plan_meal(slot, target, used_ids, used_bases_week, used_today)
                day.meals.append(meal)
                for item in meal.items:
                    used_ids[item.fdc_id] = used_ids.get(item.fdc_id, 0) + 1
                    base = self.base_of[item.fdc_id]
                    used_bases_week[base] = used_bases_week.get(base, 0) + 1
                    used_today.add(base)
                for m in MACROS:
                    remaining[m] -= meal.totals()[m]
                remaining_share -= share
            days.append(day)
        return WeekPlan(days, daily)


def suggest_substitutes(fdc_id, allowed_foods, embeddings, k=5):
    """Replacement for a rejected food: same meal role, close in meaning (embedding) and in macro profile."""
    foods = allowed_foods.reset_index(drop=True)
    row = embeddings.foods.loc[embeddings.foods["fdc_id"] == fdc_id].iloc[0]
    cands = foods[(foods["meal_role"] == row["meal_role"]) & (foods["fdc_id"] != fdc_id) & (foods["base_name"] != row["base_name"])].copy()
    sem = embeddings.similarity(embeddings.vector(fdc_id), cands["fdc_id"].tolist())

    def profile(df):
        kcal = np.maximum(df["calories"].to_numpy(dtype=float), 1)[:, None]
        p = np.column_stack([4 * df["protein_g"], 4 * df["carbs_g"], 9 * df["fat_g"]]).astype(float) / kcal
        return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)

    macro = profile(cands) @ profile(row.to_frame().T)[0]
    cands["similarity"] = 0.6 * sem + 0.4 * macro
    return cands.sort_values("similarity", ascending=False).head(k)


def generate_week_plan(profile, foods=None, embeddings=None, taste_vector=None, food_penalties=None, seed=None):
    """Convenience: profile -> targets -> parse preferences -> filter foods -> plan."""
    from src.calculator import calculate_targets
    from src.parser import apply_to_foods, parse_preferences

    foods = foods if foods is not None else pd.read_csv(FOODS_CLEAN)
    targets = calculate_targets(profile)
    prefs = parse_preferences(profile.preferences_text)
    allowed, _ = apply_to_foods(foods, prefs)
    if taste_vector is None:
        taste_vector = taste_from_preferences(prefs)
    planner = MealPlanner(allowed, embeddings, taste_vector, food_penalties, seed)
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
