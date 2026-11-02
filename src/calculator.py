"""Phase 2: daily calorie target and macro split.

Pipeline:  BMR (Mifflin-St Jeor) -> TDEE (activity multiplier) -> goal adjustment -> safety floor -> macros in grams.
No AI here: these are standard formulas (see report for sources).

Run a demo:  python -m src.calculator
"""
from dataclasses import dataclass, field

ACTIVITY_MULTIPLIERS = {
    1: 1.2,    # sedentary: little or no exercise
    2: 1.55,   # moderately active: exercise 3-5 days/week
    3: 1.725,  # very active: hard exercise 6-7 days/week
}
ACTIVITY_LABELS = {1: "Sedentary", 2: "Moderately active", 3: "Very active"}

GOAL_ADJUSTMENT = {"maintain": 0.0, "lose": -0.20, "gain": 0.10}
GOAL_LABELS = {"maintain": "Maintain weight", "lose": "Lose weight", "gain": "Gain muscle"}
GOAL_ALIASES = {"maintain weight": "maintain", "lose weight": "lose", "gain muscle": "gain", "gain weight": "gain"}

# Share of daily calories from protein / carbs / fat.
MACRO_SPLIT = {
    "maintain": {"protein": 0.25, "carbs": 0.45, "fat": 0.30},
    "lose":     {"protein": 0.35, "carbs": 0.35, "fat": 0.30},
    "gain":     {"protein": 0.30, "carbs": 0.45, "fat": 0.25},
}
KCAL_PER_GRAM = {"protein": 4, "carbs": 4, "fat": 9}
MAX_PROTEIN_G_PER_KG = 2.2  # upper end of evidence-based intake for active adults
DIABETES_MAX_CARBS = 0.40   # diabetes-friendly mode: carbs share capped, the rest goes to fat (mostly unsaturated)

CALORIE_FLOOR = {"male": 1500, "female": 1200}

# How the daily target is spread across meals (used by the planner in Phase 5), for 3, 4 or 5 meals a day.
# Key order = the order the meals are shown and planned; the last one also corrects the day's small errors.
MEAL_SPLITS = {
    3: {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30},
    4: {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.30, "snack": 0.10},
    5: {"breakfast": 0.25, "morning_snack": 0.10, "lunch": 0.30, "snack": 0.10, "dinner": 0.25},
}
MEALS_LABELS = {3: "3 meals (breakfast, lunch, dinner)", 4: "4 meals (+ a snack)", 5: "5 meals (+ two snacks)"}
DEFAULT_MEALS = 4
MEAL_SPLIT = MEAL_SPLITS[DEFAULT_MEALS]

LIMITS = {"age": (18, 100), "weight_kg": (30, 300), "height_cm": (120, 230)}


@dataclass
class UserProfile:
    name: str
    age: int
    sex: str           # "male" / "female"
    weight_kg: float
    height_cm: float
    activity_level: int  # 1, 2, 3
    goal: str          # "maintain" / "lose" / "gain"
    preferences_text: str = ""
    meals_per_day: int = DEFAULT_MEALS  # 3, 4 or 5
    health_modes: str = ""              # "", "diabetes", "low_sodium" or "diabetes|low_sodium" (Feature #10)
    egyptian_dishes: str = "sometimes"  # "often" / "sometimes" / "never" (Feature #9)
    allergies: str = ""                 # ticked allergen groups, "milk|peanut"; "" = none (safer than free text alone)


@dataclass
class NutritionTargets:
    bmr: float
    tdee: float
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    bmi: float
    floor_applied: bool = False
    warnings: list = field(default_factory=list)
    meals_per_day: int = DEFAULT_MEALS
    micros: dict = field(default_factory=dict)  # fiber, sodium, calcium... reference values (src/nutrients.py)
    health_modes: str = ""
    egyptian_dishes: str = "sometimes"

    def per_meal(self):
        """Calorie and macro targets for each meal slot."""
        return {
            meal: {
                "calories": round(self.calories * share),
                "protein_g": round(self.protein_g * share),
                "carbs_g": round(self.carbs_g * share),
                "fat_g": round(self.fat_g * share),
            }
            for meal, share in meal_split(self.meals_per_day).items()
        }


def meal_split(meals_per_day=DEFAULT_MEALS):
    """{slot: share of the day} for 3, 4 or 5 meals (a saved plan without the setting had 4)."""
    return MEAL_SPLITS[int(meals_per_day or DEFAULT_MEALS)]


def normalize_sex(sex):
    s = str(sex).strip().lower()
    if s in ("male", "m", "man"):
        return "male"
    if s in ("female", "f", "woman"):
        return "female"
    raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")


def normalize_goal(goal):
    g = str(goal).strip().lower()
    g = GOAL_ALIASES.get(g, g)
    if g not in GOAL_ADJUSTMENT:
        raise ValueError(f"goal must be one of {list(GOAL_ADJUSTMENT)}, got {goal!r}")
    return g


def validate(profile):
    for attr, (lo, hi) in LIMITS.items():
        value = getattr(profile, attr)
        if not lo <= value <= hi:
            raise ValueError(f"{attr} must be between {lo} and {hi}, got {value}")
    if profile.activity_level not in ACTIVITY_MULTIPLIERS:
        raise ValueError(f"activity_level must be 1, 2 or 3, got {profile.activity_level}")
    if profile.meals_per_day not in MEAL_SPLITS:
        raise ValueError(f"meals_per_day must be 3, 4 or 5, got {profile.meals_per_day}")
    from src.nutrients import HEALTH_MODES
    unknown = {m for m in (profile.health_modes or "").split("|") if m} - set(HEALTH_MODES)
    if unknown:
        raise ValueError(f"unknown health mode(s): {sorted(unknown)}")
    if (profile.egyptian_dishes or "sometimes") not in ("often", "sometimes", "never"):
        raise ValueError(f"egyptian_dishes must be often, sometimes or never, got {profile.egyptian_dishes!r}")


def calculate_bmr(sex, age, weight_kg, height_cm):
    """Mifflin-St Jeor: 10*kg + 6.25*cm - 5*age + 5 (male) or - 161 (female)."""
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if normalize_sex(sex) == "male" else base - 161


def apply_activity(bmr, level):
    return bmr * ACTIVITY_MULTIPLIERS[level]


def adjust_for_goal(tdee, goal):
    return tdee * (1 + GOAL_ADJUSTMENT[normalize_goal(goal)])


def apply_floor(calories, sex):
    """Never recommend less than the minimum safe intake. Returns (calories, floor_applied)."""
    floor = CALORIE_FLOOR[normalize_sex(sex)]
    return (floor, True) if calories < floor else (calories, False)


def macro_split(calories, goal, weight_kg=None, diabetes=False):
    """Convert a calorie target into grams of protein / carbs / fat.

    If weight_kg is given, protein is capped at MAX_PROTEIN_G_PER_KG and the spare calories go to carbs.
    diabetes=True caps carbs at DIABETES_MAX_CARBS of calories and gives the rest to fat.
    """
    split = dict(MACRO_SPLIT[normalize_goal(goal)])
    if diabetes and split["carbs"] > DIABETES_MAX_CARBS:
        split["fat"] += split["carbs"] - DIABETES_MAX_CARBS
        split["carbs"] = DIABETES_MAX_CARBS
    grams = {m: calories * share / KCAL_PER_GRAM[m] for m, share in split.items()}
    if weight_kg is not None:
        cap = MAX_PROTEIN_G_PER_KG * weight_kg
        if grams["protein"] > cap:
            spare = "fat" if diabetes else "carbs"
            grams[spare] += (grams["protein"] - cap) * KCAL_PER_GRAM["protein"] / KCAL_PER_GRAM[spare]
            grams["protein"] = cap
    return {f"{m}_g": round(g) for m, g in grams.items()}


def calculate_bmi(weight_kg, height_cm):
    return weight_kg / (height_cm / 100) ** 2


def calculate_targets(profile):
    """Full Phase 2 pipeline for one user."""
    profile.sex = normalize_sex(profile.sex)
    profile.goal = normalize_goal(profile.goal)
    validate(profile)

    bmr = calculate_bmr(profile.sex, profile.age, profile.weight_kg, profile.height_cm)
    tdee = apply_activity(bmr, profile.activity_level)
    calories, floor_applied = apply_floor(adjust_for_goal(tdee, profile.goal), profile.sex)
    calories = round(calories)
    modes = profile.health_modes or ""
    macros = macro_split(calories, profile.goal, profile.weight_kg, diabetes="diabetes" in modes.split("|"))
    bmi = calculate_bmi(profile.weight_kg, profile.height_cm)

    warnings = []
    if floor_applied:
        warnings.append(f"Your calculated target was below the safe minimum, so it was raised to {calories} kcal.")
    if bmi < 18.5 and profile.goal == "lose":
        warnings.append(f"Your BMI is {bmi:.1f} (underweight). Losing weight is not recommended; please consult a doctor.")
    if bmi >= 30 and profile.goal == "gain":
        warnings.append(f"Your BMI is {bmi:.1f}. Please consult a doctor before a calorie-surplus plan.")

    from src.nutrients import micro_targets
    if modes:
        warnings.append("Diet modes are general guidance, not a treatment plan. With diabetes or high blood pressure, "
                        "follow your doctor or dietitian - medicines may need adjusting when your diet changes.")
    return NutritionTargets(bmr=round(bmr, 1), tdee=round(tdee, 1), calories=calories, bmi=round(bmi, 1),
                            floor_applied=floor_applied, warnings=warnings, meals_per_day=int(profile.meals_per_day),
                            micros=micro_targets(profile.sex, profile.age, calories, modes), health_modes=modes,
                            egyptian_dishes=profile.egyptian_dishes or "sometimes", **macros)


if __name__ == "__main__":
    demo = UserProfile(name="Ahmed", age=25, sex="male", weight_kg=80, height_cm=180, activity_level=2, goal="lose")
    t = calculate_targets(demo)
    print(f"{demo.name}: BMR {t.bmr} | TDEE {t.tdee} | target {t.calories} kcal | BMI {t.bmi}")
    print(f"Macros: protein {t.protein_g} g, carbs {t.carbs_g} g, fat {t.fat_g} g")
    for meal, m in t.per_meal().items():
        print(f"  {meal:<9} {m}")
    for w in t.warnings:
        print("  !", w)
