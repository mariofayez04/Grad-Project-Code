"""Phase 5 tests: the weekly plan is safe, close to targets, varied and reproducible.

Run:  python -m pytest tests -q
"""
import pandas as pd
import pytest

from src.calculator import UserProfile, calculate_targets
from src.config import EMBEDDINGS, FOODS_CLEAN
from src.embeddings import FoodEmbeddings
from src.parser import apply_to_foods, parse_preferences
from src.planner import (DAYS, MAIN_CARB, MAIN_PROTEIN, MealPlanner, component_foods, generate_week_plan, search_swap,
                         solve_grams,
                         suggest_substitutes, swap_options)

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


@pytest.fixture(scope="module")
def emb(foods):
    return FoodEmbeddings(foods)


def make_plan(foods, emb, text="", goal="maintain", sex="male", seed=1):
    user = UserProfile("T", 30, sex, 75, 175, 2, goal, text)
    return generate_week_plan(user, foods=foods, embeddings=emb, seed=seed)


def test_solve_grams_hits_target():
    import numpy as np
    per100 = np.array([[165, 31, 0, 3.6], [130, 2.7, 28, 0.3], [35, 2.4, 7, 0.4]])  # chicken, rice, broccoli
    target = {"calories": 600, "protein_g": 45, "carbs_g": 60, "fat_g": 8}
    grams, error = solve_grams(per100, np.array([80, 60, 80.]), np.array([300, 350, 250.]), target)
    kcal = per100[:, 0] @ grams / 100
    assert abs(kcal - 600) / 600 < 0.1 and error < 0.2


@pytest.mark.parametrize("goal", ["maintain", "lose", "gain"])
def test_week_structure_and_adherence(foods, emb, goal):
    plan, targets, _ = make_plan(foods, emb, goal=goal)
    assert [d.day for d in plan.days] == DAYS
    for day in plan.days:
        assert [m.slot for m in day.meals] == ["breakfast", "lunch", "dinner", "snack"]
        assert all(m.items for m in day.meals)
    err = plan.adherence()
    assert err["calories"] < 5 and err["protein_g"] < 10 and err["carbs_g"] < 10 and err["fat_g"] < 15


def test_allergies_never_in_plan(foods, emb):
    plan, _, prefs = make_plan(foods, emb, "I'm allergic to peanuts, shrimp and milk")
    df = plan.to_dataframe().merge(foods[["fdc_id", "allergens"]], on="fdc_id")
    tags = df["allergens"].fillna("")
    assert not tags.str.contains("peanut|shellfish|milk").any(), df.loc[tags.str.contains("peanut|shellfish|milk"), "name"].tolist()


def test_vegetarian_plan(foods, emb):
    plan, _, _ = make_plan(foods, emb, "I'm vegetarian")
    df = plan.to_dataframe().merge(foods[["fdc_id", "is_vegetarian"]], on="fdc_id")
    assert df["is_vegetarian"].all()


def test_no_base_food_twice_in_a_day(foods, emb):
    plan, _, _ = make_plan(foods, emb)
    df = plan.to_dataframe().merge(foods[["fdc_id", "base_name", "meal_role"]], on="fdc_id")
    df = df[df["meal_role"] != "fat"]  # cooking oil at lunch and dinner is allowed
    assert not df.duplicated(["day_index", "base_name"]).any()


def test_variety_across_week(foods, emb):
    plan, _, _ = make_plan(foods, emb)
    df = plan.to_dataframe()
    assert df["fdc_id"].nunique() / len(df) > 0.7  # most items are different foods


def test_preferences_shape_the_plan(foods, emb):
    liked, _, _ = make_plan(foods, emb, "I love salmon and tuna", seed=3)
    plain, _, _ = make_plan(foods, emb, "", seed=3)
    fish = lambda p: p.to_dataframe()["name"].str.contains("salmon|tuna", case=False).sum()
    assert fish(liked) > fish(plain)


def test_disliked_food_avoided(foods, emb):
    plan, _, _ = make_plan(foods, emb, "I hate broccoli and rice")
    names = plan.to_dataframe()["name"]
    assert names.str.contains(r"\bbroccoli|\brice\b", case=False).sum() <= 1


def test_same_seed_same_plan(foods, emb):
    a, _, _ = make_plan(foods, emb, seed=7)
    b, _, _ = make_plan(foods, emb, seed=7)
    assert a.to_dataframe().equals(b.to_dataframe())


def test_food_penalty_removes_food(foods, emb):
    """Phase 7 hook: a heavily penalised food should disappear from the next plan."""
    first, targets, _ = make_plan(foods, emb, seed=5)
    top = first.to_dataframe()["fdc_id"].value_counts().index[0]
    allowed, _ = apply_to_foods(foods, parse_preferences(""))
    again = MealPlanner(allowed, emb, None, {top: 5.0}, seed=5).plan_week(targets)
    assert top not in set(again.to_dataframe()["fdc_id"])


def test_substitutes_same_role(foods, emb):
    allowed, _ = apply_to_foods(foods, parse_preferences("allergic to fish"))
    chicken = foods.loc[foods["description"].str.contains(r"^Chicken, broiler.*breast.*grilled"), "fdc_id"].iloc[0]
    subs = suggest_substitutes(chicken, allowed, emb, k=5)
    assert len(subs) == 5 and (subs["meal_role"] == "protein").all()
    assert not subs["allergens"].fillna("").str.contains("fish").any()


def test_swap_options_fit_the_meal(foods, emb):
    """A swapped lunch protein is another lunch protein, allergy-safe, not already in the meal, same calories."""
    allowed, _ = apply_to_foods(foods, parse_preferences("allergic to fish"))
    chicken = foods.loc[foods["description"].str.contains(r"^Chicken, broiler.*breast.*grilled"), "fdc_id"].iloc[0]
    opts = swap_options(chicken, "lunch", 300.0, allowed, emb, exclude_bases={"turkey"}, k=3)
    assert len(opts) == 3
    lunch_proteins = set(component_foods(allowed, MAIN_PROTEIN)["fdc_id"])
    by_id = foods.set_index("fdc_id")
    for item in opts:
        assert item.fdc_id in lunch_proteins and item.fdc_id != chicken
        assert by_id.loc[item.fdc_id, "base_name"] != "turkey"
        assert "fish" not in str(by_id.loc[item.fdc_id, "allergens"])
        assert abs(item.calories - 300) < 40  # grams rounded to 5 g


def test_swap_options_are_varied(foods, emb):
    """One suggestion per base food and no two with the same shown name (the 'Pasta, cooked' twice bug)."""
    allowed, _ = apply_to_foods(foods, parse_preferences(""))
    bread = foods.loc[foods["description"].str.contains(r"^Bread, white.*toasted"), "fdc_id"].iloc[0]
    opts = swap_options(bread, "lunch", 276.0, allowed, emb)
    names = [o.name for o in opts]
    bases = foods.set_index("fdc_id").loc[[o.fdc_id for o in opts], "base_name"]
    assert len(opts) > 6 and len(set(names)) == len(names) and bases.is_unique
    assert set(bases) & {"rice", "pasta", "potatoes"}


def test_search_swap_finds_typed_food(foods, emb):
    allowed, _ = apply_to_foods(foods, parse_preferences("allergic to wheat"))
    bread = foods.loc[foods["description"].str.contains(r"^Bread, white.*toasted"), "fdc_id"].iloc[0]
    items, best = search_swap("brown rice", bread, "lunch", 276.0, allowed, emb)
    assert items[0].name.startswith("Rice, brown") and best > 0.6
    assert abs(items[0].calories - 276) < 40
    carbs = set(component_foods(foods, MAIN_CARB)["meal_role"])
    by_id = foods.set_index("fdc_id")
    assert all(by_id.loc[i.fdc_id, "meal_role"] in carbs and "wheat" not in str(by_id.loc[i.fdc_id, "allergens"]) for i in items)
    _, low = search_swap("chicken", bread, "lunch", 276.0, allowed, emb)  # not a carb: weak match
    assert low < 0.6
