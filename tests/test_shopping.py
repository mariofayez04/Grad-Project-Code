"""Feature #3 tests: the weekly shopping list.

Run:  python -m pytest tests -q
"""
import pandas as pd
import pytest

from src.config import FOODS_CLEAN
from src.shopping import EGG_GRAMS, RAW_MEAT_FACTOR, format_amount, purchase, shopping_list, to_text


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


def fid(foods, pattern):
    return int(foods.loc[foods["description"].str.contains(pattern, regex=True), "fdc_id"].iloc[0])


def week(rows):
    """rows: (day_index, slot, fdc_id, grams, eaten)"""
    return pd.DataFrame(rows, columns=["day_index", "slot", "fdc_id", "grams", "eaten"])


@pytest.mark.parametrize("desc, cat, name, factor, label", [
    ("Rice, white, long-grain, regular, enriched, cooked", "Cereal Grains and Pasta", "Rice, white", 0.35, "dry"),
    ("Chicken, broilers or fryers, breast, meat only, cooked, roasted", "Poultry Products", "Chicken, breast", RAW_MEAT_FACTOR, "raw"),
    ("Beans, kidney, red, mature seeds, cooked, boiled, without salt", "Legumes and Legume Products", "Beans, kidney", 0.4, "dry"),
    ("Fish, tuna, light, canned in water, drained solids", "Finfish and Shellfish Products", "Fish, tuna, light", 1.0, ""),
    ("Asparagus, cooked, boiled, drained", "Vegetables and Vegetable Products", "Asparagus", 1.0, ""),
    ("Egg, whole, cooked, hard-boiled", "Dairy and Egg Products", "Eggs", 1.0, ""),
    ("Oil, olive, salad or cooking", "Fats and Oils", "Cooking oil", 1.0, ""),
    ("Apples, raw, with skin", "Fruits and Fruit Juices", "Apples, with skin", 1.0, ""),
])
def test_purchase(desc, cat, name, factor, label):
    assert purchase(desc, cat) == (name, factor, label)


def test_list_merges_and_converts(foods):
    rice = fid(foods, r"^Rice, white, long-grain, regular, enriched, cooked")
    egg1, egg2 = fid(foods, r"^Egg, whole, cooked, hard-boiled"), fid(foods, r"^Egg, whole, cooked, scrambled")
    oil1, oil2 = fid(foods, r"^Oil, olive, salad or cooking"), fid(foods, r"^Oil, sunflower")
    items = week([(0, "lunch", rice, 300, None), (1, "lunch", rice, 200, None), (0, "breakfast", egg1, 100, None),
                  (1, "breakfast", egg2, 150, None), (0, "lunch", oil1, 20, None), (1, "dinner", oil2, 25, None)])
    shop = shopping_list(items, foods).set_index("item")
    assert shop.loc["Rice, white", "grams"] == pytest.approx(500 * 0.35)
    assert shop.loc["Rice, white", "plate_g"] == 500 and shop.loc["Rice, white", "meals"] == 2
    assert shop.loc["Eggs", "grams"] == 250 and shop.loc["Eggs", "note"] == f"≈ {round(250 / EGG_GRAMS)} eggs"
    assert shop.loc["Cooking oil", "grams"] == 45 and "any kind" in shop.loc["Cooking oil", "note"]
    assert list(shop["section"].unique()) == ["🥚 Eggs & dairy", "🍚 Rice, pasta & grains", "🫒 Oils & fats"]


def test_skip_eaten_meals(foods):
    rice = fid(foods, r"^Rice, white, long-grain, regular, enriched, cooked")
    items = week([(0, "lunch", rice, 300, 1), (1, "lunch", rice, 200, 0), (2, "lunch", rice, 100, None)])
    assert shopping_list(items, foods, skip_eaten=True)["plate_g"].sum() == 300   # only meals not eaten yet
    assert shopping_list(items, foods)["plate_g"].sum() == 600
    assert shopping_list(items.assign(eaten=1), foods, skip_eaten=True).empty


def test_real_week_plan(foods):
    """A generated plan: every food lands in the list and nothing is lost on the way."""
    from src.calculator import UserProfile
    from src.planner import generate_week_plan
    plan, _, _ = generate_week_plan(UserProfile("T", 30, "male", 75, 175, 2, "maintain"), foods=foods, seed=3)
    items = plan.to_dataframe()
    shop = shopping_list(items, foods)
    assert shop["plate_g"].sum() == pytest.approx(items["grams"].sum())
    assert not shop.duplicated(["section", "item", "label"]).any()  # one line per thing to buy
    from src.shopping import expand_dishes
    assert len(shop) < expand_dishes(items)["fdc_id"].nunique()   # variants were merged (dishes count as their ingredients)
    text = to_text(shop)
    assert "[ ]" in text and "🥦 Vegetables" in text


@pytest.mark.parametrize("grams, text", [(3, "5 g"), (23, "25 g"), (174, "170 g"), (1240, "1.2 kg")])
def test_format_amount(grams, text):
    assert format_amount(grams) == text
