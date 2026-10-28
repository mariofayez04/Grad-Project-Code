"""Feature #9 tests: Egyptian dishes (recipes, data, planner, allergies, shopping, app).

Run:  python -m pytest tests -q
"""
import pandas as pd
import pytest

from src.allergens import diet_flags, tag_allergens
from src.calculator import UserProfile
from src.config import EMBEDDINGS, FOODS_CLEAN
from src.dishes import DISH_ID_START, DISHES, build_dishes, dishes_for, load_recipes, recipe_line
from src.planner import generate_week_plan, swap_options
from src.shopping import expand_dishes, shopping_list

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


@pytest.fixture(scope="module")
def dishes(foods):
    return foods[foods["meal_role"] == "dish"].set_index("base_name")


def person(setting="sometimes", text="", goal="maintain"):
    return UserProfile("E", 32, "male", 78, 176, 2, goal, text, 4, "", setting)


def dish_meals(plan, foods):
    ids = set(foods.loc[foods["meal_role"] == "dish", "fdc_id"])
    df = plan.to_dataframe()
    return df[df["fdc_id"].isin(ids)]


# --- recipes and data -----------------------------------------------------------------------------------------
def test_build_dishes_math():
    """Nutrients per 100 g = gram-weighted average of the ingredients; allergens and flags from the ingredients."""
    base = pd.DataFrame({
        "fdc_id": [1, 2], "description": ["Rice, cooked", "Cheese, feta"], "food_category": ["Cereal Grains and Pasta", "Dairy and Egg Products"],
        "calories": [130.0, 260.0], "protein_g": [3.0, 14.0], "carbs_g": [28.0, 4.0], "fat_g": [0.3, 21.0],
        "fiber_g": [0.4, 0.0], "estimated": ["", "fiber_g"]})
    import src.dishes as d
    saved = d.DISHES
    d.DISHES = [("Test dish", "طبق", "rice with feta", ["lunch"], [("Rice, cooked", 300), ("Cheese, feta", 100)])]
    try:
        rows, recipe = build_dishes(base, ["fiber_g"], tag_allergens, diet_flags)
    finally:
        d.DISHES = saved
    r = rows.iloc[0]
    assert r["fdc_id"] == DISH_ID_START and r["portion_g"] == 400 and r["meal_role"] == "dish"
    assert r["calories"] == pytest.approx((130 * 300 + 260 * 100) / 400)
    assert r["allergens"] == "milk" and r["is_vegetarian"] and r["estimated"] == "fiber_g"
    assert recipe["grams"].tolist() == [300, 100] and (recipe["serving_g"] == 400).all()


def test_dishes_in_food_table(foods, dishes):
    assert len(dishes) == len(DISHES) == 20
    assert (dishes["fdc_id"] >= DISH_ID_START).all() and (dishes["food_category"] == "Egyptian Dishes").all()
    assert not dishes[["calories", "protein_g", "fiber_g", "sodium_mg"]].isna().any().any()
    koshari = dishes.loc["koshari"]
    assert 500 < koshari["calories"] * koshari["portion_g"] / 100 < 750      # one plate
    assert "wheat" in koshari["allergens"] and koshari["is_vegetarian"]       # pasta
    assert "sesame" in dishes.loc["taameya", "allergens"]                     # tahini
    assert "milk" in dishes.loc["molokhia with rice and chicken", "allergens"] and not dishes.loc["molokhia with rice and chicken", "is_vegetarian"]
    assert not dishes.loc["halabessa", "is_vegetarian"]                        # chicken broth
    assert "fish" in dishes.loc["sayadiya", "allergens"]


def test_recipes_file(dishes):
    recipes = load_recipes()
    assert set(recipes["dish_fdc_id"]) == set(dishes["fdc_id"])
    fid = int(dishes.loc["koshari", "fdc_id"])
    line = recipe_line(recipes, fid, dishes.loc["koshari", "portion_g"] * 2)    # a double portion
    assert line.startswith("rice 240 g, lentils 160 g, pasta 160 g") and "cumin" not in line   # spices left out


def test_dishes_for_slots():
    assert "Ful medames" in dishes_for("breakfast") and "Koshari" not in dishes_for("breakfast")
    assert "Koshari" in dishes_for("lunch") and dishes_for("morning_snack") == dishes_for("snack")


# --- planner ---------------------------------------------------------------------------------------------------
def test_frequency_setting(foods):
    counts = {}
    for setting in ("never", "sometimes", "often"):
        plan, _, _ = generate_week_plan(person(setting), foods=foods, seed=4)
        counts[setting] = dish_meals(plan, foods).groupby(["day_index", "slot"]).ngroups
        assert all(err < 10 for err in plan.adherence().values())
    assert counts["never"] == 0 and 3 <= counts["sometimes"] <= 12 and counts["often"] > counts["sometimes"]


def test_dish_meals_are_whole_meals(foods):
    plan, _, _ = generate_week_plan(person("often"), foods=foods, seed=2)
    df = plan.to_dataframe()
    dish_ids = set(foods.loc[foods["meal_role"] == "dish", "fdc_id"])
    for (d, slot), meal in df.groupby(["day_index", "slot"]):
        n_dishes = meal["fdc_id"].isin(dish_ids).sum()
        assert n_dishes <= 1                                                      # one dish + sides, never two
        if n_dishes:
            assert len(meal) <= 4
    assert (df["grams"] % 5 == 0).all()


def test_allergies_and_diet_respected(foods):
    plan, _, _ = generate_week_plan(person("often", "I'm allergic to wheat and sesame"), foods=foods, seed=3)
    served = dish_meals(plan, foods)["fdc_id"]
    tags = foods.set_index("fdc_id").loc[served, "allergens"].fillna("")
    assert not tags.str.contains("wheat|sesame").any()
    veg, _, _ = generate_week_plan(person("often", "I'm vegetarian"), foods=foods, seed=3)
    assert foods.set_index("fdc_id").loc[dish_meals(veg, foods)["fdc_id"], "is_vegetarian"].all()


def test_liked_dish_shows_up(foods):
    plan, _, prefs = generate_week_plan(person("sometimes", "I love koshari"), foods=foods, seed=6)
    assert "koshari" in prefs.prefer_terms
    assert dish_meals(plan, foods)["name"].str.startswith("Koshari").any()


def test_swap_a_dish(foods, dishes):
    from src.embeddings import FoodEmbeddings
    from src.parser import apply_to_foods, parse_preferences
    allowed, _ = apply_to_foods(foods, parse_preferences(""))
    opts = swap_options(int(dishes.loc["koshari", "fdc_id"]), "lunch", 600.0, allowed, FoodEmbeddings(foods))
    assert opts and all(o.fdc_id >= DISH_ID_START for o in opts)              # another dish
    assert not any(o.name.startswith("Ful medames") for o in opts)            # ful is not a lunch dish here


# --- shopping and app -------------------------------------------------------------------------------------------
def test_shopping_buys_ingredients(foods, dishes):
    koshari = int(dishes.loc["koshari", "fdc_id"])
    items = pd.DataFrame({"day_index": [0], "slot": ["lunch"], "fdc_id": [koshari], "grams": [454.0], "eaten": [None]})
    parts = expand_dishes(items)
    assert koshari not in set(parts["fdc_id"]) and parts["grams"].sum() == pytest.approx(454)
    shop = shopping_list(items, foods).set_index("item")
    assert {"Rice, white", "Lentils", "Pasta", "Cumin"} <= set(shop.index)
    assert shop.loc["Lentils", "label"] == "dry" and shop.loc["Rice, white", "grams"] == pytest.approx(120 * 0.35)


def test_app_setting_and_arabic(tmp_path, monkeypatch):
    from src import db
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "t.db"))
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    at.radio[0].set_value("New user").run()
    at.text_input[0].input("Sherif")
    at.text_input[1].input("1234")
    at.text_input[2].input("1234")
    next(r for r in at.radio if "Egyptian dishes" in r.label).set_value("often")
    at.button[0].click().run()
    assert not at.exception
    uid = db.list_users()[0]["id"]
    assert db.get_user(uid)["egyptian_dishes"] == "often" and db.latest_plan(uid)["targets"]["egyptian_dishes"] == "often"
    page = " ".join(m.value for m in at.markdown)
    assert "In this portion" in page and any(a in page for a in ("فول", "كشري", "ملوخية", "شكشوكة", "بصارة", "محشي", "طعمية"))
