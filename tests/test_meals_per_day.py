"""Feature #11 tests: 3, 4 or 5 meals a day.

Run:  python -m pytest tests -q
"""
from dataclasses import asdict

import pandas as pd
import pytest

from src import db
from src.calculator import MEAL_SPLITS, UserProfile, calculate_targets, meal_split
from src.config import EMBEDDINGS, FOODS_CLEAN
from src.parser import apply_to_foods, parse_preferences
from src.planner import MACROS, MealPlanner, daily_targets, generate_week_plan, plan_slots, slot_label

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")
ORDER = {3: ["breakfast", "lunch", "dinner"], 4: ["breakfast", "lunch", "dinner", "snack"],
         5: ["breakfast", "morning_snack", "lunch", "snack", "dinner"]}


def user(n, text=""):
    return UserProfile("M", 30, "female", 62, 166, 2, "maintain", text, n)


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_splits_add_up(n):
    assert sum(meal_split(n).values()) == pytest.approx(1.0)
    t = calculate_targets(user(n))
    assert t.meals_per_day == n and list(t.per_meal()) == ORDER[n]
    assert sum(m["calories"] for m in t.per_meal().values()) == pytest.approx(t.calories, abs=3)


def test_default_and_invalid():
    assert meal_split(None) == MEAL_SPLITS[4]                     # plans saved before this setting
    assert calculate_targets(UserProfile("M", 30, "male", 75, 175, 2, "maintain")).meals_per_day == 4
    with pytest.raises(ValueError, match="meals_per_day"):
        calculate_targets(user(6))


@pytest.mark.parametrize("n", [3, 5])
def test_week_plan_with_n_meals(foods, n):
    plan, targets, _ = generate_week_plan(user(n, "I'm allergic to fish"), foods=foods, seed=5)
    base = foods.set_index("fdc_id")["base_name"]
    for day in plan.days:
        assert [m.slot for m in day.meals] == ORDER[n]
        assert all(m.items for m in day.meals)
        non_oil = [base[i.fdc_id] for m in day.meals for i in m.items if base[i.fdc_id] != "oil"]
        assert len(non_oil) == len(set(non_oil))                  # the two snacks don't repeat a food
    assert all(err < 10 for err in plan.adherence().values())
    df = plan.to_dataframe()
    assert plan_slots(df) == ORDER[n]


def test_new_morning_snack(foods):
    from src.embeddings import FoodEmbeddings
    emb = FoodEmbeddings(foods)
    targets = calculate_targets(user(5))
    allowed, _ = apply_to_foods(foods, parse_preferences(""))
    items = MealPlanner(allowed, emb, None, None, seed=1).plan_week(targets).to_dataframe()
    planner = MealPlanner(allowed, emb, None, None, seed=9)
    meal = planner.replan_meal(items, 1, "morning_snack", daily_targets(targets), 5)
    current = items[(items["day_index"] == 1) & (items["slot"] == "morning_snack")]
    assert meal.items and not {i.fdc_id for i in meal.items} & set(current["fdc_id"])
    day = planner.replan_day(items, 1, daily_targets(targets), 5)
    assert [m.slot for m in day.meals] == ORDER[5]


def test_slot_labels():
    assert slot_label("morning_snack", 5) == "Morning snack"
    assert slot_label("snack", 5) == "Afternoon snack" and slot_label("snack", 4) == "Snack"
    assert slot_label("breakfast", None) == "Breakfast"


# --- storage and app ---------------------------------------------------------------------------------------
@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


def test_profile_keeps_meals_per_day(temp_db):
    uid = db.create_user(user(5), "1234")
    assert db.get_user(uid)["meals_per_day"] == 5
    db.update_user(uid, user(3))
    assert db.get_user(uid)["meals_per_day"] == 3
    plan, targets, prefs = generate_week_plan(user(3), seed=1)
    pid = db.save_plan(uid, plan, targets, prefs)
    assert db.latest_plan(uid)["targets"]["meals_per_day"] == 3
    assert plan_slots(db.plan_items(pid)) == ORDER[3]


def test_pdf_names_the_snacks(foods):
    from src.report import week_pdf
    from tests.test_report import text_of
    plan, targets, prefs = generate_week_plan(user(5), foods=foods, seed=2)
    info = {"week_number": 1, "targets": asdict(targets), "prefs": prefs.summary()}
    text = text_of(week_pdf(info, plan.to_dataframe().assign(eaten=None), "M", compress=False))
    assert "Morning snack" in text and "Afternoon snack" in text


def test_app_five_meals(temp_db):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    at.radio[0].set_value("New user").run()
    at.text_input[0].input("Hana")
    at.text_input[1].input("1234")
    at.text_input[2].input("1234")
    next(r for r in at.radio if r.label == "Meals per day").set_value(5)
    at.button[0].click().run()
    assert not at.exception
    assert sum(c.label == "Eaten" for c in at.checkbox) == 35            # 7 days x 5 meals
    assert db.get_user(db.list_users()[0]["id"])["meals_per_day"] == 5
