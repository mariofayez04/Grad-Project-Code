"""Feature #7 tests: extra nutrients (data, estimates, reference values, plan totals, app and PDF).

Run:  python -m pytest tests -q
"""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from src.calculator import UserProfile, calculate_targets
from src.clean_data import fill_micros
from src.config import FOODS_CLEAN, MICRO_IDS
from src.nutrients import MICROS, add_micros, day_micros, micro_targets, status, week_summary


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


def test_food_table_has_every_nutrient(foods):
    assert set(MICROS) <= set(foods.columns) and "estimated" in foods.columns
    assert not foods[MICROS].isna().any().any() and (foods[MICROS] >= 0).all().all()
    lentils = foods[foods["description"].str.startswith("Lentils, mature seeds, cooked, boiled")].iloc[0]
    assert lentils["fiber_g"] > 5 and lentils["iron_mg"] > 2
    chicken = foods[foods["description"].str.contains(r"^Chicken, broilers or fryers, breast, meat only, cooked, roasted")].iloc[0]
    assert chicken["sugar_g"] == 0 and chicken["fiber_g"] == 0
    assert foods["estimated"].fillna("").str.contains("sugar_g").mean() < 0.3      # most values are measured


def test_fill_micros_rules():
    df = pd.DataFrame({
        "description": ["Beef, steak", "Bagels, egg", "Bagels, plain", "Bagels, wheat", "Apples, raw"],
        "food_category": ["Beef Products", "Baked Products", "Baked Products", "Baked Products", "Fruits and Fruit Juices"],
        "carbs_g": [0.0, 50.0, 52.0, 48.0, 14.0], "fat_g": [10.0, 2.0, 1.5, 1.8, 0.2],
        **{m: [np.nan, np.nan, 6.0, 8.0, 10.0] for m in MICRO_IDS},
    })
    out = fill_micros(df, [])
    assert out.loc[0, "sugar_g"] == 0 and out.loc[0, "fiber_g"] == 0               # no carbs -> no sugar / fiber
    assert out.loc[1, "sugar_g"] == 7.0                                             # median of the other bagels
    assert out.loc[1, "estimated"].split("|") == list(MICRO_IDS)
    assert out.loc[2, "estimated"] == ""                                            # measured values untouched
    assert out.loc[2, "sugar_g"] == 6.0


def test_reference_values():
    man, woman, older_woman = micro_targets("male", 30, 2500), micro_targets("female", 30, 2000), micro_targets("female", 55, 1800)
    assert man["fiber_g"]["value"] == 35 and man["fiber_g"]["kind"] == "min"
    assert man["sodium_mg"] == {"value": 2300, "kind": "max", "upper": 2300}
    assert man["potassium_mg"]["value"] == 3400 and woman["potassium_mg"]["value"] == 2600
    assert man["iron_mg"]["value"] == 8 and woman["iron_mg"]["value"] == 18 and older_woman["iron_mg"]["value"] == 8
    assert woman["calcium_mg"]["value"] == 1000 and older_woman["calcium_mg"]["value"] == 1200
    assert man["sat_fat_g"]["value"] == round(0.1 * 2500 / 9)
    assert man["sugar_g"]["kind"] == "info"


@pytest.mark.parametrize("value, target, expected", [
    (40, {"value": 35, "kind": "min"}, "ok"), (33, {"value": 35, "kind": "min"}, "close"),
    (20, {"value": 35, "kind": "min"}, "low"), (2500, {"value": 2300, "kind": "max"}, "high"),
    (2000, {"value": 2300, "kind": "max"}, "ok"), (60, {"value": None, "kind": "info"}, "info"),
    (60, {"value": float("nan"), "kind": "info"}, "info")])          # "no target" after a pandas round trip
def test_status(value, target, expected):
    assert status(value, target) == expected


def test_target_text():
    from src.nutrients import target_text
    assert target_text({"value": 2300, "kind": "max"}, "mg") == "≤ 2,300 mg"
    assert target_text({"value": 35, "kind": "min"}, "g") == "≥ 35 g"
    assert target_text({"value": float("nan"), "kind": "info"}, "g") == ""            # was "≤ nan g"


def test_plan_amounts(foods):
    lentils = int(foods.loc[foods["description"].str.startswith("Lentils, mature seeds, cooked, boiled"), "fdc_id"].iloc[0])
    items = pd.DataFrame({"day_index": [0, 0, 1], "fdc_id": [lentils] * 3, "grams": [200.0, 100.0, 50.0]})
    per100 = foods.set_index("fdc_id").loc[lentils, "fiber_g"]
    assert add_micros(items, foods)["fiber_g"].tolist() == pytest.approx([2 * per100, per100, per100 / 2])
    assert day_micros(items, foods)["fiber_g"].tolist() == pytest.approx([3 * per100, per100 / 2])


def test_week_summary_and_targets_saved(foods):
    from src.planner import generate_week_plan
    user = UserProfile("N", 30, "female", 60, 165, 2, "maintain")
    plan, targets, _ = generate_week_plan(user, foods=foods, seed=1)
    assert targets.micros == micro_targets("female", 30, targets.calories)
    s = week_summary(plan.to_dataframe(), foods, targets.micros)
    assert list(s["nutrient"]) == MICROS and (s["days"] == 7).all()
    assert s.loc[s["nutrient"] == "fiber_g", "average"].iloc[0] > 10
    assert s.loc[s["nutrient"] == "sugar_g", "days_met"].isna().all()               # sugars: shown only


def test_pdf_and_app_show_nutrients(foods, tmp_path, monkeypatch):
    from src.planner import generate_week_plan
    from src.report import week_pdf
    from tests.test_report import text_of
    user = UserProfile("N", 30, "female", 60, 165, 2, "maintain")
    plan, targets, prefs = generate_week_plan(user, foods=foods, seed=1)
    info = {"week_number": 1, "targets": asdict(targets), "prefs": prefs.summary()}
    text = text_of(week_pdf(info, plan.to_dataframe().assign(eaten=None), "N", foods, compress=False))
    assert "Also every day: Fiber" in text and "Calcium" in text and "Sodium" in text

    from src import db
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "t.db"))
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    at.radio[0].set_value("New user").run()
    at.text_input[0].input("Nadia")
    at.text_input[1].input("1234")
    at.text_input[2].input("1234")
    at.button[0].click().run()
    assert not at.exception
    at.session_state["goto"] = "Nutrients"
    at.run()
    assert not at.exception
    assert any("Nutrients this week" in m.value for m in at.markdown)
    assert db.latest_plan(db.list_users()[0]["id"])["targets"]["micros"]["iron_mg"]["value"] == 8   # male default
