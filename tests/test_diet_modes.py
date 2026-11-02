"""Feature #10 tests: nutrient-aware planning and the diabetes-friendly / low-salt diet modes.

Run:  python -m pytest tests -q
"""
from dataclasses import asdict

import pandas as pd
import pytest

from src import db
from src.calculator import UserProfile, calculate_targets, macro_split
from src.config import EMBEDDINGS, FOODS_CLEAN
from src.nutrients import (MODE_SODIUM_MG, SALTY_MG_PER_100G, filter_modes, micro_targets, nutrient_score, parse_modes,
                           week_summary)
from src.planner import MealPlanner, generate_week_plan

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")


def person(modes="", goal="maintain"):
    return UserProfile("D", 50, "female", 72, 162, 1, goal, "", 4, modes)


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


def test_parse_and_validate_modes():
    assert parse_modes("diabetes|low_sodium") == {"diabetes", "low_sodium"}
    assert parse_modes("") == set() and parse_modes(None) == set() and parse_modes(["diabetes", "x"]) == {"diabetes"}
    with pytest.raises(ValueError, match="health mode"):
        calculate_targets(person("keto"))


def test_diabetes_caps_carbs_and_sugar():
    plain, diab = calculate_targets(person()), calculate_targets(person("diabetes"))
    assert diab.calories == plain.calories and diab.protein_g == plain.protein_g
    assert 4 * diab.carbs_g <= 0.40 * diab.calories + 4 and diab.fat_g > plain.fat_g
    sugar_cap = round(0.10 * diab.calories / 4)
    assert diab.micros["sugar_g"] == {"value": sugar_cap, "kind": "max", "upper": sugar_cap}
    assert plain.micros["sugar_g"]["kind"] == "info"
    assert any("not a treatment" in w for w in diab.warnings) and not any("treatment" in w for w in plain.warnings)
    m = macro_split(2000, "maintain", diabetes=True)                       # 25 / 40 / 35
    assert (m["protein_g"], m["carbs_g"], m["fat_g"]) == (125, 200, 78)


def test_low_sodium_target():
    assert micro_targets("male", 40, 2500, "low_sodium")["sodium_mg"]["value"] == MODE_SODIUM_MG
    assert micro_targets("male", 40, 2500)["sodium_mg"]["value"] == 2300


def test_filters(foods):
    salty = filter_modes(foods, "low_sodium")
    assert (salty["sodium_mg"] <= SALTY_MG_PER_100G).all()
    assert not salty["description"].str.contains(r"\b(?:cured|pickled|salted)\b", case=False).any()
    sweet = filter_modes(foods, "diabetes")
    assert not sweet["description"].str.contains(r"\b(?:syrup|candied|frosted)\b", case=False).any()
    assert not ((sweet["sugar_g"] > 15) & (sweet["meal_role"] != "fruit")).any()
    assert (sweet["meal_role"] == "fruit").sum() > 100                      # whole fruit stays
    assert len(filter_modes(foods, "")) == len(foods)


def test_nutrient_score_direction(foods):
    t = micro_targets("female", 50, 1800)
    s = nutrient_score(foods, t, 1800).set_axis(foods["description"])
    by = lambda pattern: s[s.index.str.contains(pattern, regex=True)].iloc[0]
    assert by(r"^Lentils, mature seeds, cooked, boiled") > by(r"^Rice, white, long-grain, regular, enriched, cooked")
    assert by(r"^Yogurt, plain, low fat") > 0                              # calcium
    salty = nutrient_score(foods, t, 1800, "low_sodium").set_axis(foods["description"])
    cheese = salty[salty.index.str.contains(r"^Cheese, feta")].iloc[0]
    assert cheese < s[s.index.str.contains(r"^Cheese, feta")].iloc[0]      # the mode pushes salty food down


def compare(foods, modes, nutrient):
    """(value with the nutrient-aware planner in this mode, value with macros only), averaged over 2 weeks."""
    out = []
    for nutrition in (True, False):
        vals = []
        for seed in (1, 2):
            u = person(modes)
            targets = calculate_targets(u)
            from src.parser import apply_to_foods, parse_preferences
            allowed, _ = apply_to_foods(foods, parse_preferences(""))
            plan = MealPlanner(allowed, None, None, None, seed, nutrition=targets if nutrition else None).plan_week(targets)
            s = week_summary(plan.to_dataframe(), foods, targets.micros).set_index("nutrient")
            vals.append(s.loc[nutrient, "average"])
            assert all(err < 10 for err in plan.adherence().values())       # macros still on target
        out.append(sum(vals) / len(vals))
    return out


def test_planner_aims_for_calcium(foods):
    aware, plain = compare(foods, "", "calcium_mg")
    assert aware > plain


def test_low_sodium_plan(foods):
    aware, plain = compare(foods, "low_sodium", "sodium_mg")
    assert aware < plain and aware <= MODE_SODIUM_MG


def test_diabetes_plan(foods):
    aware, plain = compare(foods, "diabetes", "sugar_g")
    assert aware < plain
    plan, targets, _ = generate_week_plan(person("diabetes"), foods=foods, seed=3)
    items = plan.to_dataframe().merge(foods[["fdc_id", "description"]], on="fdc_id")
    assert not items["description"].str.contains(r"\b(?:dried|sweetened|syrup)\b", case=False).any()


# --- profile, app, PDF ------------------------------------------------------------------------------------
@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


def test_profile_and_pdf_keep_modes(temp_db, foods):
    uid = db.create_user(person("diabetes|low_sodium"), "1234")
    assert db.get_user(uid)["health_modes"] == "diabetes|low_sodium"
    plan, targets, prefs = generate_week_plan(person("diabetes|low_sodium"), foods=foods, seed=1)
    pid = db.save_plan(uid, plan, targets, prefs)
    saved = db.latest_plan(uid)["targets"]
    assert saved["health_modes"] == "diabetes|low_sodium" and saved["micros"]["sodium_mg"]["value"] == MODE_SODIUM_MG
    from src.report import week_pdf
    from tests.test_report import text_of
    info = {"week_number": 1, "targets": asdict(targets), "prefs": prefs.summary()}
    text = text_of(week_pdf(info, db.plan_items(pid), "D", foods, compress=False))
    assert "Diet mode: Diabetes-friendly" in text and "Low-salt" in text and "not a treatment" in text


def test_app_diet_mode(temp_db):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    at.radio[0].set_value("New user").run()
    at.checkbox(key="pf_consent").check()
    at.checkbox(key="pfallergy_none").check()
    at.text_input[0].input("Dina")
    at.text_input[1].input("1234")
    at.text_input[2].input("1234")
    next(c for c in at.checkbox if "Low-salt" in c.label).check()
    at.button[0].click().run()
    assert not at.exception
    assert any("Diet mode" in m.value for m in at.markdown)
    assert any("not a treatment" in w.value for w in at.warning)
    assert db.latest_plan(db.list_users()[0]["id"])["targets"]["health_modes"] == "low_sodium"


@pytest.mark.parametrize("modes", ["", "diabetes|low_sodium"])
def test_daily_caps(foods, modes):
    """Iron and calcium stay under the safe upper limits; fiber under twice its target; sodium under its limit."""
    for u in (person(modes), UserProfile("Y", 25, "male", 85, 185, 3, "gain", "", 4, modes)):
        plan, targets, _ = generate_week_plan(u, foods=foods, seed=7)
        from src.nutrients import day_micros
        days = day_micros(plan.to_dataframe(), foods)
        # safety limits (iron, calcium) and sodium: 3% leeway for the 5 g portion rounding; fiber's cap is only a
        # comfort cap (no official upper limit), so 10%
        for n, leeway in [("iron_mg", 1.03), ("calcium_mg", 1.03), ("sodium_mg", 1.03), ("fiber_g", 1.10)]:
            assert (days[n] <= targets.micros[n]["upper"] * leeway).all(), (n, days[n].max(), targets.micros[n]["upper"])


def test_caps_saved_with_targets():
    t = micro_targets("female", 55, 1800, "low_sodium")
    assert t["iron_mg"]["upper"] == 45 and t["calcium_mg"]["upper"] == 2000 and t["sodium_mg"]["upper"] == 1500
    assert t["fiber_g"]["upper"] == 2 * t["fiber_g"]["value"] and t["potassium_mg"]["upper"] is None
