"""Feature #5 tests: "New meal" / "New day" alternatives, the before / after choice, and learning from it.

Run:  python -m pytest tests -q
"""
import pandas as pd
import pytest

from src import db
from src.calculator import UserProfile, calculate_targets
from src.config import EMBEDDINGS, FOODS_CLEAN
from src.feedback import CHOICE_WEIGHT, EAT_BONUS, SKIP_PENALTY, food_penalties, next_week_inputs, week_summary
from src.parser import apply_to_foods, parse_preferences
from src.planner import MACROS, MealItem, MealPlanner, daily_targets

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")
USER = UserProfile("Regen", 30, "male", 75, 175, 2, "maintain", "I'm allergic to peanuts")


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


@pytest.fixture(scope="module")
def emb(foods):
    from src.embeddings import FoodEmbeddings
    return FoodEmbeddings(foods)


@pytest.fixture(scope="module")
def setup(foods, emb):
    targets = calculate_targets(USER)
    allowed, _ = apply_to_foods(foods, parse_preferences(USER.preferences_text))
    plan = MealPlanner(allowed, emb, None, None, seed=1).plan_week(targets)
    return allowed, plan, plan.to_dataframe(), daily_targets(targets)


def is_oil(fid, foods):
    return foods.set_index("fdc_id").loc[fid, "description"].startswith(("Oil", "Butter", "Ghee"))


@pytest.mark.parametrize("slot", ["breakfast", "lunch", "dinner", "snack"])
def test_new_meal_is_different_and_fits_the_day(foods, emb, setup, slot):
    allowed, _, items, daily = setup
    meal = MealPlanner(allowed, emb, None, None, seed=7).replan_meal(items, 2, slot, daily)
    current = items[(items["day_index"] == 2) & (items["slot"] == slot)]
    new_ids = [i.fdc_id for i in meal.items]
    assert meal.items and not set(new_ids) & set(current["fdc_id"])
    base = foods.set_index("fdc_id")["base_name"]
    day_bases = set(base[items.loc[items["day_index"] == 2, "fdc_id"]])
    assert not {base[f] for f in new_ids if not is_oil(f, foods)} & day_bases      # nothing already on that day
    others = items[(items["day_index"] == 2) & (items["slot"] != slot)]
    day_kcal = others["calories"].sum() + meal.totals()["calories"]
    assert abs(day_kcal - daily["calories"]) / daily["calories"] < 0.10
    assert set(new_ids) <= set(allowed["fdc_id"])                                    # allergies still respected


def test_new_day_is_different_and_on_target(foods, emb, setup):
    allowed, _, items, daily = setup
    day = MealPlanner(allowed, emb, None, None, seed=7).replan_day(items, 4, daily)
    new_ids = [i.fdc_id for m in day.meals for i in m.items]
    current = items.loc[items["day_index"] == 4, "fdc_id"]
    base = foods.set_index("fdc_id")["base_name"]
    assert not set(new_ids) & set(current)
    assert not {base[f] for f in new_ids if not is_oil(f, foods)} & {base[f] for f in current if not is_oil(f, foods)}
    assert [m.slot for m in day.meals] == ["breakfast", "lunch", "dinner", "snack"]
    for m in MACROS:
        assert abs(day.totals()[m] - daily[m]) / daily[m] < 0.15


# --- storage --------------------------------------------------------------------------------------------
@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


def test_replace_meal_and_choices(temp_db, setup):
    _, plan, items, _ = setup
    targets = calculate_targets(USER)
    uid = db.create_user(USER, "1234")
    pid = db.save_plan(uid, plan, targets, parse_preferences(USER.preferences_text))
    db.set_meal_eaten(pid, 0, "lunch", True)
    old = db.plan_items(pid).query("day_index == 0 and slot == 'lunch'")
    new = [MealItem(111, "Test chicken", 150.0, 250.0, 40.0, 0.0, 8.0), MealItem(222, "Oil, olive", 10.0, 88.0, 0.0, 0.0, 10.0)]
    db.replace_meal(pid, 0, "Monday", "lunch", new)
    lunch = db.plan_items(pid).query("day_index == 0 and slot == 'lunch'")
    assert lunch["fdc_id"].tolist() == [111, 222] and lunch["eaten"].isna().all()   # a new meal is not ticked yet
    assert len(db.plan_items(pid)) == len(items) - len(old) + 2

    old_foods = list(zip(old["fdc_id"], old["name"]))
    db.record_choice(pid, 0, "lunch", "new", old_foods, [(111, "Test chicken"), (222, "Oil, olive")])
    db.record_choice(pid, 1, None, "current", [(5, "Kept")], [(6, "Not kept")])
    with pytest.raises(ValueError):
        db.record_choice(pid, 1, None, "maybe", [], [])
    assert db.user_choices(uid).empty                                                   # week not finished yet
    c = db.user_choices(uid, finished_only=False)
    assert [f["fdc_id"] for f in c.loc[0, "chosen"]] == [111, 222]
    assert [f["fdc_id"] for f in c.loc[0, "rejected"]] == old["fdc_id"].tolist()
    assert c.loc[1, "chosen"] == [{"fdc_id": 5, "name": "Kept"}] and c.loc[1, "rejected"] == [{"fdc_id": 6, "name": "Not kept"}]
    assert pd.isna(c.loc[1, "slot"])


# --- learning -------------------------------------------------------------------------------------------
def choice_rows(week=1):
    return pd.DataFrame({"week_number": [week], "kept": ["new"], "slot": ["lunch"],
                         "chosen": [[{"fdc_id": 10, "name": "Chicken"}, {"fdc_id": 99, "name": "Oil, olive"}]],
                         "rejected": [[{"fdc_id": 20, "name": "Veal"}, {"fdc_id": 98, "name": "Oil, corn"}]]})


def one_week(week=1):
    return pd.DataFrame({"day_index": [0], "slot": ["dinner"], "fdc_id": [30], "eaten": [1], "name": ["Rice"],
                         "week_number": [week]})


def test_choice_penalties():
    p = food_penalties(one_week(), choices=choice_rows())
    assert p[20] == pytest.approx(CHOICE_WEIGHT * SKIP_PENALTY)        # not kept: half a skip
    assert p[10] == pytest.approx(-CHOICE_WEIGHT * EAT_BONUS)          # kept: half an eaten meal
    assert 98 not in p and 99 not in p                                 # cooking oil says nothing about taste
    older = food_penalties(pd.concat([one_week(1), one_week(2)]), choices=choice_rows(1))
    assert older[20] == pytest.approx(0.7 * CHOICE_WEIGHT * SKIP_PENALTY)  # older weeks count less


def test_choice_moves_taste(foods, emb):
    ids = lambda pat: int(foods.loc[foods["description"].str.contains(pat), "fdc_id"].iloc[0])
    chicken, veal, rice = ids(r"^Chicken, broiler.*breast.*roasted"), ids(r"^Veal, shoulder"), ids(r"^Rice, white")
    week = one_week().assign(fdc_id=rice)
    rows = choice_rows().assign(chosen=[[{"fdc_id": chicken, "name": "Chicken"}]], rejected=[[{"fdc_id": veal, "name": "Veal"}]])
    plain, _ = next_week_inputs(None, emb, week, week)
    learned, _ = next_week_inputs(None, emb, week, week, choices=rows)
    assert learned @ emb.vector(chicken) > plain @ emb.vector(chicken)
    assert learned @ emb.vector(veal) - learned @ emb.vector(chicken) < plain @ emb.vector(veal) - plain @ emb.vector(chicken)
    s = week_summary(week, choices=rows)
    assert s["not_chosen"] == [("Veal", 1)] and s["chosen"] == [("Chicken", 1)]


def test_app_new_meal_compare_and_keep(temp_db):
    """In the app: 'New meal' opens the comparison; 'Use the new one' replaces the meal and records the choice."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    at.radio[0].set_value("New user").run()
    at.text_input[0].input("Rana")
    at.text_input[1].input("1234")
    at.text_input[2].input("1234")
    at.button[0].click().run()
    uid = db.list_users()[0]["id"]
    pid = db.latest_plan(uid)["id"]
    before = db.plan_items(pid).query("day_index == 0 and slot == 'lunch'")["fdc_id"].tolist()

    next(b for b in at.button if b.key == f"newmeal_{pid}_0_lunch").click().run()
    assert not at.exception and "proposal" in at.session_state
    next(b for b in at.button if b.key == "keep_new").click().run()
    assert not at.exception and "proposal" not in at.session_state
    after = db.plan_items(pid).query("day_index == 0 and slot == 'lunch'")["fdc_id"].tolist()
    assert after and not set(after) & set(before)
    c = db.user_choices(uid, finished_only=False)
    assert c["kept"].tolist() == ["new"] and [f["fdc_id"] for f in c.loc[0, "rejected"]] == before

    next(b for b in at.button if b.key == f"newday_{pid}_3").click().run()          # whole day, keep current
    day3 = db.plan_items(pid).query("day_index == 3")["fdc_id"].tolist()
    next(b for b in at.button if b.key == "keep_current").click().run()
    assert db.plan_items(pid).query("day_index == 3")["fdc_id"].tolist() == day3
    assert db.user_choices(uid, finished_only=False)["kept"].tolist() == ["new", "current"]
