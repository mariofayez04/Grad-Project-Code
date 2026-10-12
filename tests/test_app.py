"""Phase 6 tests: SQLite storage and a headless run of the Streamlit app.

Run:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src import db
from src.calculator import UserProfile
from src.config import EMBEDDINGS, FOODS_CLEAN

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    return tmp_path / "test.db"


@pytest.fixture(scope="module")
def week():
    from src.embeddings import FoodEmbeddings
    from src.planner import generate_week_plan
    foods = pd.read_csv(FOODS_CLEAN)
    user = UserProfile("Sara", 28, "female", 60, 165, 2, "lose", "I hate fish")
    plan, targets, prefs = generate_week_plan(user, foods=foods, embeddings=FoodEmbeddings(foods), seed=1)
    return user, plan, targets, prefs


def test_save_and_update_user(temp_db, week):
    user = week[0]
    uid = db.save_user(user)
    assert db.get_user(uid)["weight_kg"] == 60
    user.weight_kg = 58
    assert db.save_user(user) == uid  # same name -> update, not a new row
    assert db.get_user(uid)["weight_kg"] == 58
    assert [u["name"] for u in db.list_users()] == ["Sara"]


def test_plan_roundtrip(temp_db, week):
    user, plan, targets, prefs = week
    uid = db.save_user(user)
    taste = np.random.default_rng(0).random(384).astype(np.float32)
    pid = db.save_plan(uid, plan, targets, prefs, seed=1, taste_vector=taste)
    saved = db.latest_plan(uid)
    assert saved["id"] == pid and saved["week_number"] == 1
    assert saved["targets"]["calories"] == targets.calories
    assert saved["prefs"]["avoid"] == ["fish"]
    assert np.allclose(saved["taste_vector"], taste)
    items = db.plan_items(pid)
    assert len(items) == len(plan.to_dataframe())
    assert items["eaten"].isna().all()
    # second plan gets week 2
    db.save_plan(uid, plan, targets, prefs)
    assert db.latest_plan(uid)["week_number"] == 2


def test_meal_checkbox(temp_db, week):
    user, plan, targets, prefs = week
    pid = db.save_plan(db.save_user(user), plan, targets, prefs)
    db.set_meal_eaten(pid, 0, "lunch", True)
    db.set_meal_eaten(pid, 1, "dinner", False)
    items = db.plan_items(pid)
    lunch = items[(items.day_index == 0) & (items.slot == "lunch")]
    dinner = items[(items.day_index == 1) & (items.slot == "dinner")]
    assert (lunch["eaten"] == 1).all() and (dinner["eaten"] == 0).all()
    assert items.loc[~items.index.isin(lunch.index.union(dinner.index)), "eaten"].isna().all()


def test_app_generates_plan(temp_db):
    """Run the real app headless: fill the form, submit, and check the plan appears."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception
    at.text_input[0].input("Omar")
    at.text_area[0].input("I'm allergic to peanuts and I love lentils")
    at.button[0].click().run()
    assert not at.exception
    assert any("Week 1 meal plan" in s.value for s in at.subheader)
    assert len(at.checkbox) == 28
    plan = db.latest_plan(db.list_users()[0]["id"])
    assert "peanut" in plan["prefs"]["block_allergen_groups"]
