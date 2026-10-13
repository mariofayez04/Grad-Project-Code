"""Phase 7 tests: taste vector, penalties, the learning effect in simulation, and the app's "Finish week" flow.

Run:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src import db
from src.calculator import UserProfile
from src.config import EMBEDDINGS, FOODS_CLEAN
from src.feedback import acceptance_rate, food_penalties, meal_status, split_foods, update_taste

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


@pytest.fixture(scope="module")
def emb(foods):
    from src.embeddings import FoodEmbeddings
    return FoodEmbeddings(foods)


def ids(foods, pattern, n=1):
    return foods.loc[foods["description"].str.contains(pattern, regex=True), "fdc_id"].head(n).tolist()


def toy_week(eaten_meal, skipped_meal, week=1):
    """Two meals: one eaten, one skipped (NULL)."""
    rows = [(0, "lunch", fid, 1) for fid in eaten_meal] + [(0, "dinner", fid, None) for fid in skipped_meal]
    df = pd.DataFrame(rows, columns=["day_index", "slot", "fdc_id", "eaten"])
    df["name"] = df["fdc_id"].astype(str)
    df["week_number"] = week
    return df


def test_unanswered_counts_as_skipped():
    week = toy_week([1, 2], [3])
    assert meal_status(week)["eaten"].tolist() == [False, True]  # dinner (NULL) skipped, lunch eaten
    assert acceptance_rate(week) == 0.5
    assert split_foods(week) == ([1, 2], [3])


def test_taste_moves_towards_eaten(foods, emb):
    chicken = ids(foods, r"^Chicken, broiler.*breast.*grilled")
    fish = ids(foods, r"^Fish, salmon", 3)
    old = emb.mean_vector(ids(foods, r"^Rice, white", 3))
    new = update_taste(old, emb, eaten_ids=chicken, skipped_ids=fish)
    assert np.linalg.norm(new) == pytest.approx(1.0, abs=1e-5)
    assert new @ emb.vector(chicken[0]) > old @ emb.vector(chicken[0])
    assert new @ emb.vector(fish[0]) < old @ emb.vector(fish[0])


def test_taste_from_nothing(foods, emb):
    assert update_taste(None, emb, [], []) is None
    v = update_taste(None, emb, ids(foods, r"^Lentils"), [])
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_penalties_and_decay():
    old = toy_week([10], [20], week=1)
    recent = toy_week([20], [10], week=2)  # tastes changed: now 10 is skipped, 20 eaten
    p = food_penalties(pd.concat([old, recent]))
    assert p[10] > p[20]                 # the recent week counts more
    assert p[10] == pytest.approx(0.8 - 0.7 * 0.2)   # skipped now (0.8) - eaten a week ago (0.2 * decay 0.7)
    assert p[20] == pytest.approx(0.7 * 0.8 - 0.2)   # skipped a week ago - eaten now


def test_family_penalty_spreads_to_same_food(foods):
    beef = ids(foods, r"^Beef, .*cooked", 4)
    rice = ids(foods, r"^Rice, white", 1)
    # beef meals always skipped; rice eaten in other meals and skipped once alongside beef
    rows = [(d, "dinner", f, None) for d, f in enumerate(beef[:3])] + [(0, "dinner", rice[0], None)]
    rows += [(d, "lunch", rice[0], 1) for d in range(3)]
    week = pd.DataFrame(rows, columns=["day_index", "slot", "fdc_id", "eaten"]).assign(name="x", week_number=1)
    p = food_penalties(week, foods)
    assert p[beef[3]] > 0                  # a beef cut never shown is still penalised (same family)
    # rice was mostly eaten -> no family penalty, only its own small score: 1 skip * 0.8 - 3 eats * 0.2
    assert p[rice[0]] == pytest.approx(0.2)


def test_simulation_learns_hidden_dislike(foods, emb):
    from src.simulate import SCENARIOS, run
    user = UserProfile("Sim", 30, "male", 75, 175, 2, "maintain", "")
    rx = SCENARIOS["dislikes fish & seafood"]
    with_fb = run(user, rx, emb, foods, weeks=3, feedback=True, seed=0)
    without = run(user, rx, emb, foods, weeks=3, feedback=False, seed=0)
    assert with_fb[-1]["acceptance"] > with_fb[0]["acceptance"]
    assert with_fb[-1]["disliked_items"] < with_fb[0]["disliked_items"]
    assert with_fb[-1]["acceptance"] >= without[-1]["acceptance"]


def test_app_finish_week(tmp_path, monkeypatch):
    """Tick a meal, press 'Finish week' -> a week 2 plan exists and week 1 is finished."""
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "t.db"))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    at.text_input[0].input("Laila")
    at.button[0].click().run()
    at.checkbox[0].check().run()
    finish = next(b for b in at.button if "Finish week" in b.label)
    finish.click().run()
    assert not at.exception
    uid = db.list_users()[0]["id"]
    plans = db.list_plans(uid)
    assert [p["week_number"] for p in plans] == [1, 2] and plans[0]["finished"] == 1
    assert db.latest_plan(uid)["taste_vector"] is not None
    assert any("What I learned" in s.value for s in at.subheader)
