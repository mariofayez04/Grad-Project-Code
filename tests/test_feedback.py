"""Phase 7 tests: taste vector, penalties, the learning effect in simulation, and the app's "Finish week" flow.

Run:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src import db
from src.calculator import UserProfile
from src.config import EMBEDDINGS, FOODS_CLEAN
from src.feedback import (acceptance_rate, food_penalties, meal_status, next_week_inputs, split_foods, update_taste,
                          week_summary)

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
    at.radio[0].set_value("New user").run()
    at.text_input[0].input("Laila")
    at.text_input[1].input("1234")
    at.text_input[2].input("1234")
    at.button[0].click().run()
    at.checkbox[0].check().run()
    at.session_state["goto"] = "Progress"
    at.run()
    finish = next(b for b in at.button if "Finish week" in b.label)
    finish.click().run()
    assert not at.exception
    uid = db.list_users()[0]["id"]
    plans = db.list_plans(uid)
    assert [p["week_number"] for p in plans] == [1, 2] and plans[0]["finished"] == 1
    assert db.latest_plan(uid)["taste_vector"] is not None
    assert any("What I learned" in s.value for s in at.subheader)


def test_swapped_food_counts_as_skipped(foods, emb):
    """A food swapped away is penalised next week even though its meal was eaten, and it moves the taste vector."""
    chicken, rice = ids(foods, r"^Chicken, broiler.*breast.*roasted")[0], ids(foods, r"^Rice, white")[0]
    week = toy_week([chicken, rice], [], week=1)  # lunch eaten, after swapping something away
    swaps = pd.DataFrame({"old_fdc_id": [chicken], "old_name": ["Chicken breast"], "week_number": [1]})
    without = food_penalties(week)
    with_swaps = food_penalties(week, swaps=swaps)
    assert with_swaps[chicken] == pytest.approx(without[chicken] + 0.8)
    assert with_swaps[rice] == without[rice]

    taste_plain, _ = next_week_inputs(None, emb, week, week)
    taste_swap, _ = next_week_inputs(None, emb, week, week, swaps)
    chicken_vec = emb.vector(chicken)
    assert taste_swap @ chicken_vec < taste_plain @ chicken_vec
    assert week_summary(week, swaps=swaps)["swapped_away"] == [("Chicken breast", 1)]


def test_old_swaps_do_not_move_this_weeks_taste(foods, emb):
    chicken = ids(foods, r"^Chicken, broiler.*breast.*roasted")[0]
    history = pd.concat([toy_week([chicken], [], week=1), toy_week([chicken], [], week=2)])
    old_swap = pd.DataFrame({"old_fdc_id": [chicken], "old_name": ["x"], "week_number": [1]})
    last = history[history["week_number"] == 2]
    assert np.allclose(next_week_inputs(None, emb, last, history, old_swap)[0], next_week_inputs(None, emb, last, history)[0])


def dish_id(foods, name):
    return int(foods.loc[foods["description"].str.startswith(name + ","), "fdc_id"].iloc[0])


def test_dishes_belong_to_their_ingredient_families(foods):
    from src.feedback import food_families
    fams = food_families(foods)
    assert {"beef", "okra", "rice"} <= fams[dish_id(foods, "Bamya with rice")]
    assert "fish" in fams[dish_id(foods, "Sayadiya")] and "egg" in fams[dish_id(foods, "Shakshuka")]
    assert not fams[dish_id(foods, "Koshari")] & {"beef", "chicken", "fish"}
    assert "oil" not in fams[dish_id(foods, "Koshari")] and "soup" not in fams[dish_id(foods, "Molokhia with rice and chicken")]


def test_family_penalty_reaches_dishes(foods):
    """Beef skipped more than average -> dishes made with beef are pushed down too, other dishes are not."""
    beef = ids(foods, r"^Beef, .*cooked", 3)
    rice = ids(foods, r"^Rice, white", 1)
    rows = [(d, "dinner", f, None) for d, f in enumerate(beef)] + [(d, "lunch", rice[0], 1) for d in range(3)]
    week = pd.DataFrame(rows, columns=["day_index", "slot", "fdc_id", "eaten"]).assign(name="x", week_number=1)
    p = food_penalties(week, foods)
    assert p.get(dish_id(foods, "Bamya with rice"), 0) > 0 and p.get(dish_id(foods, "Kofta"), 0) > 0
    assert dish_id(foods, "Koshari") not in p and dish_id(foods, "Ful medames") not in p


def test_skipped_dish_counts_for_its_ingredients(foods):
    """Skipping beef dishes again and again also pushes down a plain beef cut never shown."""
    dishes = [dish_id(foods, n) for n in ("Bamya with rice", "Kofta", "Hawawshi")]
    lentils = ids(foods, r"^Lentils, mature seeds, cooked", 1)[0]
    rows = [(d, "dinner", f, None) for d, f in enumerate(dishes)] + [(d, "lunch", lentils, 1) for d in range(3)]
    week = pd.DataFrame(rows, columns=["day_index", "slot", "fdc_id", "eaten"]).assign(name="x", week_number=1)
    p = food_penalties(week, foods)
    steak = ids(foods, r"^Beef, .*steak.*cooked", 1)[0]
    assert p.get(steak, 0) > 0 and lentils in p and p[lentils] < 0         # beef down, eaten lentils up
