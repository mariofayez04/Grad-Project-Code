"""Feature #2 tests: real-user evaluation tools (held-out sentence sheet, "Rate this week" survey, pilot data).

Run:  python -m pytest tests -q
"""
import pandas as pd
import pytest

from src import db
from src.calculator import UserProfile
from src.config import EMBEDDINGS, FOODS_CLEAN
from src.heldout import COLUMNS, EXAMPLES, load_sentences, make_template, score

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")


# --- held-out sentences ---------------------------------------------------------------------------------------
def test_template_and_loading(tmp_path):
    path = make_template(tmp_path / "sheet.xlsx")
    assert load_sentences(path).empty                                  # only the example rows so far
    sheet = pd.read_excel(path, sheet_name=None)
    assert list(sheet["Sentences"].columns) == COLUMNS and len(sheet["Sentences"]) == len(EXAMPLES)
    assert "Instructions" in sheet
    filled = pd.concat([sheet["Sentences"], pd.DataFrame([("I hate okra", "", "okra", "", "Mona"), ("", "", "", "", "")],
                                                         columns=COLUMNS)])
    filled.to_excel(path, sheet_name="Sentences", index=False)
    rows = load_sentences(path)
    assert rows["sentence"].tolist() == ["I hate okra"]                # examples and empty rows left out


def test_scoring():
    df = pd.DataFrame([
        ("I'm allergic to peanuts and I don't like broccoli", "peanuts", "broccoli", "", "A"),
        ("seafood is a no for me, love lentils", "seafood", "", "lentils", "B"),
        ("Everything is fine for me", "shrimp", "", "", "B"),           # the parser can't know -> a miss
    ], columns=COLUMNS)
    rows, m = score(df)
    assert m["sentences"] == 3 and m["writers"] == 2
    assert rows.loc[0, ["never_found", "less_often_found"]].tolist() == [1, 1]
    assert rows.loc[1, "more_often_found"] == 1 and rows.loc[2, "never_missed"] == "shrimp"
    assert m["allergy_items"] == 3 and m["allergy_recall"] == pytest.approx(2 / 3)
    assert m["never_recall"] == pytest.approx(2 / 3)


# --- survey and pilot data ------------------------------------------------------------------------------------
@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


def make_week(user_name="P"):
    from src.planner import generate_week_plan
    u = UserProfile(user_name, 30, "female", 62, 165, 2, "maintain")
    uid = db.create_user(u, "1234")
    plan, targets, prefs = generate_week_plan(u, seed=1)
    return uid, db.save_plan(uid, plan, targets, prefs)


def test_survey(temp_db):
    _, pid = make_week()
    db.save_survey(pid, 4, 3, 5, "yes", " tasty ")
    db.save_survey(pid, 5, 3, 5, "maybe")                              # updating keeps one row per week
    s = db.get_survey(pid)
    assert (s["liked"], s["would_use"], s["comment"]) == (5, "maybe", "")
    with pytest.raises(ValueError):
        db.save_survey(pid, 6, 3, 3, "yes")
    with pytest.raises(ValueError):
        db.save_survey(pid, 3, 3, 3, "sure")


def test_pilot_data(temp_db, tmp_path):
    uid, pid = make_week()
    db.set_meal_eaten(pid, 0, "lunch", True)
    db.save_survey(pid, 4, 4, 4, "yes", "more fish please")
    assert db.pilot_data()["items"].empty                              # only finished weeks count
    db.mark_plan_finished(pid)
    data = db.pilot_data()
    assert data["users"] == 1 and data["plans"] == 1 and set(data["items"]["user_id"]) == {uid}
    assert data["surveys"]["liked"].tolist() == [4]
    assert "name" not in data["items"].columns and "name" not in data["surveys"].columns   # no names
    from src.evaluate import eval_pilot
    _, weeks = eval_pilot(out=tmp_path)                               # never write test data into data/evaluation
    assert weeks["eaten"].tolist() == [pytest.approx(1 / 28)]
    assert pd.read_csv(tmp_path / "pilot_comments.csv")["comment"].tolist() == ["more fish please"]


def test_app_rate_this_week(temp_db):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    at.radio[0].set_value("New user").run()
    at.checkbox(key="pf_consent").check()
    at.text_input[0].input("Rania")
    at.text_input[1].input("1234")
    at.text_input[2].input("1234")
    at.button[0].click().run()
    at.session_state["goto"] = "Progress"
    at.run()
    assert any("Rate this week" in e.label for e in at.expander)
    next(s for s in at.select_slider if s.label.startswith("How much did you like")).set_value(5)
    next(r for r in at.radio if r.label.startswith("Would you keep")).set_value("yes")
    next(b for b in at.button if b.label == "Save my rating").click().run()
    assert not at.exception
    pid = db.latest_plan(db.list_users()[0]["id"])["id"]
    assert db.get_survey(pid)["liked"] == 5 and db.get_survey(pid)["would_use"] == "yes"


# --- simulated testers --------------------------------------------------------------------------------------------
def test_simulated_tester_runs_on_a_throwaway_database(tmp_path, monkeypatch):
    from src.embeddings import FoodEmbeddings
    from src.testers import TESTERS, food_texts, run_tester
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "testers.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.init_db()
    foods = pd.read_csv(FOODS_CLEAN)
    rows = run_tester(1, TESTERS[0], foods, FoodEmbeddings(foods), food_texts(foods), True, weeks=2)
    assert [r["week"] for r in rows] == [1, 2]
    assert all(r["signed_in"] and r["wrong_pin_refused"] for r in rows)
    assert all(r["allergy_found"] and r["allergen_items"] == 0 for r in rows)  # this tester states a shrimp allergy
    assert len(db.list_plans(db.list_users()[0]["id"])) == 2
