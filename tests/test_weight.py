"""Feature #6 tests: weight tracking (storage, pace, expected change, targets that follow the weight, app).

Run:  python -m pytest tests -q
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from src import db
from src.calculator import UserProfile, calculate_targets
from src.config import EMBEDDINGS
from src.progress import expected_line, expected_weekly_change, pace_warning, weekly_rate

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")


def days_ago(n):
    return date.today() - timedelta(days=n)


def history(weights, step=7):
    """Weights logged every `step` days, the last one today."""
    n = len(weights)
    return pd.DataFrame({"day": [days_ago(step * (n - 1 - i)).isoformat() for i in range(n)], "weight_kg": weights})


# --- calculations --------------------------------------------------------------------------------------------
def test_expected_weekly_change():
    assert expected_weekly_change(2200, 2700) == pytest.approx(-500 * 7 / 7700)    # about -0.45 kg / week
    assert expected_weekly_change(3000, 2700) > 0 and expected_weekly_change(2700, 2700) == 0


def test_weekly_rate():
    assert weekly_rate(history([80])) is None                                     # one entry
    assert weekly_rate(history([80, 79.8], step=3)) is None                        # less than a week apart
    assert weekly_rate(history([80, 79.5, 79.0, 78.5])) == pytest.approx(-0.5)     # straight line
    old_then_recent = pd.concat([history([90, 89], step=30).assign(day=lambda d: [days_ago(200).isoformat(),
                                                                                   days_ago(170).isoformat()]),
                                 history([80, 80.5, 81])])
    assert weekly_rate(old_then_recent) == pytest.approx(0.5)                      # only the last 4 weeks count


def test_pace_warning():
    assert pace_warning(None, 80) is None and pace_warning(-0.5, 80) is None
    assert "faster than the usual safe pace" in pace_warning(-1.2, 80)            # > 1% of 80 kg
    assert "muscle" in pace_warning(0.6, 80)                                       # > 0.5% of 80 kg


def test_expected_line():
    line = expected_line(history([80, 79.6]), 80, -0.7)
    assert line["day"].iloc[0] == pd.Timestamp(days_ago(7)) and line["expected_kg"].iloc[0] == 80
    assert line["expected_kg"].iloc[-1] == pytest.approx(79.3)


# --- storage --------------------------------------------------------------------------------------------------
@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "test.db"))
    db.init_db()


def person(weight=80.0):
    return UserProfile("W", 35, "male", weight, 178, 2, "lose")


def test_log_weight(temp_db):
    uid = db.create_user(person(80), "1234")
    assert db.weight_history(uid)["weight_kg"].tolist() == [80]                   # the starting weight
    db.log_weight(uid, 81, days_ago(10))
    db.log_weight(uid, 78.5)                                                       # today, replaces the first entry
    h = db.weight_history(uid)
    assert h["weight_kg"].tolist() == [81, 78.5] and h["day"].tolist() == [days_ago(10).isoformat(), date.today().isoformat()]
    assert db.get_user(uid)["weight_kg"] == 78.5                                   # profile follows the latest entry
    db.log_weight(uid, 90, days_ago(30))                                           # an older entry doesn't change it
    assert db.get_user(uid)["weight_kg"] == 78.5
    db.delete_weight(uid, date.today().isoformat())
    assert db.get_user(uid)["weight_kg"] == 81
    with pytest.raises(ValueError, match="future"):
        db.log_weight(uid, 80, date.today() + timedelta(days=1))
    with pytest.raises(ValueError, match="between"):
        db.log_weight(uid, 5)


def test_old_profiles_get_a_first_entry(temp_db):
    uid = db.save_user(person(77))                                                 # made before weight tracking
    with db.connect() as con:
        con.execute("DELETE FROM weights")
    db.init_db()
    assert db.weight_history(uid)["weight_kg"].tolist() == [77]


def test_targets_follow_weight(temp_db):
    uid = db.create_user(person(100), "1234")
    before = calculate_targets(person(100))
    db.log_weight(uid, 90)
    user = db.get_user(uid)
    after = calculate_targets(UserProfile(**{k: user[k] for k in db.PROFILE_FIELDS}))
    assert after.calories < before.calories - 50                                   # 10 kg less -> lower target


# --- app ------------------------------------------------------------------------------------------------------
def test_app_log_weight_and_rebuild(temp_db):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    at.radio[0].set_value("New user").run()
    at.checkbox(key="pf_consent").check()
    at.checkbox(key="pfallergy_none").check()
    at.text_input[0].input("Walid")
    at.text_input[1].input("1234")
    at.text_input[2].input("1234")
    next(n for n in at.number_input if n.label == "Weight (kg)").set_value(100.0)
    at.button[0].click().run()
    at.session_state["goto"] = "Progress"
    at.run()
    assert any("Weight progress" in s.value for s in at.subheader)

    at.number_input(key="log_weight_kg").set_value(88.0)
    next(b for b in at.button if b.label == "Log weight").click().run()
    assert not at.exception
    uid = db.list_users()[0]["id"]
    assert db.get_user(uid)["weight_kg"] == 88.0
    assert any("Weight saved" in s.value for s in at.success)
    assert any("your daily target is" in i.value for i in at.info)               # targets changed with the weight

    at.button(key="rebuild_week").click().run()
    assert not at.exception
    plan = db.latest_plan(uid)
    assert plan["week_number"] == 2 and plan["targets"]["calories"] == calculate_targets(
        UserProfile("Walid", 25, "male", 88.0, 170.0, 2, "maintain")).calories
