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


def sign_up(at, name, pin="1234"):
    """Fill the 'New user' form of the app (name, PIN, repeat PIN) without submitting it."""
    at.radio[0].set_value("New user").run()
    at.checkbox(key="pf_consent").check()
    at.checkbox(key="pfallergy_none").check()
    at.text_input[0].input(name)
    at.text_input[1].input(pin)
    at.text_input[2].input(pin)


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
    sign_up(at, "Omar")
    at.text_area[0].input("I'm allergic to peanuts and I love lentils")
    at.button[0].click().run()
    assert not at.exception
    assert any("Week 1 meal plan" in s.value for s in at.subheader)
    assert sum(c.label == "Eaten" for c in at.checkbox) == 28
    assert any("PDF" in e.proto.label for e in at.get("download_button"))
    at.session_state["goto"] = "Shopping"                            # open another page from the top menu
    at.run()
    assert not at.exception
    assert any("Shopping list" in m.value for m in at.markdown)
    plan = db.latest_plan(db.list_users()[0]["id"])
    assert "peanut" in plan["prefs"]["block_allergen_groups"]


def test_replace_item_keeps_eaten(temp_db, week):
    from src.planner import MealItem
    user, plan, targets, prefs = week
    pid = db.save_plan(db.save_user(user), plan, targets, prefs)
    db.set_meal_eaten(pid, 0, "lunch", True)
    item_id = int(db.plan_items(pid).query("day_index == 0 and slot == 'lunch'")["id"].iloc[0])
    db.replace_item(item_id, MealItem(123, "Test food", 150.0, 200.0, 20.0, 10.0, 5.0))
    row = db.plan_items(pid).set_index("id").loc[item_id]
    assert (row["fdc_id"], row["name"], row["grams"], row["eaten"]) == (123, "Test food", 150.0, 1)


def test_replace_item_records_swap(temp_db, week):
    from src.planner import MealItem
    user, plan, targets, prefs = week
    uid = db.save_user(user)
    pid = db.save_plan(uid, plan, targets, prefs)
    old = db.plan_items(pid).iloc[0]
    db.replace_item(int(old["id"]), MealItem(123, "Test food", 150.0, 200.0, 20.0, 10.0, 5.0))
    assert db.user_swaps(uid).empty                       # week not finished yet
    swaps = db.user_swaps(uid, finished_only=False)
    assert swaps[["old_fdc_id", "old_name", "new_fdc_id", "week_number"]].values.tolist() == [[old["fdc_id"], old["name"], 123, 1]]
    db.mark_plan_finished(pid)
    assert len(db.user_swaps(uid)) == 1


# --- private users (PIN sign-in) ------------------------------------------------------------------------
def test_create_user_pin_and_taken_name(temp_db, week):
    user = week[0]
    uid = db.create_user(user, "4821")
    assert db.verify_pin(user.name, "4821") == ("ok", uid)
    assert db.verify_pin(user.name.upper(), "4821") == ("ok", uid)  # names are not case-sensitive
    assert db.verify_pin(user.name, "0000") == ("wrong", None)
    assert db.verify_pin("Nobody", "4821") == ("wrong", None)     # unknown name looks the same as a wrong PIN
    with pytest.raises(ValueError, match="already taken"):
        db.create_user(user, "1111")                               # can't overwrite someone else's profile
    for bad in ("12", "abcd", "123456789", ""):
        with pytest.raises(ValueError):
            db.set_pin(uid, bad)
    assert "pin_hash" not in db.get_user(uid)


def test_lockout_after_wrong_pins(temp_db, week):
    user = week[0]
    db.create_user(user, "4821")
    statuses = [db.verify_pin(user.name, "0000")[0] for _ in range(db.MAX_FAILED_LOGINS)]
    assert statuses == ["wrong"] * (db.MAX_FAILED_LOGINS - 1) + ["locked"]
    assert db.verify_pin(user.name, "4821") == ("locked", None)    # even the right PIN waits for the lock


def test_old_profile_without_pin(temp_db, week):
    uid = db.save_user(week[0])                                    # made before PINs existed
    assert db.verify_pin(week[0].name, "") == ("no_pin", uid)
    assert db.session_token(uid) is None and not db.check_token(uid, "x")
    db.set_pin(uid, "2468")
    assert db.verify_pin(week[0].name, "2468") == ("ok", uid)


def test_session_token(temp_db, week):
    uid = db.create_user(week[0], "4821")
    token = db.session_token(uid)
    assert db.check_token(uid, token) and not db.check_token(uid, "0" * 32) and not db.check_token(uid, "")
    db.set_pin(uid, "9999")
    assert not db.check_token(uid, token)                          # a new PIN ends old sign-in links


def test_migration_adds_pin_columns(tmp_path, monkeypatch):
    """A database from before PINs gets the new columns and keeps its users."""
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE COLLATE NOCASE, "
                "age INTEGER NOT NULL, sex TEXT NOT NULL, weight_kg REAL NOT NULL, height_cm REAL NOT NULL, "
                "activity_level INTEGER NOT NULL, goal TEXT NOT NULL, preferences_text TEXT DEFAULT '', "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    con.execute("INSERT INTO users VALUES (1, 'Old', 30, 'male', 70, 175, 2, 'maintain', '', 'x', 'x')")
    con.commit()
    con.close()
    monkeypatch.delenv("DATABASE_URL", raising=False)              # only SQLite files can be this old
    monkeypatch.setenv("NUTRI_DB_PATH", str(path))
    db.init_db()
    assert db.verify_pin("Old", "") == ("no_pin", 1)
    assert db.get_user(1)["meals_per_day"] == 4                   # profiles from before the setting keep 4 meals


def test_hosted_sign_in_links_survive_a_new_server(tmp_path, monkeypatch):
    """A hosted server has no lasting disk: the secret must not depend on a file, or a restart signs everyone out."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "first" / "t.db"))
    first = db._secret()
    monkeypatch.setenv("NUTRI_DB_PATH", str(tmp_path / "second" / "t.db"))
    assert db._secret() == first
    assert not (tmp_path / "first" / "secret.key").exists()


def test_postgres_schema_has_no_sqlite_words():
    schema = db._postgres_schema()
    for word in ("AUTOINCREMENT", "COLLATE NOCASE", " REAL", " BLOB", " INTEGER"):
        assert word not in schema


def test_names_are_unique_ignoring_case(temp_db):
    uid = db.create_user(UserProfile("Sara", 30, "female", 60, 165, 2, "maintain", ""), "1234")
    assert db.verify_pin("SARA", "1234") == ("ok", uid)
    with pytest.raises(ValueError, match="taken"):
        db.create_user(UserProfile("sara", 30, "female", 60, 165, 2, "maintain", ""), "1234")


def test_app_link_without_token_is_not_signed_in(temp_db):
    from streamlit.testing.v1 import AppTest
    uid = db.create_user(UserProfile("Nour", 30, "female", 60, 165, 2, "maintain"), "4821")
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.query_params["user"] = str(uid)                              # guessing the id is not enough
    at.run()
    assert not at.exception
    assert not any("meal plan" in s.value for s in at.subheader)
    assert at.radio[0].value == "Sign in"


def test_app_sign_in(temp_db):
    from streamlit.testing.v1 import AppTest
    uid = db.create_user(UserProfile("Nour", 30, "female", 60, 165, 2, "maintain"), "4821")
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    at.text_input[0].input("nour")
    at.text_input[1].input("1111")
    at.button[0].click().run()
    assert any("Wrong name or PIN" in e.value for e in at.error)
    at.text_input[1].input("4821")
    at.button[0].click().run()
    assert not at.exception
    assert any(f"Signed in as **Nour**" in m.value for m in at.markdown)
    assert at.session_state.user_id == uid


# --- consent and deleting an account ----------------------------------------------------------------------------
def sign_in(at, name, pin):
    at.text_input[0].input(name)
    at.text_input[1].input(pin)
    at.button[0].click().run()
    assert not at.exception


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_new_user_must_agree_before_anything_is_saved(temp_db):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    sign_up(at, "Omar")
    at.checkbox(key="pf_consent").uncheck()
    button(at, "Create profile and weekly plan").click().run()
    assert any("tick the box" in e.value for e in at.error)
    assert db.list_users() == []
    at.checkbox(key="pf_consent").check()
    at.checkbox(key="pfallergy_none").check()
    button(at, "Create profile and weekly plan").click().run()
    assert not at.exception
    assert db.get_user(db.list_users()[0]["id"])["consented_at"]


def test_old_account_is_asked_for_consent_and_can_agree(temp_db):
    from streamlit.testing.v1 import AppTest
    uid = db.create_user(UserProfile("Nour", 30, "female", 60, 165, 2, "maintain"), "4821")
    with db.connect() as con:                                       # an account from before the consent box
        con.execute("UPDATE users SET consented_at = NULL WHERE id = ?", (uid,))
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    sign_in(at, "Nour", "4821")
    assert not any("Signed in as" in m.value for m in at.markdown)  # nothing else opens until they answer
    button(at, "I agree and continue").click().run()
    assert not at.exception
    assert db.get_user(uid)["consented_at"]
    assert any("Signed in as **Nour**" in m.value for m in at.markdown)


def test_delete_account_needs_the_pin_and_removes_everything(temp_db, week):
    from streamlit.testing.v1 import AppTest
    user, plan, targets, prefs = week
    uid = db.create_user(user, "4821")
    other = db.create_user(UserProfile("Nour", 30, "female", 60, 165, 2, "maintain"), "1111")
    pid = db.save_plan(uid, plan, targets, prefs)
    kept = db.save_plan(other, plan, targets, prefs)
    db.save_survey(pid, 4, 4, 4, "yes")
    item = db.plan_items(pid).iloc[0]
    db.record_choice(pid, 0, item["slot"], "current", [(item["fdc_id"], item["name"])], [(item["fdc_id"], item["name"])])
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    sign_in(at, user.name, "4821")
    at.session_state["goto"] = "Profile"
    at.run()
    at.text_input(key="delete_pin").input("0000")
    at.checkbox(key="delete_sure").check()
    button(at, "Delete my account").click().run()
    assert any("Nothing was deleted" in e.value for e in at.error)
    assert db.get_user(uid)
    at.text_input(key="delete_pin").input("4821")
    at.checkbox(key="delete_sure").check()
    button(at, "Delete my account").click().run()
    assert not at.exception
    assert any("have been deleted" in x.value for x in at.success)
    assert db.get_user(uid) is None
    assert db.plan_items(pid).empty and db.weight_history(uid).empty
    assert db.get_survey(pid) is None
    with db.connect() as con:
        assert con.execute("SELECT COUNT(*) AS n FROM choices WHERE plan_id = ?", (pid,)).fetchone()["n"] == 0
    assert db.get_user(other) and not db.plan_items(kept).empty     # nobody else's data is touched


# --- allergy checklist ---------------------------------------------------------------------------------------------
def served_allergens(plan_id, foods):
    tags = dict(zip(foods["fdc_id"], foods["allergens"].fillna("").str.split("|").map(set)))
    return set().union(*(tags[int(f)] for f in db.plan_items(plan_id)["fdc_id"]))


def test_new_user_must_answer_the_allergy_checklist(temp_db):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    sign_up(at, "Omar")
    at.checkbox(key="pfallergy_none").uncheck()
    button(at, "Create profile and weekly plan").click().run()
    assert any("tick your allergies" in e.value for e in at.error)
    assert db.list_users() == []
    at.checkbox(key="pfallergy_peanut").check()
    at.checkbox(key="pfallergy_none").check()
    button(at, "Create profile and weekly plan").click().run()
    assert any("Please choose one" in e.value for e in at.error)
    assert db.list_users() == []


def test_ticked_allergy_stays_out_of_the_plan_and_of_new_meals(temp_db):
    from streamlit.testing.v1 import AppTest
    foods = pd.read_csv(FOODS_CLEAN)
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    sign_up(at, "Omar")
    at.checkbox(key="pfallergy_none").uncheck()
    at.checkbox(key="pfallergy_milk").check()
    at.checkbox(key="pfallergy_egg").check()
    button(at, "Create profile and weekly plan").click().run()
    assert not at.exception
    uid = db.list_users()[0]["id"]
    assert db.get_user(uid)["allergies"] == "egg|milk" or db.get_user(uid)["allergies"] == "milk|egg"
    plan = db.latest_plan(uid)
    assert not served_allergens(plan["id"], foods) & {"milk", "egg"}
    button(at, "New meal").click().run()                             # a new version must respect the ticks too
    assert not at.exception
    button(at, "Use the new one").click().run()
    assert not at.exception
    assert not served_allergens(plan["id"], foods) & {"milk", "egg"}


def test_old_account_answers_the_checklist_and_its_plan_is_rebuilt(temp_db, week):
    from streamlit.testing.v1 import AppTest
    foods = pd.read_csv(FOODS_CLEAN)
    user, plan, targets, prefs = week
    uid = db.create_user(user, "4821")
    db.save_plan(uid, plan, targets, prefs)
    with db.connect() as con:                                        # an account from before the checklist
        con.execute("UPDATE users SET allergies = NULL WHERE id = ?", (uid,))
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    sign_in(at, user.name, "4821")
    assert not any("Signed in as" in m.value for m in at.markdown)   # nothing else opens until they answer
    button(at, "Save").click().run()
    assert any("tick your allergies" in e.value for e in at.error)
    at.checkbox(key="pfallergy_wheat").check()
    button(at, "Save").click().run()
    assert not at.exception
    assert db.get_user(uid)["allergies"] == "wheat"
    newest = db.latest_plan(uid)
    assert newest["week_number"] == 2
    assert not served_allergens(newest["id"], foods) & {"wheat", "gluten"}
    assert any("Signed in as" in m.value for m in at.markdown)


def test_old_account_with_no_allergies_keeps_its_plan(temp_db, week):
    from streamlit.testing.v1 import AppTest
    user, plan, targets, prefs = week
    uid = db.create_user(user, "4821")
    db.save_plan(uid, plan, targets, prefs)
    with db.connect() as con:
        con.execute("UPDATE users SET allergies = NULL WHERE id = ?", (uid,))
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    sign_in(at, user.name, "4821")
    at.checkbox(key="pfallergy_none").check()
    button(at, "Save").click().run()
    assert not at.exception
    assert db.get_user(uid)["allergies"] == ""
    assert db.latest_plan(uid)["week_number"] == 1
