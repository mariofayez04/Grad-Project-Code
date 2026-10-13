"""Phase 6: SQLite storage for users, weekly plans and the eaten / not-eaten checkboxes.

Tables
  users       one row per person (profile + free-text preferences)
  plans       one row per generated week (targets, parsed preferences, taste vector for Phase 7)
  plan_items  one row per food in a plan; `eaten` is set per meal (all items of a meal share it)
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL,
    weight_kg REAL NOT NULL,
    height_cm REAL NOT NULL,
    activity_level INTEGER NOT NULL,
    goal TEXT NOT NULL,
    preferences_text TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    week_number INTEGER NOT NULL,
    seed INTEGER,
    targets_json TEXT NOT NULL,
    prefs_json TEXT NOT NULL,
    taste_vector BLOB,
    created_at TEXT NOT NULL,
    finished INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS plan_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    day_index INTEGER NOT NULL,
    day TEXT NOT NULL,
    slot TEXT NOT NULL,
    fdc_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    grams REAL NOT NULL,
    calories REAL NOT NULL,
    protein_g REAL NOT NULL,
    carbs_g REAL NOT NULL,
    fat_g REAL NOT NULL,
    eaten INTEGER            -- NULL = not answered yet, 1 = eaten, 0 = skipped
);
CREATE INDEX IF NOT EXISTS idx_items_plan ON plan_items(plan_id, day_index, slot);
"""

PROFILE_FIELDS = ["name", "age", "sex", "weight_kg", "height_cm", "activity_level", "goal", "preferences_text"]


def db_path():
    """NUTRI_DB_PATH lets tests use a temporary database."""
    return Path(os.environ.get("NUTRI_DB_PATH", DB_PATH))


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)


def _now():
    return datetime.now().isoformat(timespec="seconds")


# --- users ---------------------------------------------------------------------------------------------
def save_user(profile):
    """Insert or update a user by name. Returns the user id."""
    data = {f: getattr(profile, f) for f in PROFILE_FIELDS}
    with connect() as con:
        row = con.execute("SELECT id FROM users WHERE name = ?", (data["name"],)).fetchone()
        if row:
            sets = ", ".join(f"{f} = ?" for f in PROFILE_FIELDS)
            con.execute(f"UPDATE users SET {sets}, updated_at = ? WHERE id = ?", (*data.values(), _now(), row["id"]))
            return row["id"]
        cols = ", ".join(PROFILE_FIELDS)
        cur = con.execute(f"INSERT INTO users ({cols}, created_at, updated_at) VALUES ({', '.join('?' * len(data))}, ?, ?)",
                          (*data.values(), _now(), _now()))
        return cur.lastrowid


def list_users():
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT id, name FROM users ORDER BY name")]


def get_user(user_id):
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


# --- plans ---------------------------------------------------------------------------------------------
def save_plan(user_id, plan, targets, prefs, seed=None, taste_vector=None):
    """Store a WeekPlan. Returns the plan id."""
    with connect() as con:
        week = con.execute("SELECT COALESCE(MAX(week_number), 0) + 1 FROM plans WHERE user_id = ?", (user_id,)).fetchone()[0]
        targets_json = json.dumps({k: v for k, v in asdict(targets).items()})
        prefs_json = json.dumps(prefs.summary())
        blob = None if taste_vector is None else np.asarray(taste_vector, dtype=np.float32).tobytes()
        cur = con.execute(
            "INSERT INTO plans (user_id, week_number, seed, targets_json, prefs_json, taste_vector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", (user_id, week, seed, targets_json, prefs_json, blob, _now()))
        plan_id = cur.lastrowid
        rows = plan.to_dataframe()
        con.executemany(
            "INSERT INTO plan_items (plan_id, day_index, day, slot, fdc_id, name, grams, calories, protein_g, carbs_g, fat_g) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(plan_id, int(r.day_index), r.day, r.slot, int(r.fdc_id), r.name, float(r.grams), float(r.calories),
              float(r.protein_g), float(r.carbs_g), float(r.fat_g)) for r in rows.itertuples()])
        return plan_id


def latest_plan(user_id):
    """The user's most recent plan as a dict (with targets, prefs, taste_vector decoded), or None."""
    with connect() as con:
        row = con.execute("SELECT * FROM plans WHERE user_id = ? ORDER BY week_number DESC LIMIT 1", (user_id,)).fetchone()
    if not row:
        return None
    plan = dict(row)
    plan["targets"] = json.loads(plan.pop("targets_json"))
    plan["prefs"] = json.loads(plan.pop("prefs_json"))
    blob = plan.pop("taste_vector")
    plan["taste_vector"] = None if blob is None else np.frombuffer(blob, dtype=np.float32).copy()
    return plan


def list_plans(user_id):
    with connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT id, week_number, created_at, finished FROM plans WHERE user_id = ? ORDER BY week_number", (user_id,))]


def plan_items(plan_id):
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM plan_items WHERE plan_id = ? ORDER BY day_index, id", con, params=(plan_id,))


def user_history(user_id, finished_only=True):
    """All plan items of a user's weeks, with the week number (input for the feedback loop)."""
    query = ("SELECT pi.*, p.week_number FROM plan_items pi JOIN plans p ON p.id = pi.plan_id WHERE p.user_id = ?"
             + (" AND p.finished = 1" if finished_only else "") + " ORDER BY p.week_number, pi.day_index, pi.id")
    with connect() as con:
        return pd.read_sql_query(query, con, params=(user_id,))


def set_meal_eaten(plan_id, day_index, slot, eaten):
    """eaten: True / False / None (unanswered). Applies to every item of the meal."""
    value = None if eaten is None else int(bool(eaten))
    with connect() as con:
        con.execute("UPDATE plan_items SET eaten = ? WHERE plan_id = ? AND day_index = ? AND slot = ?",
                    (value, plan_id, day_index, slot))


def mark_plan_finished(plan_id):
    with connect() as con:
        con.execute("UPDATE plans SET finished = 1 WHERE id = ?", (plan_id,))
