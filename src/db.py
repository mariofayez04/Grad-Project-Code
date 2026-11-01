"""Phase 6: storage for users, weekly plans and the eaten / not-eaten checkboxes.

SQLite on your own computer (data/app.db). With the DATABASE_URL setting: Postgres, for hosts whose disk isn't
kept, such as Vercel. The queries below are written for both.

Tables
  users       one row per person (profile + free-text preferences + hashed PIN for signing in)
  plans       one row per generated week (targets, parsed preferences, taste vector for Phase 7)
  plan_items  one row per food in a plan; `eaten` is set per meal (all items of a meal share it)
  swaps       every "Swap a food" the user made (the replaced food is a dislike signal for Phase 7)
  choices     every "New meal" / "New day" comparison: which version the user kept (a like / dislike signal)
  weights     the user's logged weights, one per day (Feature #6); the latest one is also the profile weight
  surveys     the optional "Rate this week" answers (Feature #2, real-user evaluation), one per plan
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import lru_cache
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
    meals_per_day INTEGER NOT NULL DEFAULT 4,
    health_modes TEXT NOT NULL DEFAULT '',
    egyptian_dishes TEXT NOT NULL DEFAULT 'sometimes',
    pin_hash TEXT,
    pin_salt TEXT,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
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
CREATE TABLE IF NOT EXISTS swaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    day_index INTEGER NOT NULL,
    slot TEXT NOT NULL,
    old_fdc_id INTEGER NOT NULL,
    old_name TEXT NOT NULL,
    new_fdc_id INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS choices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    day_index INTEGER NOT NULL,
    slot TEXT,                   -- NULL = the whole day was compared
    kept TEXT NOT NULL,          -- 'new' or 'current'
    current_foods TEXT NOT NULL, -- JSON [{fdc_id, name}] of the version that was there
    new_foods TEXT NOT NULL,     -- JSON [{fdc_id, name}] of the version that was proposed
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day TEXT NOT NULL,           -- YYYY-MM-DD
    weight_kg REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, day)
);
CREATE TABLE IF NOT EXISTS surveys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL UNIQUE REFERENCES plans(id) ON DELETE CASCADE,
    liked INTEGER NOT NULL,      -- 1-5: how much the user liked the week's meals
    portions INTEGER NOT NULL,   -- 1-5: portions realistic
    easy INTEGER NOT NULL,       -- 1-5: easy to follow
    would_use TEXT NOT NULL,     -- yes / maybe / no
    comment TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_items_plan ON plan_items(plan_id, day_index, slot);
"""

PROFILE_FIELDS = ["name", "age", "sex", "weight_kg", "height_cm", "activity_level", "goal", "preferences_text",
                  "meals_per_day", "health_modes", "egyptian_dishes"]
# Columns added after the first release; init_db adds them to an existing database.
MIGRATIONS = {"users": {"meals_per_day": "INTEGER NOT NULL DEFAULT 4", "health_modes": "TEXT NOT NULL DEFAULT ''",
                        "egyptian_dishes": "TEXT NOT NULL DEFAULT 'sometimes'", "pin_hash": "TEXT", "pin_salt": "TEXT", "failed_logins": "INTEGER NOT NULL DEFAULT 0",
                        "locked_until": "TEXT"}}

PIN_RE = re.compile(r"^\d{4,8}$")
PIN_ITERATIONS = 200_000   # PBKDF2 rounds: slow to brute-force a stolen database
MAX_FAILED_LOGINS = 5      # wrong PINs in a row before the profile is locked
LOCK_MINUTES = 5


def db_path():
    """NUTRI_DB_PATH lets tests use a temporary database."""
    return Path(os.environ.get("NUTRI_DB_PATH", DB_PATH))


def database_url():
    """The hosted Postgres database, if this server has one."""
    return os.environ.get("DATABASE_URL", "")


def _postgres_schema():
    """SCHEMA in Postgres types. Names stay unique ignoring case, as COLLATE NOCASE does in SQLite."""
    schema = (SCHEMA.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
              .replace(" COLLATE NOCASE", "").replace(" INTEGER", " BIGINT")
              .replace(" REAL", " DOUBLE PRECISION").replace(" BLOB", " BYTEA"))
    return schema + "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_name ON users (lower(name));"


@lru_cache(maxsize=1)
def _postgres_pool(url):
    """Connections are kept open, because each page refresh runs many queries. The check replaces a connection
    the database closed while idle. prepare_threshold=None: needed by connection poolers such as Neon's."""
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    return ConnectionPool(url, min_size=1, max_size=5, open=True, check=ConnectionPool.check_connection,
                          kwargs={"row_factory": dict_row, "prepare_threshold": None})


class _PostgresConnection:
    """The two sqlite3 calls the queries below use, on a Postgres connection."""

    def __init__(self, con):
        self.con = con

    def execute(self, sql, params=()):
        return self.con.execute(sql.replace("?", "%s"), params or None)

    def executemany(self, sql, rows):
        with self.con.cursor() as cur:
            cur.executemany(sql.replace("?", "%s"), rows)


@contextmanager
def connect():
    if database_url():
        with _postgres_pool(database_url()).connection() as con:  # commits at the end, undoes everything on an error
            yield _PostgresConnection(con)
        return
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


def _read_sql(query, params=()):
    """Result of a query as a DataFrame."""
    with connect() as con:
        cur = con.execute(query, params)
        columns = [c[0] for c in cur.description]
        rows = [tuple(dict(r).values()) for r in cur.fetchall()]
    return pd.DataFrame.from_records(rows, columns=columns, coerce_float=True)


def init_db():
    with connect() as con:
        if database_url():
            con.execute("SELECT pg_advisory_xact_lock(727401)")  # several servers can start at the same time
            con.execute(_postgres_schema())
        else:
            con.executescript(SCHEMA)
            for table, cols in MIGRATIONS.items():
                have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
                for col, decl in cols.items():
                    if col not in have:
                        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        # Profiles from before weight tracking: their profile weight becomes the first entry.
        con.execute("INSERT INTO weights (user_id, day, weight_kg, created_at) SELECT id, substr(created_at, 1, 10), "
                    "weight_kg, created_at FROM users WHERE id NOT IN (SELECT user_id FROM weights)")


def _now():
    return datetime.now().isoformat(timespec="seconds")


# --- users ---------------------------------------------------------------------------------------------
def save_user(profile):
    """Insert or update a user by name, without a PIN. Returns the user id.
    For scripts and tests only: the app uses create_user / update_user, so nobody can overwrite another
    person's profile by typing their name."""
    data = {f: getattr(profile, f) for f in PROFILE_FIELDS}
    with connect() as con:
        row = con.execute("SELECT id FROM users WHERE lower(name) = lower(?)", (data["name"],)).fetchone()
        if row:
            sets = ", ".join(f"{f} = ?" for f in PROFILE_FIELDS)
            con.execute(f"UPDATE users SET {sets}, updated_at = ? WHERE id = ?", (*data.values(), _now(), row["id"]))
            return row["id"]
        cols = ", ".join(PROFILE_FIELDS)
        cur = con.execute(f"INSERT INTO users ({cols}, created_at, updated_at) VALUES ({', '.join('?' * len(data))}, ?, ?) "
                          "RETURNING id", (*data.values(), _now(), _now()))
        return cur.fetchone()["id"]


def create_user(profile, pin):
    """New user with a PIN. Raises ValueError if the PIN is invalid or the name is already taken."""
    validate_pin(pin)
    with connect() as con:
        if con.execute("SELECT 1 FROM users WHERE lower(name) = lower(?)", (profile.name,)).fetchone():
            raise ValueError(f"The name '{profile.name}' is already taken. Sign in, or choose another name.")
    user_id = save_user(profile)
    set_pin(user_id, pin)
    log_weight(user_id, profile.weight_kg)
    return user_id


def update_user(user_id, profile):
    """Update a signed-in user's profile (the name stays the same)."""
    fields = [f for f in PROFILE_FIELDS if f != "name"]
    with connect() as con:
        con.execute(f"UPDATE users SET {', '.join(f'{f} = ?' for f in fields)}, updated_at = ? WHERE id = ?",
                    (*[getattr(profile, f) for f in fields], _now(), user_id))


# --- PIN sign-in -------------------------------------------------------------------------------------------
def validate_pin(pin):
    if not PIN_RE.match(pin or ""):
        raise ValueError("The PIN must be 4 to 8 digits.")


def _hash_pin(pin, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt_hex), PIN_ITERATIONS).hex()


def set_pin(user_id, pin):
    validate_pin(pin)
    salt = secrets.token_hex(16)
    with connect() as con:
        con.execute("UPDATE users SET pin_hash = ?, pin_salt = ?, failed_logins = 0, locked_until = NULL WHERE id = ?",
                    (_hash_pin(pin, salt), salt, user_id))


def verify_pin(name, pin):
    """Check a sign-in. Returns (status, user_id): status is "ok", "no_pin" (profile made before PINs existed,
    the user must create one), "wrong" (unknown name or wrong PIN - not told apart) or "locked"."""
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE lower(name) = lower(?)", ((name or "").strip(),)).fetchone()
        if row is None:
            return "wrong", None
        if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > datetime.now():
            return "locked", None
        if row["pin_hash"] is None:
            return "no_pin", row["id"]
        if PIN_RE.match(pin or "") and hmac.compare_digest(_hash_pin(pin, row["pin_salt"]), row["pin_hash"]):
            con.execute("UPDATE users SET failed_logins = 0, locked_until = NULL WHERE id = ?", (row["id"],))
            return "ok", row["id"]
        failed = row["failed_logins"] + 1
        locked = (datetime.now() + timedelta(minutes=LOCK_MINUTES)).isoformat(timespec="seconds") if failed >= MAX_FAILED_LOGINS else None
        con.execute("UPDATE users SET failed_logins = ?, locked_until = ? WHERE id = ?",
                    (0 if locked else failed, locked, row["id"]))
        return ("locked" if locked else "wrong"), None


def has_pin(user_id):
    with connect() as con:
        row = con.execute("SELECT pin_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        return bool(row and row["pin_hash"])


def _secret():
    """Server secret for sign-in links. With a hosted database it comes from the database URL, which every server
    of the app shares and which stays out of the database. Otherwise: created once next to the database (data/secret.key)."""
    if database_url():
        return hashlib.sha256(f"nutriplan sign-in links:{database_url()}".encode()).digest()
    path = db_path().parent / "secret.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return bytes.fromhex(path.read_text(encoding="utf-8").strip())


def session_token(user_id):
    """Signature put in the page URL so a reload keeps you signed in. It depends on the PIN hash, so
    changing the PIN ends old links, and nobody can open a profile by just typing ?user=ID."""
    with connect() as con:
        row = con.execute("SELECT pin_hash FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row or not row["pin_hash"]:
        return None
    return hmac.new(_secret(), f"{user_id}:{row['pin_hash']}".encode(), hashlib.sha256).hexdigest()[:32]


def check_token(user_id, token):
    expected = session_token(user_id)
    return bool(expected and token and hmac.compare_digest(expected, token))


def list_users():
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT id, name FROM users ORDER BY name")]


def get_user(user_id):
    """Profile of a user (PIN fields left out)."""
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        return None
    return {k: row[k] for k in row.keys() if k not in ("pin_hash", "pin_salt", "failed_logins", "locked_until")}


# --- plans ---------------------------------------------------------------------------------------------
def save_plan(user_id, plan, targets, prefs, seed=None, taste_vector=None):
    """Store a WeekPlan. Returns the plan id."""
    with connect() as con:
        week = con.execute("SELECT COALESCE(MAX(week_number), 0) + 1 AS week FROM plans WHERE user_id = ?", (user_id,)).fetchone()["week"]
        targets_json = json.dumps({k: v for k, v in asdict(targets).items()})
        prefs_json = json.dumps(prefs.summary())
        blob = None if taste_vector is None else np.asarray(taste_vector, dtype=np.float32).tobytes()
        cur = con.execute(
            "INSERT INTO plans (user_id, week_number, seed, targets_json, prefs_json, taste_vector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id", (user_id, week, seed, targets_json, prefs_json, blob, _now()))
        plan_id = cur.fetchone()["id"]
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
    return _read_sql("SELECT * FROM plan_items WHERE plan_id = ? ORDER BY day_index, id", (plan_id,))


def user_history(user_id, finished_only=True):
    """All plan items of a user's weeks, with the week number (input for the feedback loop)."""
    query = ("SELECT pi.*, p.week_number FROM plan_items pi JOIN plans p ON p.id = pi.plan_id WHERE p.user_id = ?"
             + (" AND p.finished = 1" if finished_only else "") + " ORDER BY p.week_number, pi.day_index, pi.id")
    return _read_sql(query, (user_id,))


def set_meal_eaten(plan_id, day_index, slot, eaten):
    """eaten: True / False / None (unanswered). Applies to every item of the meal."""
    value = None if eaten is None else int(bool(eaten))
    with connect() as con:
        con.execute("UPDATE plan_items SET eaten = ? WHERE plan_id = ? AND day_index = ? AND slot = ?",
                    (value, plan_id, day_index, slot))


def replace_item(item_id, item):
    """Swap one food of a plan for another (a planner.MealItem). The meal's eaten state is kept and the
    swap is recorded, so next week's plan learns that the replaced food was not wanted."""
    with connect() as con:
        old = con.execute("SELECT plan_id, day_index, slot, fdc_id, name FROM plan_items WHERE id = ?", (item_id,)).fetchone()
        con.execute("INSERT INTO swaps (plan_id, day_index, slot, old_fdc_id, old_name, new_fdc_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", (old["plan_id"], old["day_index"], old["slot"], old["fdc_id"], old["name"],
                                                     int(item.fdc_id), _now()))
        con.execute("UPDATE plan_items SET fdc_id = ?, name = ?, grams = ?, calories = ?, protein_g = ?, carbs_g = ?, "
                    "fat_g = ? WHERE id = ?", (int(item.fdc_id), item.name, float(item.grams), float(item.calories),
                                               float(item.protein_g), float(item.carbs_g), float(item.fat_g), item_id))


def user_swaps(user_id, finished_only=True):
    """Every swap of a user's weeks, with the week number (input for the feedback loop)."""
    query = ("SELECT s.*, p.week_number FROM swaps s JOIN plans p ON p.id = s.plan_id WHERE p.user_id = ?"
             + (" AND p.finished = 1" if finished_only else "") + " ORDER BY s.id")
    return _read_sql(query, (user_id,))


def replace_meal(plan_id, day_index, day, slot, items):
    """Put a new meal (list of planner.MealItem) in a saved plan. The new meal is not ticked yet."""
    with connect() as con:
        con.execute("DELETE FROM plan_items WHERE plan_id = ? AND day_index = ? AND slot = ?", (plan_id, day_index, slot))
        con.executemany(
            "INSERT INTO plan_items (plan_id, day_index, day, slot, fdc_id, name, grams, calories, protein_g, carbs_g, fat_g) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(plan_id, day_index, day, slot, int(i.fdc_id), i.name, float(i.grams), float(i.calories), float(i.protein_g),
              float(i.carbs_g), float(i.fat_g)) for i in items])


def record_choice(plan_id, day_index, slot, kept, current_foods, new_foods):
    """Remember which version the user kept after "New meal" / "New day". foods: [(fdc_id, name)]."""
    if kept not in ("new", "current"):
        raise ValueError(f"kept must be 'new' or 'current', got {kept!r}")
    as_json = lambda foods: json.dumps([{"fdc_id": int(f), "name": n} for f, n in foods])
    with connect() as con:
        con.execute("INSERT INTO choices (plan_id, day_index, slot, kept, current_foods, new_foods, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", (plan_id, day_index, slot, kept, as_json(current_foods), as_json(new_foods), _now()))


def user_choices(user_id, finished_only=True):
    """Every kept / not-kept comparison of a user's weeks: week_number, slot, kept, chosen, rejected
    (chosen / rejected = [{fdc_id, name}] of the version kept / not kept)."""
    query = ("SELECT c.*, p.week_number FROM choices c JOIN plans p ON p.id = c.plan_id WHERE p.user_id = ?"
             + (" AND p.finished = 1" if finished_only else "") + " ORDER BY c.id")
    df = _read_sql(query, (user_id,))
    current, new = df["current_foods"].map(json.loads), df["new_foods"].map(json.loads)
    df["chosen"] = [n if k == "new" else c for k, c, n in zip(df["kept"], current, new)]
    df["rejected"] = [c if k == "new" else n for k, c, n in zip(df["kept"], current, new)]
    return df.drop(columns=["current_foods", "new_foods"])


# --- weight tracking (Feature #6) ------------------------------------------------------------------------------
WEIGHT_LIMITS = (30, 300)


def log_weight(user_id, weight_kg, day=None):
    """Save a weight for a day (today by default); a second entry on the same day replaces the first.
    The most recent entry also becomes the profile weight, so next week's targets follow it."""
    weight_kg = float(weight_kg)
    if not WEIGHT_LIMITS[0] <= weight_kg <= WEIGHT_LIMITS[1]:
        raise ValueError(f"Weight must be between {WEIGHT_LIMITS[0]} and {WEIGHT_LIMITS[1]} kg.")
    day = (day or datetime.now().date())
    day = day.isoformat() if hasattr(day, "isoformat") else str(day)
    if day > datetime.now().date().isoformat():
        raise ValueError("The date can't be in the future.")
    with connect() as con:
        con.execute("INSERT INTO weights (user_id, day, weight_kg, created_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (user_id, day) DO UPDATE SET weight_kg = excluded.weight_kg, created_at = excluded.created_at",
                    (user_id, day, weight_kg, _now()))
        latest = con.execute("SELECT weight_kg FROM weights WHERE user_id = ? ORDER BY day DESC LIMIT 1", (user_id,)).fetchone()
        con.execute("UPDATE users SET weight_kg = ?, updated_at = ? WHERE id = ?", (latest["weight_kg"], _now(), user_id))


def weight_history(user_id):
    """All logged weights of a user, oldest first (columns day, weight_kg)."""
    return _read_sql("SELECT day, weight_kg FROM weights WHERE user_id = ? ORDER BY day", (user_id,))


def delete_weight(user_id, day):
    """Remove one entry (a typo); the profile weight goes back to the latest remaining entry."""
    with connect() as con:
        con.execute("DELETE FROM weights WHERE user_id = ? AND day = ?", (user_id, str(day)))
        latest = con.execute("SELECT weight_kg FROM weights WHERE user_id = ? ORDER BY day DESC LIMIT 1", (user_id,)).fetchone()
        if latest:
            con.execute("UPDATE users SET weight_kg = ? WHERE id = ?", (latest["weight_kg"], user_id))


# --- real-user evaluation (Feature #2) --------------------------------------------------------------------------
SURVEY_ANSWERS = ("yes", "maybe", "no")


def save_survey(plan_id, liked, portions, easy, would_use, comment=""):
    """Save (or update) the week's rating. Scores 1-5; would_use yes / maybe / no."""
    for v in (liked, portions, easy):
        if int(v) not in range(1, 6):
            raise ValueError("Ratings must be between 1 and 5.")
    if would_use not in SURVEY_ANSWERS:
        raise ValueError(f"would_use must be one of {SURVEY_ANSWERS}")
    with connect() as con:
        con.execute("INSERT INTO surveys (plan_id, liked, portions, easy, would_use, comment, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (plan_id) DO UPDATE SET liked = excluded.liked, "
                    "portions = excluded.portions, easy = excluded.easy, would_use = excluded.would_use, "
                    "comment = excluded.comment, created_at = excluded.created_at",
                    (plan_id, int(liked), int(portions), int(easy), would_use, (comment or "").strip(), _now()))


def get_survey(plan_id):
    with connect() as con:
        row = con.execute("SELECT * FROM surveys WHERE plan_id = ?", (plan_id,)).fetchone()
        return dict(row) if row else None


def pilot_data():
    """Everything the evaluation needs from real use, without names or profiles:
    (items of finished weeks with user / week, surveys, swaps per plan, choices per plan, weights per user)."""
    items = _read_sql("SELECT p.user_id, p.week_number, pi.day_index, pi.slot, pi.fdc_id, pi.eaten "
                      "FROM plan_items pi JOIN plans p ON p.id = pi.plan_id WHERE p.finished = 1")
    surveys = _read_sql("SELECT s.liked, s.portions, s.easy, s.would_use, s.comment, p.user_id, p.week_number "
                        "FROM surveys s JOIN plans p ON p.id = s.plan_id")
    swaps = _read_sql("SELECT plan_id, COUNT(*) AS n FROM swaps GROUP BY plan_id")
    choices = _read_sql("SELECT plan_id, kept FROM choices")
    weights = _read_sql("SELECT user_id, COUNT(*) AS n FROM weights GROUP BY user_id")
    with connect() as con:
        n_users = con.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        n_plans = con.execute("SELECT COUNT(*) AS n FROM plans").fetchone()["n"]
    return {"users": n_users, "plans": n_plans, "items": items, "surveys": surveys, "swaps": swaps, "choices": choices,
            "weights": weights}


def mark_plan_finished(plan_id):
    with connect() as con:
        con.execute("UPDATE plans SET finished = 1 WHERE id = ?", (plan_id,))
