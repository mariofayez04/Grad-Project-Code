"""Phase 6: Streamlit web app.

Run:  streamlit run app.py
"""
import html
import random
import re
from datetime import date
from functools import partial

import altair as alt

import pandas as pd
import streamlit as st

from src import db
from src.calculator import ACTIVITY_LABELS, DEFAULT_MEALS, GOAL_LABELS, MEALS_LABELS, UserProfile, calculate_targets
from src.config import FOODS_CLEAN
from src.dishes import ARABIC, load_recipes, recipe_line
from src.embeddings import FoodEmbeddings
from src.feedback import NEUTRAL_RE, acceptance_rate, choice_foods, food_penalties, next_week_inputs, week_summary
from src.nutrients import (HEALTH_MODES, LABELS, MICROS, day_micros, filter_modes, fmt, micro_targets, parse_modes,
                           status, target_text)
from src.nutrients import week_summary as nutrient_summary
from src.parser import apply_to_foods, load_ner, parse_preferences
from src.planner import (DAYS, EGYPTIAN_OPTIONS, MACROS, SLOT_ICONS, MealItem, MealPlanner, component_of, daily_targets,
                         plan_slots,
                         search_swap, slot_label, split_name, swap_options, taste_from_preferences)
from src.progress import expected_line, expected_weekly_change, pace_warning, weekly_rate
from src.report import week_pdf
from src.shopping import SECTIONS, format_amount, shopping_list, to_text

st.set_page_config(page_title="NutriPlan AI", page_icon="🥗", layout="wide")

GROUP_LABELS = {"milk": "Milk / dairy", "egg": "Egg", "peanut": "Peanut", "tree_nut": "Tree nuts", "shellfish": "Shellfish",
                "fish": "Fish", "soy": "Soy", "wheat": "Wheat", "sesame": "Sesame", "gluten": "Gluten"}
FLAG_LABELS = {"meat": "Meat (vegetarian)", "pork": "Pork", "alcohol": "Alcohol", "meat_not_fish": "Meat except fish"}
MEAL_BODY_PX = 385       # height of the food list in every meal card
TARGET_CHANGE_KCAL = 50   # a new weight changing the calorie target by this much is pointed out
TURNED_DOWN_PENALTY = 1.5  # this week's swapped-away / not-chosen foods, in new proposals right away
TURNED_DOWN_FAMILY = 0.5   # ... and other versions of the same food (e.g. another veal cut)
SWAP_PAGE = 3            # suggestions shown at a time
SEARCH_MIN_MATCH = 0.6   # below this, typed search says there is no close match
KIND_LABELS = {"protein": "protein foods", "carb": "carb foods (bread, rice, pasta, potatoes, beans...)",
               "vegetable": "vegetables", "fruit": "fruits", "fat": "cooking fats", "main": "breakfast main foods",
               "extra": "snack foods (yogurt, milk, nuts...)"}

CSS = """
<style>
:root { --bg: #000000; --panel: #1E1E1E; --panel-line: #2F2F2F; --red: #E63946; --red-dark: #C1121F;
        --text: #F5F5F5; --muted: #B3B3B3; }
/* Profile menu on the left: dark-gray panel like the meal cards */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--panel-line); }
[data-testid="stSidebar"] [data-testid="stForm"] { border-color: var(--panel-line); }
/* Every button red */
.stButton button, .stDownloadButton button, .stFormSubmitButton button, [data-testid="stPopover"] > div > button {
        background: var(--red) !important; color: #FFFFFF !important; border: 1px solid var(--red) !important; }
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover,
[data-testid="stPopover"] > div > button:hover { background: var(--red-dark) !important; border-color: var(--red-dark) !important; }
.stButton button:disabled { background: #4A1F23 !important; border-color: #4A1F23 !important; color: #9A9A9A !important; }
.hero { background: var(--panel); border-left: 6px solid var(--red); border-radius: 16px; padding: 18px 24px; margin-bottom: 8px; }
.hero h1 { margin: 0; padding: 0; color: var(--text); font-size: 2.1rem; }
.hero p { margin: 4px 0 0; color: var(--muted); font-size: .95rem; }
[data-testid="stMetric"] { background: var(--panel); border-radius: 12px; padding: 10px 14px; border: 1px solid var(--panel-line); }
[class*="st-key-mealcard"], [class*="st-key-shopcard"], [class*="st-key-cmpcard"] { background: var(--panel); border: 1px solid var(--panel-line); border-top: 5px solid var(--red);
        border-radius: 14px; padding: 14px 16px 10px; }
.meal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.meal-name { font-size: 1.25rem; font-weight: 700; color: var(--text); }
.meal-kcal { background: var(--red); color: #fff; border-radius: 999px; padding: 2px 12px; font-size: .85rem; font-weight: 600; }
.food { padding: 8px 0 10px; border-bottom: 1px dashed #3A3A3A; }
.food:last-child { border-bottom: none; }
.food-title { font-size: 1.12rem; font-weight: 700; color: var(--text); line-height: 1.3; }
.food-grams { font-size: .9rem; font-weight: 600; color: var(--red); margin-left: 6px; }
.food-sub { font-size: .92rem; color: var(--muted); margin-top: 1px; }
.cmp-head { font-size: 1.15rem; font-weight: 700; margin: 0 0 6px; }
.cmp-totals { background: #141414; border: 1px solid var(--panel-line); border-radius: 10px; padding: 8px 12px;
        font-size: .9rem; color: var(--muted); margin-top: 6px; }
.cmp-totals b { color: var(--text); }
.cmp-best { color: #fff; background: var(--red); border-radius: 999px; padding: 1px 10px; font-size: .78rem; font-weight: 600; }
.nut-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 12px; }
.nut { background: var(--panel); border: 1px solid var(--panel-line); border-radius: 999px; padding: 3px 11px;
        font-size: .82rem; color: var(--muted); }
.nut b { color: var(--text); }
.nut.ok { border-color: #2E7D32; } .nut.ok::before { content: "✓ "; color: #66BB6A; }
.nut.close { border-color: #B8860B; } .nut.close::before { content: "≈ "; color: #E0B040; }
.nut.low, .nut.high { border-color: var(--red); }
.nut.low::before { content: "↓ "; color: var(--red); } .nut.high::before { content: "↑ "; color: var(--red); }
.food-recipe { font-size: .8rem; color: #C9A227; }
.food-ar { font-weight: 600; color: #C9A227; margin-left: 4px; }
.food-macros { font-size: .84rem; color: #8F8F8F; margin-top: 3px; font-variant-numeric: tabular-nums; }
</style>
"""


# --- cached resources (loaded once per server) -------------------------------------------------------
@st.cache_resource(show_spinner="Loading food data and AI models...")
def load_resources():
    foods = pd.read_csv(FOODS_CLEAN)
    emb = FoodEmbeddings(foods)
    load_ner()  # warm up the NER model so the first plan is fast
    db.init_db()
    return foods, emb


@st.cache_data(show_spinner=False)
def allowed_foods(preferences_text, modes=""):
    """Foods left after the user's allergies, diet and diet modes (same filters the planner used)."""
    foods, _ = load_resources()
    return filter_modes(apply_to_foods(foods, parse_preferences(preferences_text))[0], modes)


def _other_bases(fdc_id, day_ids):
    foods, _ = load_resources()
    base_of = dict(zip(foods["fdc_id"], foods["base_name"]))
    return {base_of[i] for i in day_ids if i != fdc_id and i in base_of}


@st.cache_data(show_spinner="Finding similar foods...")
def swap_choices(fdc_id, slot, calories, preferences_text, day_ids, modes=""):
    """Ranked replacement foods for one item of a meal (same component, similar food, same calories)."""
    _, emb = load_resources()
    return swap_options(fdc_id, slot, calories, allowed_foods(preferences_text, modes), emb, _other_bases(fdc_id, day_ids))


@st.cache_data(show_spinner="Searching foods...")
def search_choices(query, fdc_id, slot, calories, preferences_text, day_ids, modes=""):
    """Replacement foods matching what the user typed. Returns (items, best match score)."""
    _, emb = load_resources()
    return search_swap(query, fdc_id, slot, calories, allowed_foods(preferences_text, modes), emb,
                       _other_bases(fdc_id, day_ids))


def swap_kind(fdc_id, slot):
    """Which kind of food a swap is chosen from (for messages), e.g. 'protein foods'."""
    foods, _ = load_resources()
    comp = component_of(foods.loc[foods["fdc_id"] == fdc_id].iloc[0], slot)
    return KIND_LABELS.get(comp.name if comp else "", "similar foods")


def build_plan(profile, foods, emb, taste_vector=None, penalties=None):
    """Targets -> parse preferences -> filter foods -> plan. Returns everything needed to save it."""
    targets = calculate_targets(profile)
    prefs = parse_preferences(profile.preferences_text)
    allowed, _ = apply_to_foods(foods, prefs)
    if taste_vector is None:
        taste_vector = taste_from_preferences(prefs)
    seed = random.randint(0, 1_000_000)
    plan = MealPlanner(allowed, emb, taste_vector, penalties, seed, nutrition=targets).plan_week(targets)
    return plan, targets, prefs, seed, taste_vector


def learned_inputs(user_id, foods):
    """What earlier weeks taught us: (taste_vector of the latest plan, food penalties from finished weeks)."""
    last = db.latest_plan(user_id)
    taste = last["taste_vector"] if last else None
    return taste, food_penalties(db.user_history(user_id), foods, db.user_swaps(user_id), db.user_choices(user_id))


def finish_week(plan, items, foods, emb):
    """End the week: learn from the checkboxes and build next week's plan."""
    db.mark_plan_finished(plan["id"])
    user = db.get_user(plan["user_id"])
    profile = profile_of(user)
    taste, penalties = next_week_inputs(plan["taste_vector"], emb, items, db.user_history(user["id"]),
                                        db.user_swaps(user["id"]), db.user_choices(user["id"]))
    new_plan, targets, prefs, seed, taste = build_plan(profile, foods, emb, taste, penalties)
    db.save_plan(user["id"], new_plan, targets, prefs, seed, taste)


# --- sidebar: profile form -------------------------------------------------------------------------------
def set_user(user_id):
    """Remember the signed-in user in the session and in the URL (?user=ID&t=signature), so a page reload
    keeps you signed in. The signature comes from the server secret + the PIN, so the link can't be guessed."""
    st.session_state.user_id = user_id
    st.query_params.clear()
    if user_id is not None:
        st.query_params.update(user=str(user_id), t=db.session_token(user_id))


def current_user_id():
    if "user_id" not in st.session_state:
        try:
            uid = int(st.query_params.get("user", ""))
        except ValueError:
            uid = None
        ok = uid is not None and db.get_user(uid) is not None and db.check_token(uid, st.query_params.get("t", ""))
        st.session_state.user_id = uid if ok else None
        if not ok and st.query_params:
            st.query_params.clear()
    return st.session_state.user_id


def sign_in_form():
    """Name + PIN. Profiles made before PINs existed are asked to create one."""
    pending = st.session_state.get("needs_pin")
    if pending:
        st.sidebar.info(f"**{pending['name']}** has no PIN yet. Create one to protect this profile.")
        with st.sidebar.form("new_pin"):
            pin = st.text_input("New PIN (4-8 digits)", type="password", max_chars=8)
            again = st.text_input("Repeat PIN", type="password", max_chars=8)
            ok = st.form_submit_button("Save PIN & sign in", type="primary", width="stretch")
        if ok:
            try:
                if pin != again:
                    raise ValueError("The two PINs don't match.")
                db.set_pin(pending["id"], pin)
                st.session_state.pop("needs_pin")
                set_user(pending["id"])
                st.rerun()
            except ValueError as e:
                st.sidebar.error(str(e))
        st.sidebar.button("Cancel", on_click=lambda: st.session_state.pop("needs_pin", None), width="stretch")
        return

    with st.sidebar.form("sign_in"):
        name = st.text_input("Name")
        pin = st.text_input("PIN", type="password", max_chars=8)
        ok = st.form_submit_button("Sign in", type="primary", width="stretch")
    if ok:
        status, uid = db.verify_pin(name, pin)
        if status == "ok":
            set_user(uid)
            st.rerun()
        elif status == "no_pin":
            st.session_state.needs_pin = {"id": uid, "name": name.strip()}
            st.rerun()
        elif status == "locked":
            st.sidebar.error(f"Too many wrong PINs. This profile is locked for {db.LOCK_MINUTES} minutes.")
        else:
            st.sidebar.error("Wrong name or PIN.")


def profile_form(user):
    """Profile fields. New user: name + PIN; signed-in user: name fixed. Returns (profile, pin) when submitted."""
    u = user or {}
    with st.sidebar.form("profile"):
        name = st.text_input("Name", value=u.get("name", ""), disabled=user is not None)
        if user is None:
            c1, c2 = st.columns(2)
            pin = c1.text_input("PIN (4-8 digits)", type="password", max_chars=8, help="You sign in with your name and this PIN.")
            again = c2.text_input("Repeat PIN", type="password", max_chars=8)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 18, 100, int(u.get("age", 25)))
        sex = c2.selectbox("Biological sex", ["male", "female"], index=["male", "female"].index(u.get("sex", "male")),
                           format_func=str.title, help="Needed by the Mifflin-St Jeor BMR formula.")
        weight = c1.number_input("Weight (kg)", 30.0, 300.0, float(u.get("weight_kg", 70.0)), step=0.5)
        height = c2.number_input("Height (cm)", 120.0, 230.0, float(u.get("height_cm", 170.0)), step=1.0)
        activity = st.selectbox("Activity level", [1, 2, 3], index=int(u.get("activity_level", 2)) - 1,
                                format_func=lambda k: f"{k} - {ACTIVITY_LABELS[k]}")
        goals = list(GOAL_LABELS)
        goal = st.selectbox("Goal", goals, index=goals.index(u.get("goal", "maintain")), format_func=GOAL_LABELS.get)
        meal_counts = list(MEALS_LABELS)
        meals = st.selectbox("Meals per day", meal_counts, index=meal_counts.index(int(u.get("meals_per_day") or DEFAULT_MEALS)),
                             format_func=MEALS_LABELS.get, help="The day's calories are shared over this many meals.")
        modes = st.multiselect("Health focus (optional)", list(HEALTH_MODES), default=sorted(parse_modes(u.get("health_modes"))),
                               format_func=HEALTH_MODES.get, placeholder="None",
                               help="Diet modes for diabetes or high blood pressure. General guidance only - follow your "
                                    "doctor's advice.")
        egypt = list(EGYPTIAN_OPTIONS)
        egyptian = st.selectbox("🥘 Egyptian dishes", egypt, index=egypt.index(u.get("egyptian_dishes") or "sometimes"),
                                format_func=EGYPTIAN_OPTIONS.get,
                                help="Koshari, ful medames, molokhia, mahshi... as whole meals. Sometimes = about one a day.")
        prefs_text = st.text_area("Preferences & allergies", value=u.get("preferences_text", ""),
                                  placeholder="e.g. I hate mushrooms and I'm allergic to shrimp. I love grilled chicken.",
                                  help="Write normally. Allergies are always double-checked against an allergen dictionary.")
        label = "🥗 Generate my weekly plan" if user else "🥗 Create profile & weekly plan"
        submitted = st.form_submit_button(label, type="primary", width="stretch")

    if not submitted:
        return None
    if not name.strip():
        st.sidebar.error("Please enter your name.")
        return None
    if user is None:
        if pin != again:
            st.sidebar.error("The two PINs don't match.")
            return None
        try:
            db.validate_pin(pin)
        except ValueError as e:
            st.sidebar.error(str(e))
            return None
    profile = UserProfile(name.strip(), int(age), sex, float(weight), float(height), int(activity), goal, prefs_text.strip(),
                          int(meals), "|".join(sorted(modes)), egyptian)
    return profile, (pin if user is None else None)


def sidebar_form():
    """Signed out: Sign in / New user. Signed in: your profile + Sign out. Returns (profile, pin) or None."""
    st.sidebar.header("👤 Your profile")
    uid = current_user_id()
    if uid is None:
        mode = st.sidebar.radio("Account", ["Sign in", "New user"], horizontal=True, label_visibility="collapsed")
        if mode == "Sign in":
            sign_in_form()
            return None
        return profile_form(None)
    user = db.get_user(uid)
    c1, c2 = st.sidebar.columns([3, 2], vertical_alignment="center")
    c1.markdown(f"Signed in as **{user['name']}**")
    c2.button("Sign out", on_click=set_user, args=(None,), width="stretch")
    return profile_form(user)


# --- main area ---------------------------------------------------------------------------------------------
def plan_micro_targets(plan):
    """Nutrient reference values saved with the plan (plans from before Feature #7: from the profile)."""
    if plan["targets"].get("micros"):
        return plan["targets"]["micros"]
    user = db.get_user(plan["user_id"]) or {"sex": "male", "age": 30}
    return micro_targets(user["sex"], user["age"], plan["targets"]["calories"])


def nutrient_pills(values, targets):
    """Row of small pills: value (target) per nutrient, coloured by status."""
    pills = []
    for m in MICROS:
        label, unit = LABELS[m]
        t = targets[m]
        aim = f" <small>({target_text(t, unit)})</small>" if t["value"] is not None else ""
        pills.append(f'<span class="nut {status(values[m], t)}">{label} <b>{fmt(values[m], unit)}</b>{aim}</span>')
    return '<div class="nut-row">' + "".join(pills) + "</div>"


def show_targets(targets, micros):
    st.subheader("🎯 Your daily targets")
    c = st.columns(4)
    c[0].metric("Calories", f"{targets['calories']:,} kcal")
    c[1].metric("Protein", f"{targets['protein_g']} g")
    c[2].metric("Carbs", f"{targets['carbs_g']} g")
    c[3].metric("Fat", f"{targets['fat_g']} g")
    st.caption(f"BMR {targets['bmr']:,.0f} kcal · TDEE {targets['tdee']:,.0f} kcal · BMI {targets['bmi']}")
    modes = parse_modes(targets.get("health_modes"))
    if modes:
        st.markdown("🩺 **Diet mode:** " + " · ".join(HEALTH_MODES[m] for m in sorted(modes)))
    st.caption("Also every day: " + " · ".join(f"{LABELS[m][0]} {target_text(micros[m], LABELS[m][1])}"
                                               for m in MICROS if micros[m]["value"] is not None))
    for w in targets.get("warnings", []):
        st.warning(w)


def show_understood(prefs):
    st.subheader("🧠 What I understood from your preferences")
    blocked = [GROUP_LABELS.get(g, g) for g in prefs["block_allergen_groups"]]
    blocked += [FLAG_LABELS.get(f, f) for f in prefs["block_diet_flags"]]
    blocked += [t for t in prefs["block"] if t not in {g.lower() for g in blocked}]
    c = st.columns(3)
    c[0].markdown("**🚫 Never included**<br>" + (", ".join(sorted(set(blocked))) or "—"), unsafe_allow_html=True)
    c[1].markdown("**👎 Less often**<br>" + (", ".join(prefs["avoid"]) or "—"), unsafe_allow_html=True)
    c[2].markdown("**👍 More often**<br>" + (", ".join(prefs["prefer"]) or "—"), unsafe_allow_html=True)


def on_check(plan_id, day_index, slot, key):
    db.set_meal_eaten(plan_id, day_index, slot, st.session_state[key])


def do_swap(item_id, new_item, picker_key):
    db.replace_item(item_id, new_item)
    for k in [k for k in st.session_state if str(k).startswith(picker_key)]:
        del st.session_state[k]  # reset the picker, page and search box


def next_page(page_key, n_pages):
    st.session_state[page_key] = (st.session_state.get(page_key, 0) + 1) % n_pages


def swap_buttons(items, key, picker_key, pick):
    for i, s in enumerate(items):
        title, sub = split_name(s.name)
        st.button(f"{title} · {s.grams:.0f} g" + (f"  —  {sub}" if sub else "") + f"  ·  {s.calories:.0f} kcal, P {s.protein_g:.0f}",
                  key=f"{key}_{i}", on_click=do_swap, args=(pick, s, picker_key), width="stretch")


def swap_food(plan, meal, day, d, slot, preferences_text):
    """'Swap a food' popover: pick a food of the meal, then an AI-suggested similar food (3 at a time,
    'Show more options' for the next ones) or type what you would like instead."""
    options = {r.id: r for r in meal.itertuples() if not re.match(NEUTRAL_RE, r.name)}
    if not options:
        return
    with st.popover("🔄 Swap", help="Replace one food of this meal with a similar one.", width="stretch"):
        key = f"swap_{plan['id']}_{d}_{slot}"
        pick = st.selectbox("Food to replace", list(options), index=None, key=key, placeholder="Choose a food",
                            format_func=lambda i: options[i].name)
        if pick is None:
            st.caption("Suggestions are similar foods (sentence embeddings) with a similar macro profile, "
                       "sized to keep the same calories. Your allergies stay excluded, and next week shows "
                       "the food you swapped away less often.")
            return
        r = options[pick]
        day_ids = tuple(int(i) for i in day["fdc_id"])  # no base food twice in a day, like the planner
        modes = plan["targets"].get("health_modes", "")  # swaps respect the diet modes too
        subs = swap_choices(int(r.fdc_id), slot, float(r.calories), preferences_text, day_ids, modes)
        if not subs:
            st.caption("No similar food found for this one.")
        else:
            page_key = f"{key}_page_{pick}"
            n_pages = -(-len(subs) // SWAP_PAGE)
            page = st.session_state.get(page_key, 0) % n_pages
            st.markdown("**Suggested instead**")
            swap_buttons(subs[page * SWAP_PAGE:(page + 1) * SWAP_PAGE], f"{key}_opt{page}", key, pick)
            if n_pages > 1:
                st.button("↺ Back to the first suggestions" if page == n_pages - 1 else
                          f"➕ Show more options ({page + 2} of {n_pages})",
                          key=f"{key}_more", on_click=next_page, args=(page_key, n_pages))

        st.markdown("**Or type what you'd like**")
        query = st.text_input("What would you like instead?", key=f"{key}_query", label_visibility="collapsed",
                              placeholder="e.g. brown rice, potatoes, salmon... then press Enter")
        if query.strip():
            found, best = search_choices(query.strip(), int(r.fdc_id), slot, float(r.calories), preferences_text, day_ids,
                                         modes)
            if not found or best < SEARCH_MIN_MATCH:
                st.warning(f"No close match for \u201c{query.strip()}\u201d among {swap_kind(int(r.fdc_id), slot)} "
                           "that fit your allergies." + (" Closest ones:" if found else ""))
            swap_buttons(found, f"{key}_found", key, pick)


@st.cache_data(show_spinner=False)
def recipes():
    return load_recipes()


def food_html(r):
    """One food of a meal: bold title + grams, then the details, then the macros.
    An Egyptian dish also shows its Arabic name and what is in this portion."""
    title, sub = split_name(r.name)
    arabic = ARABIC.get(title.lower())
    title_html, sub_html = html.escape(title), html.escape(sub)
    if arabic:
        # <bdi> keeps the right-to-left name from swapping places with the grams that follow it
        title_html += f' <bdi class="food-ar">{html.escape(arabic)}</bdi>'
        line = recipe_line(recipes(), r.fdc_id, r.grams)
        if line:
            sub_html += f'<br><span class="food-recipe">🥘 {html.escape(line)}</span>'
    macros = f"(kcal : {r.calories:.0f}), (P : {r.protein_g:.0f}), (F : {r.fat_g:.0f}), (C : {r.carbs_g:.0f})"
    return (f'<div class="food"><div class="food-title">{title_html}<span class="food-grams">{r.grams:.0f} g</span></div>'
            + (f'<div class="food-sub">{sub_html}</div>' if sub_html else "")
            + f'<div class="food-macros">{macros}</div></div>')


# --- "New meal" / "New day": compare before and after, keep one (Feature #5) --------------------------------
def propose(plan, items, day_index, slot, preferences_text):
    """Plan a different meal (slot) or whole day (slot=None) and open the comparison."""
    foods, emb = load_resources()
    uid = plan["user_id"]
    penalties = food_penalties(db.user_history(uid), foods, db.user_swaps(uid), db.user_choices(uid))
    # Weekly learning only uses finished weeks; but a food turned down earlier this week is not offered right back.
    swaps, choices = db.user_swaps(uid, finished_only=False), db.user_choices(uid, finished_only=False)
    week = plan["week_number"]
    turned_down = set(swaps.loc[swaps["week_number"] == week, "old_fdc_id"].astype(int).tolist()
                      + choice_foods(choices[choices["week_number"] == week])[1])
    base_of = dict(zip(foods["fdc_id"], foods["base_name"]))
    bases = {base_of[f] for f in turned_down if f in base_of}
    for f, b in base_of.items():
        extra = TURNED_DOWN_PENALTY * (f in turned_down) + TURNED_DOWN_FAMILY * (b in bases)
        if extra:
            penalties[f] = penalties.get(f, 0.0) + extra
    with st.spinner("Planning a different version..."):
        planner = MealPlanner(allowed_foods(preferences_text, plan["targets"].get("health_modes", "")), emb,
                              plan["taste_vector"], penalties, random.randint(0, 1_000_000), nutrition=plan["targets"])
        daily = daily_targets(plan["targets"])
        if slot is None:
            new = {m.slot: m.items for m in planner.replan_day(items, day_index, daily, meals_of(plan)).meals}
        else:
            new = {slot: planner.replan_meal(items, day_index, slot, daily, meals_of(plan)).items}
    current = {sl: [MealItem(**{k: r[k] for k in MealItem.__dataclass_fields__}) for _, r in
                    items[(items["day_index"] == day_index) & (items["slot"] == sl)].iterrows()] for sl in new}
    others = items[(items["day_index"] == day_index) & ~items["slot"].isin(list(new))]
    st.session_state.proposal = {"plan_id": plan["id"], "day_index": day_index, "slot": slot, "current": current,
                                 "new": new, "others": others[MACROS].sum().to_dict(), "daily": daily,
                                 "meals": meals_of(plan)}


def meals_of(plan):
    """Number of meals a saved plan was made for (plans from before this setting had 4)."""
    return int(plan["targets"].get("meals_per_day") or DEFAULT_MEALS)


def meal_head(slot, kcal, meals):
    return (f'<div class="meal-head"><span class="meal-name">{SLOT_ICONS.get(slot, "🍴")} {slot_label(slot, meals)}</span>'
            f'<span class="meal-kcal">{kcal:.0f} kcal</span></div>')


def cancel_proposal():
    st.session_state.pop("proposal", None)


def _day_error(day_totals, daily):
    """Mean % distance of a day's calories and macros from the targets."""
    return sum(abs(day_totals[m] - daily[m]) / daily[m] for m in MACROS) / len(MACROS) * 100


def _version_column(title, meals, others, daily, best, key, n_meals):
    """One side of the comparison: the meals, then the day's totals with this version."""
    totals = {m: others[m] + sum(getattr(i, m) for items in meals.values() for i in items) for m in MACROS}
    with st.container(key=key):
        st.markdown(f'<div class="cmp-head">{title}' + (' <span class="cmp-best">🎯 closer to your targets</span>' if best else "")
                    + "</div>", unsafe_allow_html=True)
        for slot, items in meals.items():
            kcal = sum(i.calories for i in items)
            st.markdown(meal_head(slot, kcal, n_meals) + "".join(food_html(i) for i in items), unsafe_allow_html=True)
        st.markdown('<div class="cmp-totals">Day with this version: '
                    + " · ".join(f"<b>{totals[m]:.0f}{'' if m == 'calories' else ' g'}</b> {label} ({totals[m] - daily[m]:+.0f})"
                                 for m, label in zip(MACROS, ["kcal", "protein", "carbs", "fat"]))
                    + "</div>", unsafe_allow_html=True)
    return totals


def apply_choice(keep_new):
    """Save the user's pick: replace the meal(s) if the new version was kept, and log the choice for learning."""
    p = st.session_state.pop("proposal")
    if keep_new:
        for slot, items in p["new"].items():
            db.replace_meal(p["plan_id"], p["day_index"], DAYS[p["day_index"]], slot, items)
            for k in [k for k in st.session_state if str(k).startswith((f"eat_{p['plan_id']}_{p['day_index']}_{slot}",
                                                                        f"swap_{p['plan_id']}_{p['day_index']}_{slot}"))]:
                del st.session_state[k]  # widgets re-read the new meal from the database
    foods_of = lambda meals: [(i.fdc_id, i.name) for items in meals.values() for i in items]
    db.record_choice(p["plan_id"], p["day_index"], p["slot"], "new" if keep_new else "current",
                     foods_of(p["current"]), foods_of(p["new"]))


@st.dialog("Which one do you prefer?", width="large", on_dismiss=cancel_proposal)
def compare_dialog():
    p = st.session_state.proposal
    what = slot_label(p["slot"], p["meals"]).lower() if p["slot"] else "whole day"
    st.caption(f"{DAYS[p['day_index']]} · {what}. Keep the version you like better. Your choice also teaches next "
               "week's plan: foods you pick show up more, the others a bit less. Close this window to decide later.")
    err = {k: _day_error({m: p["others"][m] + sum(getattr(i, m) for items in p[k].values() for i in items) for m in MACROS},
                         p["daily"]) for k in ("current", "new")}
    left, right = st.columns(2)
    with left:
        _version_column("Current", p["current"], p["others"], p["daily"], err["current"] < err["new"] - 0.5, "cmpcard_cur", p["meals"])
        if st.button("Keep current", key="keep_current", width="stretch"):
            apply_choice(False)
            st.rerun()
    with right:
        _version_column("New", p["new"], p["others"], p["daily"], err["new"] < err["current"] - 0.5, "cmpcard_new", p["meals"])
        if st.button("Use the new one", key="keep_new", type="primary", width="stretch"):
            apply_choice(True)
            st.rerun()


def show_week(plan, items):
    targets = plan["targets"]
    micros = plan_micro_targets(plan)
    nutrients = day_micros(items, load_resources()[0])
    preferences_text = (db.get_user(plan["user_id"]) or {}).get("preferences_text", "")
    st.subheader(f"📅 Week {plan['week_number']} meal plan")
    tabs = st.tabs(DAYS)
    for d, tab in enumerate(tabs):
        day = items[items["day_index"] == d]
        with tab:
            tot = day[["calories", "protein_g", "carbs_g", "fat_g"]].sum()
            c = st.columns(4)
            c[0].metric("Calories", f"{tot.calories:,.0f}", f"{tot.calories - targets['calories']:+.0f} vs target", delta_color="off")
            c[1].metric("Protein", f"{tot.protein_g:.0f} g", f"{tot.protein_g - targets['protein_g']:+.0f} g", delta_color="off")
            c[2].metric("Carbs", f"{tot.carbs_g:.0f} g", f"{tot.carbs_g - targets['carbs_g']:+.0f} g", delta_color="off")
            c[3].metric("Fat", f"{tot.fat_g:.0f} g", f"{tot.fat_g - targets['fat_g']:+.0f} g", delta_color="off")
            if d in nutrients.index:
                st.markdown(nutrient_pills(nutrients.loc[d], micros), unsafe_allow_html=True)
            _, right = st.columns([3, 1])
            if right.button("🎲 New version of this day", key=f"newday_{plan['id']}_{d}", width="stretch",
                            help="See a different plan for this whole day next to the current one, and keep the one you prefer."):
                propose(plan, items, d, None, preferences_text)

            for i, slot in enumerate(plan_slots(day)):
                if i % 2 == 0:
                    cols = st.columns(2)  # one row per pair, in time order: breakfast | lunch, dinner | snack
                meal = day[day["slot"] == slot]
                with cols[i % 2].container(key=f"mealcard_{plan['id']}_{d}_{slot}"):
                    st.markdown(meal_head(slot, meal["calories"].sum(), meals_of(plan)), unsafe_allow_html=True)
                    # Fixed height so every meal card is the same size (a long meal scrolls inside its card).
                    with st.container(height=MEAL_BODY_PX, border=False):
                        st.markdown("".join(food_html(r) for r in meal.itertuples()), unsafe_allow_html=True)
                    left, right = st.columns(2, vertical_alignment="center")
                    with left:
                        swap_food(plan, meal, day, d, slot, preferences_text)
                    if right.button("🎲 New meal", key=f"newmeal_{plan['id']}_{d}_{slot}", width="stretch",
                                  help="See a different meal next to this one, and keep the one you prefer."):
                        propose(plan, items, d, slot, preferences_text)
                    key = f"eat_{plan['id']}_{d}_{slot}"
                    eaten = bool(meal["eaten"].fillna(0).max()) if not meal.empty else False
                    st.checkbox("Eaten ✅", value=eaten, key=key, on_change=on_check, args=(plan["id"], d, slot, key))


@st.cache_data(show_spinner="Preparing the PDF...", max_entries=20)
def plan_pdf(plan_info, items, user_name):
    """Printable PDF of the week (Feature #8); rebuilt only when the plan or its ticks change."""
    foods, _ = load_resources()
    return week_pdf(plan_info, items, user_name, foods)


def show_overview(plan, items):
    st.subheader("🗓️ Week at a glance")
    slots = plan_slots(items)
    overview = (items.groupby(["day_index", "day", "slot"])["name"].apply(lambda s: " + ".join(n.split(",")[0] for n in s))
                .unstack("slot")[slots].reset_index(level=0, drop=True)
                .rename(columns=lambda sl: slot_label(sl, meals_of(plan))))
    st.dataframe(overview, width="stretch")

    name = (db.get_user(plan["user_id"]) or {}).get("name", "")
    info = {k: plan[k] for k in ("week_number", "targets", "prefs")}
    c1, c2, _ = st.columns([1, 1, 2])
    # Files are made only when a button is clicked (a function instead of the data). Otherwise the browser fetches
    # every file on each page load, and download managers save the PDF on their own at sign-in.
    c1.download_button("🖨️ Download plan (PDF)", partial(plan_pdf, info, items, name), on_click="ignore",
                       file_name=f"meal_plan_week{plan['week_number']}.pdf", mime="application/pdf", type="primary",
                       width="stretch", help="Printable: every day with tick boxes, plus the shopping list.")
    c2.download_button("⬇️ Download plan (CSV)", partial(items.drop(columns=["id", "plan_id"]).to_csv, index=False),
                       on_click="ignore", file_name="meal_plan.csv", mime="text/csv", width="stretch")


def show_nutrients(plan, items, foods):
    """Feature #7: the week's fiber, sugars, saturated fat, sodium, potassium, calcium and iron."""
    st.subheader("🧪 Nutrients this week")
    s = nutrient_summary(items, foods, plan_micro_targets(plan))
    icon = {"ok": "✅", "close": "🟡", "low": "🔻", "high": "🔺", "info": "ℹ️"}
    rows = []
    for r in s.itertuples():
        t = {"value": r.target, "kind": r.kind}
        rows.append({"": icon[status(r.average, t)], "Nutrient": r.label, "Daily average": fmt(r.average, r.unit),
                     "Target per day": target_text(t, r.unit) or "no limit (see note)",
                     "Days on target": f"{int(r.days_met)} / {r.days}" if pd.notna(r.days_met) else "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Targets for adults: fiber 14 g per 1,000 kcal and saturated fat under 10% of calories (Dietary "
               "Guidelines for Americans), sodium, potassium, calcium and iron (US National Academies). Sugars include "
               "the natural sugar of fruit and milk, so no limit is applied unless the diabetes mode is on. Sodium counts the "
               "foods only, not the salt you add when cooking (1 teaspoon of salt = about 2,300 mg). A few values missing in "
               "USDA are estimated from similar foods.")


def show_shopping(plan, items, foods):
    """Feature #3: the week's shopping list, grouped by shop section, with tick boxes and downloads."""
    st.subheader("🛒 Shopping list")
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    c1.caption("Everything for this week's plan. Rice, pasta, beans and meat show the approximate raw / dry "
               "amount to buy. Tick items off as you shop (ticks last while this page is open).")
    skip = c2.toggle("Leave out meals already eaten", key=f"shop_skip_{plan['id']}")
    shop = shopping_list(items, foods, skip_eaten=skip)
    if shop.empty:
        st.info("Nothing left to buy: every meal is ticked as eaten.")
        return

    cols, filled = st.columns(3), [0, 0, 0]
    for section, rows in shop.groupby("section", sort=False):
        i = filled.index(min(filled))  # next section goes to the shortest column
        filled[i] += len(rows) + 2
        with cols[i].container(key=f"shopcard_{plan['id']}_{SECTIONS.index(section)}"):
            st.markdown(f'<div class="meal-head"><span class="meal-name">{section}</span>'
                        f'<span class="meal-kcal">{len(rows)} items</span></div>', unsafe_allow_html=True)
            for r in rows.itertuples():
                extra = " · ".join(x for x in (r.label, r.note) if x)
                st.checkbox(f"**{r.item}** — {format_amount(r.grams)}" + (f"  :gray[({extra})]" if extra else ""),
                            key=f"shop_{plan['id']}_{r.item}_{r.label}")

    d1, d2, _ = st.columns([1, 1, 2])
    title = f"NutriPlan shopping list - week {plan['week_number']}"
    d1.download_button("⬇️ Shopping list (text)", partial(to_text, shop, title), on_click="ignore",
                       file_name="shopping_list.txt", mime="text/plain", width="stretch")
    csv = shop.assign(amount=shop["grams"].map(format_amount)).drop(columns=["plate_g"])
    d2.download_button("⬇️ Shopping list (CSV)", partial(csv.to_csv, index=False), on_click="ignore",
                       file_name="shopping_list.csv", mime="text/csv", width="stretch")


# --- weight tracking (Feature #6) -------------------------------------------------------------------------------
def profile_of(user):
    return UserProfile(**{k: user[k] for k in db.PROFILE_FIELDS})


def weight_chart(history, expected):
    """Logged weights (red line + points) and the weight the plan should give (grey dashed line)."""
    hist = history.assign(day=pd.to_datetime(history["day"]))
    base = alt.Chart(hist).encode(x=alt.X("day:T", title=None), y=alt.Y("weight_kg:Q", title="kg", scale=alt.Scale(zero=False)),
                                  tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("weight_kg:Q", title="Weight (kg)")])
    actual = base.mark_line(color="#E63946", strokeWidth=3) + base.mark_point(color="#E63946", filled=True, size=70)
    plan_line = alt.Chart(expected).mark_line(color="#9E9E9E", strokeDash=[6, 4]).encode(
        x="day:T", y="expected_kg:Q", tooltip=[alt.Tooltip("day:T", title="Date"),
                                                alt.Tooltip("expected_kg:Q", title="Expected (kg)", format=".1f")])
    return (plan_line + actual).properties(height=260)


def show_weight(plan, user_id, foods, emb):
    st.subheader("⚖️ Weight progress")
    user = db.get_user(user_id)
    history = db.weight_history(user_id)
    t = plan["targets"]
    expected = expected_weekly_change(t["calories"], t["tdee"])
    expected = expected if abs(expected) >= 0.005 else 0.0  # "maintain" shows 0.00, not -0.00
    rate = weekly_rate(history)
    now, start = history["weight_kg"].iloc[-1], history["weight_kg"].iloc[0]
    since = pd.to_datetime(history["day"].iloc[0]).strftime("%d %b")

    c = st.columns(4)
    c[0].metric("Current weight", f"{now:.1f} kg", f"{now - start:+.1f} kg since {since}", delta_color="off")
    c[1].metric("Expected per week", f"{expected:+.2f} kg",
                help="(daily calorie target - TDEE) x 7 / 7,700 kcal per kg. A rule of thumb: water, salt and "
                     "muscle make real weights go up and down from day to day.")
    c[2].metric("Your pace per week", f"{rate:+.2f} kg" if rate is not None else "-",
                help="Straight line through your last 4 weeks of entries. Needs entries at least a week apart.")
    c[3].metric("BMI", f"{now / (user['height_cm'] / 100) ** 2:.1f}")
    warning = pace_warning(rate, now)
    if warning:
        st.warning(warning)
    if len(history) > 1:
        st.altair_chart(weight_chart(history, expected_line(history, start, expected)), width="stretch")
    else:
        st.caption("Log your weight once a week (same time of day, e.g. in the morning) to see your progress here.")

    with st.form("log_weight", border=False):
        c1, c2, c3 = st.columns([2, 2, 2], vertical_alignment="bottom")
        weight = c1.number_input("Weight (kg)", 30.0, 300.0, float(now), step=0.1, format="%.1f", key="log_weight_kg")
        day = c2.date_input("Date", value=date.today(), max_value=date.today(), format="DD/MM/YYYY", key="log_weight_day")
        logged = c3.form_submit_button("➕ Log weight", type="primary", width="stretch")
    if logged:
        try:
            db.log_weight(user_id, weight, day)
            st.session_state.weight_logged = f"Saved {weight:.1f} kg for {day:%d %b %Y}."
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    if st.session_state.pop("weight_logged", None):
        st.success("Weight saved. Your profile now uses your latest weight.")

    # The latest weight changes the targets: point it out and offer to rebuild this week now.
    current = calculate_targets(profile_of(user))
    if abs(current.calories - t["calories"]) >= TARGET_CHANGE_KCAL:
        st.info(f"With your latest weight ({user['weight_kg']:.1f} kg) your daily target is **{current.calories:,} kcal** "
                f"(this week's plan uses {t['calories']:,} kcal). Next week's plan uses it automatically - or rebuild "
                "this week now.")
        if st.button("🔄 Rebuild this week with my new targets", key="rebuild_week"):
            with st.spinner("Building your 7-day plan..."):
                new_week_plan(user_id, profile_of(user), foods, emb)
            st.rerun()


RATING_LABELS = {1: "1 - not at all", 2: "2", 3: "3 - okay", 4: "4", 5: "5 - very much"}


def rate_week(plan):
    """Optional 30-second survey about the week (real-user evaluation, Feature #2)."""
    saved = db.get_survey(plan["id"]) or {}
    with st.expander("📝 Rate this week (optional, 30 seconds) - helps evaluate the app", expanded=False):
        with st.form(f"survey_{plan['id']}", border=False):
            liked = st.select_slider("How much did you like this week's meals?", options=list(RATING_LABELS),
                                     value=saved.get("liked", 3), format_func=RATING_LABELS.get)
            portions = st.select_slider("Were the portions realistic?", options=list(RATING_LABELS),
                                        value=saved.get("portions", 3), format_func=RATING_LABELS.get)
            easy = st.select_slider("Was the plan easy to follow?", options=list(RATING_LABELS),
                                    value=saved.get("easy", 3), format_func=RATING_LABELS.get)
            answers = list(db.SURVEY_ANSWERS)
            use = st.radio("Would you keep using an app like this?", answers, horizontal=True,
                           index=answers.index(saved.get("would_use", "maybe")), format_func=str.capitalize)
            comment = st.text_area("Anything to improve? (optional)", value=saved.get("comment", ""), max_chars=500)
            if st.form_submit_button("Save my rating", type="primary"):
                db.save_survey(plan["id"], liked, portions, easy, use, comment)
                st.success("Thank you - your rating is saved." + (" (updated)" if saved else ""))


def show_feedback(plan, items, foods, emb):
    st.subheader("🔁 Finish the week")
    meals = items.groupby(["day_index", "slot"])["eaten"].max()
    done = int(meals.fillna(0).sum())
    st.progress(done / len(meals), text=f"Meals eaten this week: {done} / {len(meals)}")
    st.caption("When the week is over, press the button. Unticked meals count as skipped: next week shows those foods "
               "(and similar ones) less often, and more of what you ate.")
    rate_week(plan)
    if st.button("✅ Finish week & plan next week", type="primary", disabled=done == 0,
                 help=None if done else "Tick at least one meal you ate first."):
        with st.spinner("Learning from this week and building the next one..."):
            finish_week(plan, items, foods, emb)
        st.rerun()


def show_learning(user_id):
    """What the system learned from the previous weeks, and acceptance over time."""
    history = db.user_history(user_id)
    if history.empty:
        return
    st.subheader("📈 What I learned from your previous weeks")
    last_week = history[history["week_number"] == history["week_number"].max()]
    swaps, choices = db.user_swaps(user_id), db.user_choices(user_id)
    week = last_week["week_number"].max()
    s = week_summary(last_week, swaps=swaps[swaps["week_number"] == week], choices=choices[choices["week_number"] == week])
    c = st.columns(3)
    c[0].markdown("**Showing less often** (skipped last week)<br>" +
                  ("<br>".join(f"• {n} ×{k}" for n, k in s["most_skipped"]) or "—"), unsafe_allow_html=True)
    c[2].markdown("**Swapped away or not chosen** (shown less often)<br>" +
                  ("<br>".join([f"• {n} ×{k}" for n, k in s["swapped_away"]] + [f"• {n} (not chosen)" for n, _ in s["not_chosen"]])
                   or "—"), unsafe_allow_html=True)
    c[1].markdown("**Showing more like** (eaten last week)<br>" +
                  ("<br>".join(f"• {n} ×{k}" for n, k in s["most_eaten"]) or "—"), unsafe_allow_html=True)
    rates = pd.DataFrame([{"Week": f"Week {w}", "Meals eaten (%)": round(100 * acceptance_rate(g), 1)}
                          for w, g in history.groupby("week_number")])
    st.bar_chart(rates, x="Week", y="Meals eaten (%)", height=220, color="#E63946")


def new_week_plan(user_id, profile, foods, emb):
    """Build and save a new week for a user from their profile, keeping what earlier weeks taught us."""
    taste, penalties = learned_inputs(user_id, foods)
    plan, targets, prefs, seed, taste = build_plan(profile, foods, emb, taste, penalties)
    db.save_plan(user_id, plan, targets, prefs, seed, taste)


def main():
    foods, emb = load_resources()
    db.init_db()
    submitted = sidebar_form()

    if submitted is not None:
        profile, pin = submitted
        try:
            with st.spinner("Building your 7-day plan..."):
                calculate_targets(profile)  # validate before saving anything
                if pin is not None:
                    user_id = db.create_user(profile, pin)
                else:
                    user_id = current_user_id()
                    history = db.weight_history(user_id)
                    db.update_user(user_id, profile)
                    if history.empty or abs(history["weight_kg"].iloc[-1] - profile.weight_kg) > 0.05:
                        db.log_weight(user_id, profile.weight_kg)  # a new weight in the profile is a new entry
                new_week_plan(user_id, profile, foods, emb)
            set_user(user_id)
            st.rerun()
        except ValueError as e:
            st.sidebar.error(str(e))

    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🥗 NutriPlan AI</h1><p>Personalized nutrition recommendations · '
                '⚠️ For education only, not medical advice. Consult a doctor or dietitian for medical conditions.</p></div>',
                unsafe_allow_html=True)

    user_id = current_user_id()
    plan = db.latest_plan(user_id) if user_id else None
    if plan is None:
        st.info("👈 Press **Generate my weekly plan** to build your first plan." if user_id else
                "👈 **Sign in** with your name and PIN, or choose **New user** to create your profile and first plan.")
        st.markdown(
            "**How it works**\n"
            "1. Your calorie target comes from the Mifflin-St Jeor formula, your activity level and your goal.\n"
            "2. An AI model (FoodBaseBERT-NER) reads your preferences; allergies are double-checked with an allergen dictionary.\n"
            "3. Meals are built from 3,000+ USDA foods, with portions solved to hit your calorie and macro targets.\n"
            "4. Tick the meals you ate. Next week's plan learns from what you skipped.")
        return

    show_targets(plan["targets"], plan_micro_targets(plan))
    show_understood(plan["prefs"])
    items = db.plan_items(plan["id"])
    show_week(plan, items)
    show_overview(plan, items)
    show_nutrients(plan, items, foods)
    show_shopping(plan, items, foods)
    show_feedback(plan, items, foods, emb)
    show_learning(user_id)
    show_weight(plan, user_id, foods, emb)
    if st.session_state.get("proposal", {}).get("plan_id") == plan["id"]:
        compare_dialog()


main()
