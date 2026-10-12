"""Phase 6: Streamlit web app.

Run:  streamlit run app.py
"""
import random

import pandas as pd
import streamlit as st

from src import db
from src.calculator import ACTIVITY_LABELS, GOAL_LABELS, UserProfile, calculate_targets
from src.config import FOODS_CLEAN
from src.embeddings import FoodEmbeddings
from src.parser import apply_to_foods, load_ner, parse_preferences
from src.planner import DAYS, MealPlanner, taste_from_preferences

st.set_page_config(page_title="NutriPlan AI", page_icon="🥗", layout="wide")

SLOT_ICONS = {"breakfast": "🍳", "lunch": "🍲", "dinner": "🍽️", "snack": "🍎"}
GROUP_LABELS = {"milk": "Milk / dairy", "egg": "Egg", "peanut": "Peanut", "tree_nut": "Tree nuts", "shellfish": "Shellfish",
                "fish": "Fish", "soy": "Soy", "wheat": "Wheat", "sesame": "Sesame", "gluten": "Gluten"}
FLAG_LABELS = {"meat": "Meat (vegetarian)", "pork": "Pork", "alcohol": "Alcohol", "meat_not_fish": "Meat except fish"}


# --- cached resources (loaded once per server) -------------------------------------------------------
@st.cache_resource(show_spinner="Loading food data and AI models...")
def load_resources():
    foods = pd.read_csv(FOODS_CLEAN)
    emb = FoodEmbeddings(foods)
    load_ner()  # warm up the NER model so the first plan is fast
    db.init_db()
    return foods, emb


def build_plan(profile, foods, emb, taste_vector=None, food_penalties=None):
    """Targets -> parse preferences -> filter foods -> plan. Returns everything needed to save it."""
    targets = calculate_targets(profile)
    prefs = parse_preferences(profile.preferences_text)
    allowed, _ = apply_to_foods(foods, prefs)
    if taste_vector is None:
        taste_vector = taste_from_preferences(prefs)
    seed = random.randint(0, 1_000_000)
    plan = MealPlanner(allowed, emb, taste_vector, food_penalties, seed).plan_week(targets)
    return plan, targets, prefs, seed, taste_vector


# --- sidebar: profile form -------------------------------------------------------------------------------
def set_user(user_id):
    """Remember the current user in the session and in the URL (?user=ID), so a page reload keeps it."""
    st.session_state.user_id = user_id
    if user_id is None:
        st.query_params.clear()
    else:
        st.query_params["user"] = str(user_id)


def current_user_id():
    if "user_id" not in st.session_state:
        try:
            uid = int(st.query_params.get("user", ""))
            st.session_state.user_id = uid if db.get_user(uid) else None
        except ValueError:
            st.session_state.user_id = None
    return st.session_state.user_id


def sidebar_form():
    st.sidebar.header("👤 Your profile")
    users = db.list_users()
    names = ["➕ New user"] + [u["name"] for u in users]
    current = current_user_id()
    index = next((i + 1 for i, u in enumerate(users) if u["id"] == current), 0)
    choice = st.sidebar.selectbox("User", names, index=index)

    user = None
    if choice != "➕ New user":
        user = db.get_user(next(u["id"] for u in users if u["name"] == choice))
        if current != user["id"]:
            set_user(user["id"])
            st.rerun()
    elif current is not None:
        set_user(None)
        st.rerun()

    u = user or {}
    with st.sidebar.form("profile"):
        name = st.text_input("Name", value=u.get("name", ""))
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
        prefs_text = st.text_area("Preferences & allergies", value=u.get("preferences_text", ""),
                                  placeholder="e.g. I hate mushrooms and I'm allergic to shrimp. I love grilled chicken.",
                                  help="Write normally. Allergies are always double-checked against an allergen dictionary.")
        submitted = st.form_submit_button("🥗 Generate my weekly plan", type="primary", use_container_width=True)

    if submitted:
        if not name.strip():
            st.sidebar.error("Please enter your name.")
            return None
        return UserProfile(name.strip(), int(age), sex, float(weight), float(height), int(activity), goal, prefs_text.strip())
    return None


# --- main area ---------------------------------------------------------------------------------------------
def show_targets(targets):
    st.subheader("🎯 Your daily targets")
    c = st.columns(4)
    c[0].metric("Calories", f"{targets['calories']:,} kcal")
    c[1].metric("Protein", f"{targets['protein_g']} g")
    c[2].metric("Carbs", f"{targets['carbs_g']} g")
    c[3].metric("Fat", f"{targets['fat_g']} g")
    st.caption(f"BMR {targets['bmr']:,.0f} kcal · TDEE {targets['tdee']:,.0f} kcal · BMI {targets['bmi']}")
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


def show_week(plan, items):
    targets = plan["targets"]
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

            for i, slot in enumerate(["breakfast", "lunch", "dinner", "snack"]):
                if i % 2 == 0:
                    cols = st.columns(2)  # one row per pair, so the order reads breakfast | lunch, dinner | snack
                meal = day[day["slot"] == slot]
                with cols[i % 2].container(border=True):
                    st.markdown(f"#### {SLOT_ICONS[slot]} {slot.title()}  ·  {meal['calories'].sum():.0f} kcal")
                    for r in meal.itertuples():
                        st.markdown(f"- **{r.name}** — {r.grams:.0f} g  \n"
                                    f"  <small>{r.calories:.0f} kcal · P {r.protein_g:.0f} g · C {r.carbs_g:.0f} g · F {r.fat_g:.0f} g</small>",
                                    unsafe_allow_html=True)
                    key = f"eat_{plan['id']}_{d}_{slot}"
                    eaten = bool(meal["eaten"].fillna(0).max()) if not meal.empty else False
                    st.checkbox("Eaten ✅", value=eaten, key=key, on_change=on_check, args=(plan["id"], d, slot, key))


def show_overview(items):
    st.subheader("🗓️ Week at a glance")
    overview = (items.groupby(["day_index", "day", "slot"])["name"].apply(lambda s: " + ".join(n.split(",")[0] for n in s))
                .unstack("slot")[["breakfast", "lunch", "dinner", "snack"]].reset_index(level=0, drop=True))
    st.dataframe(overview, use_container_width=True)

    meals = items.groupby(["day_index", "slot"])["eaten"].max()
    done = int(meals.fillna(0).sum())
    st.progress(done / len(meals), text=f"Meals eaten this week: {done} / {len(meals)}")
    st.download_button("⬇️ Download plan (CSV)", items.drop(columns=["id", "plan_id"]).to_csv(index=False),
                       file_name="meal_plan.csv", mime="text/csv")


def main():
    foods, emb = load_resources()
    db.init_db()
    profile = sidebar_form()

    if profile is not None:
        try:
            with st.spinner("Building your 7-day plan..."):
                plan, targets, prefs, seed, taste = build_plan(profile, foods, emb)
                user_id = db.save_user(profile)
                db.save_plan(user_id, plan, targets, prefs, seed, taste)
            set_user(user_id)
            st.rerun()
        except ValueError as e:
            st.sidebar.error(str(e))

    st.title("🥗 NutriPlan AI")
    st.caption("Personalized nutrition recommendations · ⚠️ For education only, not medical advice. "
               "Consult a doctor or dietitian for medical conditions.")

    user_id = current_user_id()
    plan = db.latest_plan(user_id) if user_id else None
    if plan is None:
        st.info("👈 Fill in your profile on the left and press **Generate my weekly plan**.")
        st.markdown(
            "**How it works**\n"
            "1. Your calorie target comes from the Mifflin-St Jeor formula, your activity level and your goal.\n"
            "2. An AI model (FoodBaseBERT-NER) reads your preferences; allergies are double-checked with an allergen dictionary.\n"
            "3. Meals are built from 3,000+ USDA foods, with portions solved to hit your calorie and macro targets.\n"
            "4. Tick the meals you ate. Next week's plan learns from what you skipped.")
        return

    show_targets(plan["targets"])
    show_understood(plan["prefs"])
    items = db.plan_items(plan["id"])
    show_week(plan, items)
    show_overview(items)


main()
