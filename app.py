"""Phase 6: Streamlit web app.

Run:  streamlit run app.py
"""
import html
import random
import re
from datetime import date, datetime
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
from src.nutrients import HEALTH_MODES, LABELS, MICROS, day_micros, filter_modes, fmt, micro_targets, parse_modes, status
from src.nutrients import week_summary as nutrient_summary
from src.parser import apply_to_foods, load_ner, parse_preferences
from src.planner import (DAYS, EGYPTIAN_OPTIONS, MACROS, MealItem, MealPlanner, component_of, daily_targets, plan_slots,
                         search_swap, slot_label, split_name, swap_options, taste_from_preferences)
from src.progress import expected_line, expected_weekly_change, pace_warning, weekly_rate
from src.report import week_pdf
from src.shopping import SECTIONS, format_amount, shopping_list, to_text

st.set_page_config(page_title="NutriPlan AI", page_icon=":material/eco:", layout="wide")

PAGES = ["Plan", "Nutrients", "Shopping", "Progress", "Profile"]
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
GOAL_NOTES = {"maintain": "Eat what you burn", "lose": "20% under your daily need", "gain": "10% over, more protein"}
ACTIVITY_NOTES = {1: "Mostly sitting", 2: "Exercise a few times a week", 3: "Hard training or physical work"}
MEALS_SHORT = {3: "3 meals", 4: "4 meals", 5: "5 meals"}
MEALS_NOTES = {3: "Breakfast, lunch, dinner", 4: "Plus a snack", 5: "Plus two snacks"}
MODE_SHORT = {"diabetes": ("Diabetes-friendly", "Less sugar, more fiber, carbs at most 40%"),
              "low_sodium": ("Low-salt / blood pressure", "Sodium at most 1,500 mg, DASH-style")}
MACRO_COLORS = {"protein_g": "var(--protein)", "carbs_g": "var(--carbs)", "fat_g": "var(--fat)"}
# status -> (words, dot colour, chip background, chip text)
STATUS = {"ok": ("On target", "var(--ok)", "var(--green-soft)", "var(--green-ink)"), "close": ("Close", "var(--close)", "var(--close-bg)", "var(--close-ink)"),
          "low": ("A little under", "var(--off)", "var(--never-bg)", "var(--never-ink)"), "high": ("A little over", "var(--off)", "var(--never-bg)", "var(--never-ink)"),
          "info": ("No limit", "var(--info)", "var(--line2)", "var(--ink2)")}

# Stroke icons (24 x 24), drawn in the text colour
ICONS = {
    "leaf": '<path d="M5 19c0-8 6-14 15-14 0 9-6 15-14 15"/><path d="M5 19l7-7"/>',
    "sun": '<path d="M12 3v3M5.6 7.6l1.4 1.4M18.4 7.6 17 9M3 15h18M7 15a5 5 0 0 1 10 0M8 19h8"/>',
    "bowl": '<path d="M3 11h18a9 9 0 0 1-18 0Z"/><path d="M9 4c0 2 2 2 2 4M14 4c0 2 2 2 2 4"/>',
    "moon": '<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z"/>',
    "apple": '<path d="M12 7c-1.5-1-5-1.5-6.5 1.5S5 16 7.5 19c1.5 1.8 3 1.5 4.5.8 1.5.7 3 1 4.5-.8 2.5-3 3-7.5 2-10.5S13.5 6 12 7Z"/>'
             '<path d="M12 7c0-2 1-3.5 3-4"/>',
    "never": '<circle cx="12" cy="12" r="9"/><path d="M6 6l12 12"/>',
    "less": '<path d="M12 5v14M6 13l6 6 6-6"/>',
    "more": '<path d="M12 19V5M6 11l6-6 6 6"/>',
}
SLOT_SVG = {"breakfast": "sun", "morning_snack": "apple", "lunch": "bowl", "snack": "apple", "dinner": "moon"}

CSS = """
<style>
:root { --card: #FFFFFF; --line: #E3E8E1; --line2: #EEF1ED; --ink: #15201C; --ink2: #3A4641; --muted: #5B6863;
        --muted2: #4A5852; --faint: #6A7671; --green: #1E5B48; --accent-bg: #1E5B48; --onaccent: #FFFFFF; --green-soft: #E6F0EA;
        --green-ink: #1A4D3D; --mint: #BFD8C9; --wash: #F7F9F6; --select-bg: #F3F8F5; --btn-line: #D5DDD3;
        --never-bg: #F8EDE6; --never-ink: #7A3A1C; --never-icon: #9A4A26; --blue-bg: #EAF1F6; --close-bg: #FBF1DC;
        --close-ink: #6E4A0B; --protein: #2E6A8E; --carbs: #D98A3D; --fat: #8C7BC0; --protein-ink: #1F4E6B;
        --carbs-ink: #8A4F12; --fat-ink: #5C4B8A; --ar: #8A5A12; --recipe: #7A5410; --strike: #7C8782; --far: #8A4A26;
        --ok: #2F7D55; --close: #C98A1B; --off: #B4532A; --info: #9AA59F; --shadow: rgba(21,32,28,.12);
        --heading: 'Schibsted Grotesk', 'Hanken Grotesk', sans-serif; }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 1.25rem; padding-bottom: 2.5rem; }
[data-testid="stHeader"] { background: transparent; }
/* Cards: containers whose key starts with card_, mealcard, shopcard or cmpcard */
[class*="st-key-card_"], [class*="st-key-mealcard"], [class*="st-key-shopcard"], [class*="st-key-cmpcard"] {
        background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 22px 24px; }
[class*="st-key-mealcard"], [class*="st-key-shopcard"] { padding: 18px 20px; }
[class*="st-key-cmpcard_"][class*="_best"] { border: 1.5px solid var(--green); }
.st-key-topbar { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 8px 14px; margin-bottom: 10px; }
[class*="st-key-card_"] h3, [class*="st-key-card_"] h2 { padding-top: 0; font-size: 1.4rem; }
/* Buttons: primary green (theme), secondary white with a soft border */
button[data-testid^="stBaseButton-secondary"], [data-testid="stPopoverButton"] {
        background: var(--card); border: 1px solid var(--btn-line); color: var(--ink); font-weight: 600; min-height: 44px; }
button[data-testid^="stBaseButton-secondary"]:hover, [data-testid="stPopoverButton"]:hover {
        background: var(--card); border-color: var(--green); color: var(--green); }
button[data-testid^="stBaseButton-primary"] { font-weight: 600; min-height: 44px; }
button[data-testid^="stBaseButton-tertiary"] { color: var(--green); font-weight: 600; min-height: 44px; }
/* Top navigation */
.st-key-nav [role="radiogroup"] { flex-wrap: wrap; row-gap: 4px; }
.st-key-nav button { border: none !important; background: transparent !important; color: var(--muted) !important;
        font-weight: 500; border-radius: 10px !important; min-height: 40px; padding: 0 14px; box-shadow: none !important; }
.st-key-nav button:hover { color: var(--ink) !important; background: var(--wash) !important; }
.st-key-nav button[data-selected="true"] { background: var(--green-soft) !important; color: var(--ink) !important; }
.st-key-nav button[data-selected="true"] p { font-weight: 600; }
/* Sign in / New user switch */
.st-key-account_mode .stRadio, .st-key-account_mode [data-testid="stRadio"] { width: 100%; }
.st-key-account_mode [role="radiogroup"] { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; padding: 4px;
        background: var(--line2); border-radius: 12px; width: 100%; }
.st-key-account_mode [role="radiogroup"] > div { margin: 0; }
.st-key-account_mode [data-testid="stRadioOption"] { width: 100%; margin: 0; justify-content: center; border-radius: 9px;
        min-height: 40px; padding: 0 10px; display: flex; align-items: center; }
.st-key-account_mode [data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-account_mode [data-testid="stRadioOption"][data-selected="true"] { background: var(--card); box-shadow: 0 1px 2px rgba(21,32,28,.12); }
.st-key-account_mode [data-testid="stRadioOption"][data-selected="true"] p { font-weight: 600; }
/* Profile choices drawn as cards */
[class*="st-key-pf_"] [role="radiogroup"] { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); grid-auto-rows: 1fr;
        gap: 8px; width: 100%; }
.st-key-page_profile [class*="st-key-pfsec_"] { background: var(--card); border: 1px solid var(--line); border-radius: 20px;
        padding: 20px 22px; }
.st-key-card_auth [class*="st-key-pfsec_"] { border-top: 1px solid var(--line2); padding-top: 14px; }
[class*="st-key-pf_"] [role="radiogroup"] > div { margin: 0; display: flex; flex-direction: column; justify-content: flex-start;
        border: 1px solid var(--btn-line); border-radius: 12px; padding: 12px 14px; background: var(--card); min-height: 44px; }
[class*="st-key-pf_"] [role="radiogroup"] > div:has([data-testid="stRadioCaption"]) { min-height: 100px; }
[class*="st-key-pf_"] [role="radiogroup"] > div[data-selected="true"] { border-color: var(--green); background: var(--select-bg);
        box-shadow: inset 0 0 0 1px var(--green); }
[class*="st-key-pf_"] [data-testid="stRadioOption"] { width: 100%; margin: 0; }
[class*="st-key-pf_"] [data-testid="stRadioCaption"] { margin: 6px 0 0; padding: 0; }
[class*="st-key-pf_"] [data-testid="stRadioCaption"] p { margin: 0; font-size: .82rem; line-height: 1.35; word-break: normal; overflow-wrap: normal; }
[class*="st-key-pf_"] [data-testid="stRadioOption"][data-selected="true"] p { font-weight: 600; }
[class*="st-key-pfmode_"] label { border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; width: 100%; }
[class*="st-key-pfmode_"] label:has(input:checked) { border-color: var(--mint); background: var(--select-bg); }
/* "Eaten" tick drawn as a pill */
[class*="st-key-eat_"] label { border: 1px solid var(--btn-line); border-radius: 10px; padding: 8px 12px; background: var(--card);
        width: 100%; min-height: 44px; align-items: center; }
[class*="st-key-eat_"] label:has(input:checked), [class*="st-key-eat_"] label[data-selected="true"] {
        background: var(--green-soft); border-color: var(--mint); }
[class*="st-key-eat_"] label:has(input:checked) p { color: var(--green-ink); font-weight: 600; }
[class*="st-key-shopitem_"] label:has(input:checked) p, [class*="st-key-shopitem_"] label[data-selected="true"] p {
        text-decoration: line-through; color: var(--strike); }
/* Day tabs drawn as pills */
.stTabs [role="tablist"] { gap: 8px; flex-wrap: wrap; border: none; box-shadow: none; padding-bottom: 4px; }
.stTabs [role="tablist"]::before, .stTabs [role="tablist"]::after { display: none; }
.stTabs [role="tab"] { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 8px 18px;
        height: auto; min-height: 42px; }
.stTabs [role="tab"]:hover { border-color: var(--green); }
.stTabs [role="tab"] p { font-weight: 600; color: var(--ink); }
.stTabs [role="tab"][aria-selected="true"] { background: var(--accent-bg); border-color: var(--accent-bg); }
.stTabs [role="tab"][aria-selected="true"] p { color: var(--onaccent); }
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stProgressBarTrack"] { background: var(--line2); height: 10px; border-radius: 99px; }
[data-testid="stProgressBarTrack"] > div { border-radius: 99px; }
[data-testid="stMetric"] { background: var(--wash); border-radius: 16px; padding: 14px 16px; }
[data-testid="stMetricValue"] { font-family: var(--heading); font-weight: 700; }
[data-testid="stExpander"] details { border-radius: 16px; border-color: var(--line); background: var(--card); }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div { background: var(--card); }
/* Text pieces */
.eyebrow { font-size: .8rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.page-title { font-family: var(--heading); font-size: 2.4rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.1;
        margin: 4px 0 8px; color: var(--ink); }
.page-intro { color: var(--muted2); font-size: 1.05rem; max-width: 660px; margin: 0 0 6px; }
.card-title { font-family: var(--heading); font-size: 1.25rem; font-weight: 600; color: var(--ink); line-height: 1.3; }
.small-muted { font-size: .85rem; color: var(--muted); }
.row-between { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 6px 16px; }
.dot { width: 8px; height: 8px; border-radius: 99px; display: inline-block; flex: none; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 14px 0 4px; }
.tile { background: var(--wash); border-radius: 16px; padding: 14px 16px; }
.tile-label { display: flex; align-items: center; gap: 8px; font-size: .82rem; font-weight: 600; color: var(--muted2); }
.tile-value { font-family: var(--heading); font-size: 1.9rem; font-weight: 700; margin: 2px 0; color: var(--ink); }
.tile-value small { font-size: .9rem; font-weight: 500; color: var(--muted); margin-left: 4px; }
.tile-note { font-size: .82rem; color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px; font-size: .82rem; font-weight: 500; }
.chip-blue { background: var(--blue-bg); color: var(--protein-ink); font-weight: 600; }
.chip-never { background: var(--never-bg); color: var(--never-ink); } .chip-less { background: var(--line2); color: var(--ink2); }
.chip-more { background: var(--green-soft); color: var(--green-ink); }
.chip-week { border: 1px solid var(--line); color: var(--muted2); background: var(--card); white-space: nowrap; }
.also { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; font-size: .85rem; color: var(--muted2); margin-top: 12px; }
.pref-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 8px; }
.pref-box { border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; }
.pref-head { display: flex; align-items: center; gap: 8px; font-weight: 600; margin-bottom: 8px; color: var(--ink); }
.empty { color: var(--muted); font-size: .88rem; }
.ring { display: flex; align-items: center; gap: 16px; }
.ring-title { font-weight: 600; color: var(--ink); }
/* Day summary */
table.tbl.tbl-day { min-width: 0; font-variant-numeric: tabular-nums; }
.tbl-day td, .tbl-day tbody th, .tbl-day thead th { padding: 9px 14px; }
.tbl-day td b { color: var(--ink); }
.tbl-day .result { display: inline-flex; align-items: center; gap: 8px; }
.tbl-day tr.first-micro td, .tbl-day tr.first-micro th { border-top: 2px solid var(--line); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: .76rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
/* Meal cards */
.meal-head { display: flex; align-items: center; gap: 12px; }
.meal-icon { width: 42px; height: 42px; border-radius: 12px; background: var(--green-soft); color: var(--green);
        display: flex; align-items: center; justify-content: center; flex: none; }
.meal-name { font-family: var(--heading); font-size: 1.2rem; font-weight: 600; color: var(--ink); line-height: 1.2; }
.meal-macros { font-size: .8rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.meal-kcal { margin-left: auto; font-family: var(--heading); font-weight: 700; font-size: 1.15rem; white-space: nowrap;
        color: var(--ink); font-variant-numeric: tabular-nums; }
.meal-kcal small { font-size: .8rem; font-weight: 500; color: var(--muted); margin-left: 3px; }
.split { display: flex; height: 6px; border-radius: 99px; overflow: hidden; gap: 2px; margin: 12px 0 2px; }
.food { padding: 11px 0; border-top: 1px solid var(--line2); }
.food-top { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
.food-title { font-weight: 600; font-size: 1rem; color: var(--ink); line-height: 1.35; }
.food-grams { flex: none; font-size: .88rem; font-weight: 600; color: var(--green); font-variant-numeric: tabular-nums; }
.food-sub { font-size: .88rem; color: var(--muted); }
.food-ar { font-weight: 600; color: var(--ar); margin-left: 4px; }
.food-recipe { font-size: .8rem; color: var(--recipe); margin-top: 2px; }
.food-macros { font-size: .8rem; color: var(--faint); margin-top: 2px; font-variant-numeric: tabular-nums; }
.mp { color: var(--protein-ink); } .mc { color: var(--carbs-ink); } .mf { color: var(--fat-ink); }
/* Tables */
.tbl-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; margin-top: 10px; }
table.tbl { width: 100%; min-width: 720px; border-collapse: collapse; font-size: .88rem; margin: 0; }
.tbl thead th { background: var(--wash); text-align: left; font-weight: 600; color: var(--muted2); padding: 11px 14px; border: none; }
.tbl td, .tbl tbody th { padding: 11px 14px; border: none; border-top: 1px solid var(--line2); color: var(--ink2);
        text-align: left; vertical-align: middle; }
.tbl tbody th { font-weight: 600; color: var(--ink); }
.barline { display: flex; align-items: center; gap: 12px; }
.barline b { min-width: 72px; font-variant-numeric: tabular-nums; color: var(--ink); }
.bar { position: relative; height: 8px; border-radius: 99px; background: var(--line2); flex: 1 1 auto; min-width: 80px; }
.bar > span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; }
.learn-list { list-style: none; margin: 0; padding: 0; font-size: .9rem; }
.learn-list li { display: flex; justify-content: space-between; gap: 10px; padding: 3px 0; margin: 0; }
/* Compare dialog */
.cmp-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-family: var(--heading);
        font-size: 1.15rem; font-weight: 600; margin-bottom: 8px; color: var(--ink); }
.cmp-best { padding: 3px 10px; border-radius: 999px; background: var(--green-soft); color: var(--green-ink); font-size: .75rem;
        font-weight: 600; font-family: 'Hanken Grotesk', sans-serif; }
.cmp-meal { margin-top: 6px; }
.cmp-totals { background: var(--wash); border-radius: 14px; padding: 10px 12px; font-size: .8rem; color: var(--muted2); margin-top: 10px; }
.cmp-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; font-variant-numeric: tabular-nums; margin-top: 4px; }
.cmp-grid b { display: block; color: var(--ink); font-size: .95rem; }
.near { color: var(--green-ink); } .far { color: var(--far); }
/* Brand, welcome and footer */
.brand { display: flex; align-items: center; gap: 10px; font-family: var(--heading); font-weight: 700; font-size: 1.2rem; color: var(--ink); }
.brand-mark { width: 34px; height: 34px; border-radius: 10px; background: var(--accent-bg); color: var(--onaccent);
        display: flex; align-items: center; justify-content: center; flex: none; }
.hero-tag { display: inline-block; padding: 6px 12px; border-radius: 999px; background: var(--green-soft); color: var(--green-ink);
        font-size: .82rem; font-weight: 600; margin-top: 22px; }
.hero-title { font-family: var(--heading); font-size: 3rem; line-height: 1.05; font-weight: 700; letter-spacing: -.025em;
        margin: 16px 0 12px; color: var(--ink); }
.hero-sub { font-size: 1.1rem; color: var(--muted2); max-width: 580px; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 26px 0 0 !important;
        padding: 0 !important; list-style: none; }
.step { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 16px 18px !important; margin: 0 !important; }
.learn-list li, .steps li { margin: 0 !important; }
.step-n { font-family: var(--heading); font-weight: 700; color: var(--green); font-size: .85rem; }
.step-t { font-weight: 600; color: var(--ink); margin: 2px 0; }
.step-d { font-size: .86rem; color: var(--muted); line-height: 1.5; }
.foot { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; font-size: .8rem; color: var(--muted);
        border-top: 1px solid var(--line); margin-top: 28px; padding-top: 18px; }
@media (max-width: 640px) { .hero-title { font-size: 2.2rem; } .page-title { font-size: 1.9rem; }
        table.tbl.tbl-day { font-size: .8rem; } .tbl-day td, .tbl-day tbody th, .tbl-day thead th { padding: 8px; } }
/* Profile choices: one column when the form itself is narrow (phone, or the sign-up card next to the welcome text) */
[class*="st-key-pf_"] { container-type: inline-size; }
@container (max-width: 440px) {
        [role="radiogroup"]:has([data-testid="stRadioOption"]) { grid-template-columns: minmax(0, 1fr) !important; }
        [role="radiogroup"] > div:has([data-testid="stRadioCaption"]) { min-height: 72px !important; } }
</style>
"""
# Dark mode: the same names with deep green-black surfaces and lighter text and accents
DARK_CSS = """
<style>
:root { --card: #161E1A; --line: #2A3530; --line2: #222C27; --ink: #E8EEEA; --ink2: #C4CEC8; --muted: #9DABA4;
        --muted2: #B3BFB9; --faint: #8E9C95; --green: #7CC6A4; --accent-bg: #2E7D5E; --onaccent: #FFFFFF; --green-soft: #1C3329;
        --green-ink: #9BD6BA; --mint: #2F5A47; --wash: #1B2420; --select-bg: #1A2E25; --btn-line: #34413A;
        --never-bg: #3A2419; --never-ink: #F0B79A; --never-icon: #E8956E; --blue-bg: #1A2B38; --close-bg: #3A2E14;
        --close-ink: #EBC77E; --protein: #6FA8CF; --carbs: #E3A35F; --fat: #A898D6; --protein-ink: #8FC0E0;
        --carbs-ink: #E8B07A; --fat-ink: #BBAEE6; --ar: #E3B66E; --recipe: #D9AE6A; --strike: #75827C; --far: #E8A884;
        --ok: #5FB38E; --close: #E0A93A; --off: #E07A50; --info: #6E7C75; --shadow: rgba(0,0,0,.4); }
</style>
"""
# Streamlit keeps the chosen theme (Light / Dark / System) in the browser under this key, the same one its
# Settings menu writes. The switch saves the choice there and reloads; the URL keeps the user signed in.
THEME_JS = """<script>
const w = (window.parent && window.parent.localStorage) ? window.parent : window;
w.localStorage.setItem("stActiveTheme-" + w.location.pathname + "-v2", JSON.stringify("%s"));
w.location.reload();
</script>"""


def is_dark():
    theme = st.context.theme
    return bool(theme and theme.type == "dark")


def chart_color():
    return "#7CC6A4" if is_dark() else "#1E5B48"


def flip_theme():
    st.session_state.apply_theme = "Dark" if st.session_state.dark_mode else "Light"


def theme_switch():
    """'Dark mode' switch. It follows the theme in use, also when it was changed in Streamlit's own Settings."""
    if "apply_theme" not in st.session_state:
        st.session_state.dark_mode = is_dark()
    st.toggle("Dark mode", key="dark_mode", on_change=flip_theme)


def svg(name, size=20, stroke=1.8, color="currentColor"):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" style="stroke:{color}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


BRAND = f'<div class="brand"><span class="brand-mark">{svg("leaf")}</span>NutriPlan</div>'
FOOTER = ('<div class="foot"><span>For education only, not medical advice. Consult a doctor or dietitian for medical '
          'conditions.</span><span>Foods from the USDA FoodData Central database</span></div>')


def page_header(eyebrow, title, intro=""):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div>'
                + (f'<p class="page-intro">{intro}</p>' if intro else ""), unsafe_allow_html=True)


def go_to(page):
    """Open another page on the next run (the nav widget can't be changed once it is drawn)."""
    st.session_state.goto = page


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


CONSENT_TEXT = ("I agree that NutriPlan may store and use my details to plan my meals, including my body measurements, "
                "allergies and health focus. They may be kept on servers outside Egypt. I can delete my account at any time.")


# --- signing in and the profile form ------------------------------------------------------------------------
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
        st.info(f"**{pending['name']}** has no PIN yet. Create one to protect this profile.")
        with st.form("new_pin", border=False):
            pin = st.text_input("New PIN (4 to 8 digits)", type="password", max_chars=8)
            again = st.text_input("Repeat PIN", type="password", max_chars=8)
            ok = st.form_submit_button("Save PIN and sign in", type="primary", width="stretch")
        if ok:
            try:
                if pin != again:
                    raise ValueError("The two PINs don't match.")
                db.set_pin(pending["id"], pin)
                st.session_state.pop("needs_pin")
                set_user(pending["id"])
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        st.button("Cancel", on_click=lambda: st.session_state.pop("needs_pin", None), width="stretch")
        return

    with st.form("sign_in", border=False):
        name = st.text_input("Name", autocomplete="username")
        pin = st.text_input("PIN", type="password", max_chars=8, autocomplete="current-password")
        ok = st.form_submit_button("Sign in", type="primary", width="stretch")
    st.caption("You stay signed in on this device after a reload.")
    if ok:
        result, uid = db.verify_pin(name, pin)
        if result == "ok":
            set_user(uid)
            st.rerun()
        elif result == "no_pin":
            st.session_state.needs_pin = {"id": uid, "name": name.strip()}
            st.rerun()
        elif result == "locked":
            st.error(f"Too many wrong PINs. This profile is locked for {db.LOCK_MINUTES} minutes.")
        else:
            st.error("Wrong name or PIN.")


def profile_form(user, targets=None):
    """Profile fields. New user: name + PIN; signed-in user: name fixed. Returns (profile, pin) when submitted.
    A signed-in user gets the wide layout (form + a summary column with the button)."""
    u = user or {}
    with st.form("profile", border=False):
        if user is not None:
            main, aside = st.columns([2.2, 1], gap="large")
        else:
            main = aside = st.container()
        with main:
            with st.container(key="pfsec_account"):
                st.markdown('<div class="card-title">Account</div>', unsafe_allow_html=True)
                name = st.text_input("Name", value=u.get("name", ""), disabled=user is not None)
                if user is None:
                    c1, c2 = st.columns(2)
                    pin = c1.text_input("PIN (4 to 8 digits)", type="password", max_chars=8,
                                        help="You sign in with your name and this PIN.")
                    again = c2.text_input("Repeat PIN", type="password", max_chars=8)

            with st.container(key="pfsec_body"):
                st.markdown('<div class="card-title">About you</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                age = c1.number_input("Age", 18, 100, int(u.get("age", 25)))
                weight = c2.number_input("Weight (kg)", 30.0, 300.0, float(u.get("weight_kg", 70.0)), step=0.5)
                height = c3.number_input("Height (cm)", 120.0, 230.0, float(u.get("height_cm", 170.0)), step=1.0)
                sex = st.radio("Biological sex", ["male", "female"], index=["male", "female"].index(u.get("sex", "male")),
                               format_func=str.title, horizontal=True, key="pf_sex", width="stretch",
                               help="Needed by the Mifflin-St Jeor BMR formula.")

            with st.container(key="pfsec_goal"):
                st.markdown('<div class="card-title">Goal and activity</div>', unsafe_allow_html=True)
                goals = list(GOAL_LABELS)
                goal = st.radio("Goal", goals, index=goals.index(u.get("goal", "maintain")), format_func=GOAL_LABELS.get,
                                captions=[GOAL_NOTES[g] for g in goals], horizontal=True, key="pf_goal", width="stretch")
                activity = st.radio("Activity level", [1, 2, 3], index=int(u.get("activity_level", 2)) - 1,
                                    format_func=ACTIVITY_LABELS.get, captions=[ACTIVITY_NOTES[k] for k in (1, 2, 3)],
                                    horizontal=True, key="pf_activity", width="stretch")

            with st.container(key="pfsec_meals"):
                st.markdown('<div class="card-title">Your meals</div>', unsafe_allow_html=True)
                meal_counts = list(MEALS_LABELS)
                meals = st.radio("Meals per day", meal_counts,
                                 index=meal_counts.index(int(u.get("meals_per_day") or DEFAULT_MEALS)),
                                 format_func=MEALS_SHORT.get, captions=[MEALS_NOTES[k] for k in meal_counts],
                                 horizontal=True, key="pf_meals", width="stretch", help="The day's calories are shared over this many meals.")
                st.markdown("**Health focus** (optional)")
                current = parse_modes(u.get("health_modes"))
                chosen = [m for m in HEALTH_MODES
                          if st.checkbox(f"**{MODE_SHORT[m][0]}**  \n{MODE_SHORT[m][1]}", value=m in current, key=f"pfmode_{m}", width="stretch")]
                st.caption("General guidance only. Follow your doctor's advice.")
                egypt = list(EGYPTIAN_OPTIONS)
                egyptian = st.radio("Egyptian dishes", egypt, index=egypt.index(u.get("egyptian_dishes") or "sometimes"),
                                    format_func=EGYPTIAN_OPTIONS.get, horizontal=True, key="pf_egypt", width="stretch",
                                    help="Koshari, ful medames, molokhia, mahshi and more, as whole meals. "
                                         "Sometimes means about one a day.")

            with st.container(key="pfsec_prefs"):
                st.markdown('<div class="card-title">Preferences and allergies</div>', unsafe_allow_html=True)
                prefs_text = st.text_area("Preferences & allergies", value=u.get("preferences_text", ""),
                                          placeholder="e.g. I hate mushrooms and I'm allergic to shrimp. I love grilled chicken.",
                                          help="Write normally. Allergies are always double-checked against an allergen "
                                               "dictionary.")

        with aside:
            with st.container(key="pfsec_submit"):
                if targets:
                    st.markdown(f'<div class="tile-label">This week\'s daily target</div><div class="tile-value">'
                                f'{targets["calories"]:,}<small>kcal</small></div><div class="tile-note">Protein '
                                f'{targets["protein_g"]} g · Carbs {targets["carbs_g"]} g · Fat {targets["fat_g"]} g</div>',
                                unsafe_allow_html=True)
                consent = user is not None or st.checkbox(CONSENT_TEXT, key="pf_consent", width="stretch")
                label = "Generate my weekly plan" if user else "Create profile and weekly plan"
                submitted = st.form_submit_button(label, type="primary", width="stretch", icon=":material/eco:")
                st.caption("Builds a fresh plan for this week. What earlier weeks taught it is kept." if user else
                           "Your 7-day plan is ready in a few seconds.")

    if not submitted:
        return None
    if not name.strip():
        st.error("Please enter your name.")
        return None
    if user is None:
        if not consent:
            st.error("Please tick the box to agree, so that your details can be saved.")
            return None
        if pin != again:
            st.error("The two PINs don't match.")
            return None
        try:
            db.validate_pin(pin)
        except ValueError as e:
            st.error(str(e))
            return None
    profile = UserProfile(name.strip(), int(age), sex, float(weight), float(height), int(activity), goal, prefs_text.strip(),
                          int(meals), "|".join(sorted(chosen)), egyptian)
    return profile, (pin if user is None else None)


def consent_page(user):
    """Accounts made before the consent box existed: agree, sign out, or delete everything."""
    st.markdown(BRAND, unsafe_allow_html=True)
    with st.container(key="card_consent"):
        st.markdown(f'<div class="card-title">One question before you continue, {html.escape(user["name"])}</div>',
                    unsafe_allow_html=True)
        st.markdown(CONSENT_TEXT)
        st.button("I agree and continue", type="primary", key="consent_yes", on_click=db.record_consent, args=(user["id"],),
                  width="stretch")
        st.button("Sign out", key="consent_signout", on_click=set_user, args=(None,), width="stretch")
        delete_account(user)


def delete_account(user):
    """Permanently remove the account. Asks for the PIN, so nobody else at an open browser can do it."""
    with st.expander("Delete my account"):
        st.markdown("This permanently removes your profile, plans, meals, weight entries and ratings. It can't be undone.")
        with st.form("delete_account", border=False):
            pin = st.text_input("Your PIN", type="password", max_chars=8, key="delete_pin")
            sure = st.checkbox("Yes, delete everything", key="delete_sure")
            ok = st.form_submit_button("Delete my account", width="stretch", icon=":material/delete:")
    if ok:
        if not sure:
            st.error("Please tick the box to confirm.")
            return
        result, _ = db.verify_pin(user["name"], pin)
        if result == "ok":
            db.delete_user(user["id"])
            set_user(None)
            st.session_state.account_deleted = True
            st.rerun()
        elif result == "locked":
            st.error(f"Too many wrong PINs. Try again in {db.LOCK_MINUTES} minutes.")
        else:
            st.error("That PIN isn't right. Nothing was deleted.")


def signed_out_page():
    """Welcome: what the app does on the left, Sign in / New user on the right. Returns (profile, pin) or None."""
    if st.session_state.pop("account_deleted", False):
        st.success("Your account and all its data have been deleted.")
    brand, switch = st.columns([3, 1], vertical_alignment="center")
    brand.markdown(BRAND, unsafe_allow_html=True)
    with switch.container(horizontal=True, horizontal_alignment="right"):
        theme_switch()
    left, right = st.columns([1.2, 1], gap="large")
    left.markdown(
        '<span class="hero-tag">Personal weekly meal plans</span>'
        '<div class="hero-title">Seven days of meals that fit your body, your taste and your table.</div>'
        '<p class="hero-sub">Tell NutriPlan about yourself once. It works out your calories, plans every meal from 3,000+ '
        'USDA foods, and gets better each week from what you actually eat.</p>'
        '<ol class="steps">'
        '<li class="step"><div class="step-n">01</div><div class="step-t">Your targets</div><div class="step-d">Your calorie '
        'target comes from the Mifflin-St Jeor formula, your activity level and your goal.</div></li>'
        '<li class="step"><div class="step-n">02</div><div class="step-t">Your words</div><div class="step-d">An AI model '
        '(FoodBaseBERT-NER) reads your preferences. Allergies are double-checked with an allergen dictionary.</div></li>'
        '<li class="step"><div class="step-n">03</div><div class="step-t">Your plate</div><div class="step-d">Meals are built '
        'from 3,000+ USDA foods, with portions solved to hit your calorie and macro targets.</div></li>'
        '<li class="step"><div class="step-n">04</div><div class="step-t">Your feedback</div><div class="step-d">Tick the '
        'meals you ate. Next week\'s plan learns from what you skipped.</div></li></ol>', unsafe_allow_html=True)
    with right.container(key="card_auth"):
        new = st.session_state.get("account_mode") == "New user"
        st.markdown(f'<div class="page-title" style="font-size:1.6rem">{"Start your first week" if new else "Welcome back"}</div>',
                    unsafe_allow_html=True)
        mode = st.radio("Account", ["Sign in", "New user"], horizontal=True, label_visibility="collapsed", key="account_mode",
                        width="stretch")
        if mode == "Sign in":
            sign_in_form()
            return None
        return profile_form(None)


def top_bar(user, plan):
    """Brand, the page menu, the week and the account menu. Returns the open page."""
    if "goto" in st.session_state:
        st.session_state.nav = st.session_state.pop("goto")
    if "nav" not in st.session_state:  # a reload opens the page in the URL (?p=shopping)
        p = str(st.query_params.get("p", "plan")).capitalize()
        st.session_state.nav = p if p in PAGES else "Plan"
    with st.container(key="topbar"):
        brand, nav, right = st.columns([1.1, 3.6, 2.3], vertical_alignment="center")
        brand.markdown(BRAND, unsafe_allow_html=True)
        with nav:
            page = st.segmented_control("Page", PAGES, required=True, key="nav", label_visibility="collapsed")
        with right.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
            theme_switch()
            with st.popover(user["name"], icon=":material/person:"):
                st.markdown(f"Signed in as **{user['name']}**")
                st.button("Edit profile", on_click=go_to, args=("Profile",), key="menu_profile", width="stretch",
                          icon=":material/tune:")
                st.button("Sign out", on_click=set_user, args=(None,), key="menu_signout", width="stretch",
                          icon=":material/logout:")
    if st.query_params.get("p") != page.lower():
        st.query_params["p"] = page.lower()
    return page


# --- Plan page ------------------------------------------------------------------------------------------------
def plan_micro_targets(plan):
    """Nutrient reference values saved with the plan (plans from before Feature #7: from the profile)."""
    if plan["targets"].get("micros"):
        return plan["targets"]["micros"]
    user = db.get_user(plan["user_id"]) or {"sex": "male", "age": 30}
    return micro_targets(user["sex"], user["age"], plan["targets"]["calories"])


def aim_text(target, unit):
    """'at least 29 g' / 'under 1,500 mg' / '' for display."""
    if status(0, target) == "info":
        return ""
    return ("at least " if target["kind"] == "min" else "under ") + fmt(target["value"], unit)


def day_totals_table(tot, targets, nutrients, micros):
    """The day against its targets: calories and macros, then the other nutrients (if known)."""
    def row(name, amount, aim, color, result, css=""):
        return (f'<tr class="{css}"><th scope="row">{name}</th><td><b>{amount}</b></td><td>{aim}</td><td><span class="result">'
                f'<span class="dot" style="background:{color}"></span>{result}</span></td></tr>')
    rows = []
    for m, name, unit in (("calories", "Calories", " kcal"), ("protein_g", "Protein", " g"), ("carbs_g", "Carbs", " g"), ("fat_g", "Fat", " g")):
        diff = tot[m] - targets[m]
        color = "var(--ok)" if abs(diff) <= 0.05 * targets[m] else "var(--close)"
        rows.append(row(name, f"{tot[m]:,.0f}{unit}", f"{targets[m]:,.0f}{unit}", color, delta_words(diff, unit)))
    if nutrients is not None:
        for i, m in enumerate(MICROS):
            label, unit = LABELS[m]
            word, color = STATUS[status(nutrients[m], micros[m])][:2]
            rows.append(row(label, fmt(nutrients[m], unit), aim_text(micros[m], unit), color, word, "first-micro" if i == 0 else ""))
    return ('<div class="tbl-wrap"><table class="tbl tbl-day"><thead><tr><th scope="col">Nutrient</th><th scope="col">This day</th>'
            f'<th scope="col">Target</th><th scope="col">Result</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def meals_done(items):
    meals = items.groupby(["day_index", "slot"])["eaten"].max()
    return int(meals.fillna(0).sum()), len(meals)


def ring_svg(done, total, size=84):
    c = 213.6  # circumference of r=34
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 84 84" role="img" aria-label="{done} of {total} meals eaten">'
            '<circle cx="42" cy="42" r="34" fill="none" style="stroke:var(--green-soft)" stroke-width="9"/>'
            f'<circle cx="42" cy="42" r="34" fill="none" style="stroke:var(--green)" stroke-width="9" stroke-linecap="round" '
            f'stroke-dasharray="{done / max(total, 1) * c:.1f} {c}" transform="rotate(-90 42 42)"/>'
            f'<text x="42" y="47" text-anchor="middle" font-family="Schibsted Grotesk, sans-serif" font-size="17" '
            f'font-weight="700" style="fill:var(--ink)">{done}/{total}</text></svg>')


def show_hello(user, plan, items):
    """Greeting, the week and today's date, and the week's progress ring."""
    done, total = meals_done(items)
    hour = datetime.now().hour
    part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    if done == 0:
        cheer = "Your week is ready. Tick each meal as you eat it and the plan learns what you like."
    elif done < total / 2:
        cheer = f"{total - done} meals to go this week. Steady beats perfect."
    elif done < total:
        cheer = f"Over halfway there with {total - done} meals to go. Your consistency is paying off."
    else:
        cheer = "Every meal ticked. Finish the week to plan the next one."
    left, right = st.columns([2.3, 1], vertical_alignment="center")
    left.markdown(f'<div class="eyebrow">Week {plan["week_number"]} · {date.today():%A, %d %B}</div><div class="page-title">'
                  f'Good {part}, {html.escape(user["name"])}</div><p class="page-intro">{cheer}</p>', unsafe_allow_html=True)
    with right.container(key="card_ring"):
        st.markdown(f'<div class="ring">{ring_svg(done, total)}<div><div class="ring-title">Meals eaten</div>'
                    '<div class="small-muted">This week so far</div></div></div>', unsafe_allow_html=True)
        st.button("Finish the week", type="tertiary", icon=":material/arrow_forward:", key="to_progress",
                  on_click=go_to, args=("Progress",))


def show_targets(targets, micros):
    with st.container(key="card_targets"):
        cal = targets["calories"]
        tiles = [("Calories", f"{cal:,}", "kcal", "var(--green)", "Every day")]
        for m, label, per_g in (("protein_g", "Protein", 4), ("carbs_g", "Carbs", 4), ("fat_g", "Fat", 9)):
            tiles.append((label, f"{targets[m]}", "g", MACRO_COLORS[m], f"{targets[m] * per_g / cal * 100:.0f}% of calories"))
        html_tiles = "".join(f'<div class="tile"><div class="tile-label"><span class="dot" style="background:{c}"></span>{l}'
                             f'</div><div class="tile-value">{v}<small>{u}</small></div><div class="tile-note">{n}</div></div>'
                             for l, v, u, c, n in tiles)
        modes = parse_modes(targets.get("health_modes"))
        mode = ("".join(f'<span class="chip chip-blue">Diet mode: {MODE_SHORT[m][0]}</span>' for m in sorted(modes))
                if modes else "")
        also = "Also every day: " + " · ".join(f"{LABELS[m][0]} {aim_text(micros[m], LABELS[m][1])}"
                                               for m in MICROS if micros[m]["value"] is not None)
        st.markdown(f'<div class="row-between"><span class="card-title">Your daily targets</span><span class="small-muted">'
                    f'BMR {targets["bmr"]:,.0f} kcal · TDEE {targets["tdee"]:,.0f} kcal · BMI {targets["bmi"]}</span></div>'
                    f'<div class="tiles">{html_tiles}</div><div class="also">{mode}<span>{also}</span></div>',
                    unsafe_allow_html=True)
        for w in targets.get("warnings", []):
            st.warning(w)


def show_understood(prefs):
    blocked = [GROUP_LABELS.get(g, g) for g in prefs["block_allergen_groups"]]
    blocked += [FLAG_LABELS.get(f, f) for f in prefs["block_diet_flags"]]
    blocked += [t for t in prefs["block"] if t not in {g.lower() for g in blocked}]

    def box(icon, color, title, words, cls):
        chips = "".join(f'<span class="chip {cls}">{html.escape(w)}</span>' for w in words)
        return (f'<div class="pref-box"><div class="pref-head">{svg(icon, 18, color=color)}{title}</div>'
                f'<div class="chips">{chips or "<span class=empty>Nothing yet</span>"}</div></div>')

    with st.container(key="card_understood"):
        left, right = st.columns([4, 1], vertical_alignment="center")
        left.markdown('<div class="card-title">What I understood from your preferences</div>', unsafe_allow_html=True)
        right.button("Edit preferences", type="tertiary", key="edit_prefs", on_click=go_to, args=("Profile",),
                     width="stretch")
        st.markdown('<div class="pref-grid">' + box("never", "var(--never-icon)", "Never included", sorted(set(blocked)), "chip-never")
                    + box("less", "var(--muted)", "Less often", prefs["avoid"], "chip-less")
                    + box("more", "var(--green)", "More often", prefs["prefer"], "chip-more") + "</div>", unsafe_allow_html=True)
        for clause in prefs.get("unclear", []):
            st.warning(f'Please check this one: "{clause}". I could not tell what you mean, so nothing was left out because of '
                       'it. If it is an allergy or a health matter, write it like "I\'m allergic to peanuts" under '
                       "Edit preferences, and follow your doctor's advice.")
        if prefs.get("corrections"):
            read_as = ", ".join(f'"{typed}" as "{new}"' for typed, new in prefs["corrections"].items())
            st.caption(f"I corrected some spelling: I read {read_as}. If that's wrong, change it under Edit preferences.")


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
        st.button(f"{title} · {s.grams:.0f} g" + (f" · {sub}" if sub else "") + f" · {s.calories:.0f} kcal, P {s.protein_g:.0f}",
                  key=f"{key}_{i}", on_click=do_swap, args=(pick, s, picker_key), width="stretch")


def swap_food(plan, meal, day, d, slot, preferences_text):
    """'Swap a food' popover: pick a food of the meal, then an AI-suggested similar food (3 at a time,
    'Show more options' for the next ones) or type what you would like instead."""
    options = {r.id: r for r in meal.itertuples() if not re.match(NEUTRAL_RE, r.name)}
    if not options:
        return
    with st.popover("Swap a food", icon=":material/swap_horiz:", help="Replace one food of this meal with a similar one.",
                    width="stretch"):
        key = f"swap_{plan['id']}_{d}_{slot}"
        pick = st.selectbox("Food to replace", list(options), index=None, key=key, placeholder="Choose a food",
                            format_func=lambda i: options[i].name)
        if pick is None:
            st.caption("Suggestions are similar foods with a similar macro profile, sized to keep the same calories. "
                       "Your allergies stay excluded, and next week shows the food you swapped away less often.")
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
                st.button("Back to the first suggestions" if page == n_pages - 1 else
                          f"Show more options ({page + 2} of {n_pages})", type="tertiary",
                          key=f"{key}_more", on_click=next_page, args=(page_key, n_pages))

        st.markdown("**Or type what you'd like**")
        query = st.text_input("What would you like instead?", key=f"{key}_query", label_visibility="collapsed",
                              placeholder="e.g. brown rice, potatoes, salmon, then press Enter")
        if query.strip():
            found, best = search_choices(query.strip(), int(r.fdc_id), slot, float(r.calories), preferences_text, day_ids,
                                         modes)
            if not found or best < SEARCH_MIN_MATCH:
                st.warning(f"No close match for “{query.strip()}” among {swap_kind(int(r.fdc_id), slot)} "
                           "that fit your allergies." + (" Closest ones:" if found else ""))
            swap_buttons(found, f"{key}_found", key, pick)


@st.cache_data(show_spinner=False)
def recipes():
    return load_recipes()


def food_html(r):
    """One food of a meal: title + grams, then the details, then the macros.
    An Egyptian dish also shows its Arabic name and what is in this portion."""
    title, sub = split_name(r.name)
    arabic = ARABIC.get(title.lower())
    title_html, sub_html = html.escape(title), html.escape(sub)
    if arabic:
        # <bdi> keeps the right-to-left name from swapping places with the grams that follow it
        title_html += f' <bdi class="food-ar">{html.escape(arabic)}</bdi>'
        line = recipe_line(recipes(), r.fdc_id, r.grams)
        if line:
            sub_html += f'<div class="food-recipe">In this portion: {html.escape(line)}</div>'
    macros = (f'{r.calories:.0f} kcal · <span class="mp">P {r.protein_g:.0f}</span> · <span class="mc">C {r.carbs_g:.0f}</span>'
              f' · <span class="mf">F {r.fat_g:.0f}</span>')
    return (f'<div class="food"><div class="food-top"><span class="food-title">{title_html}</span>'
            f'<span class="food-grams">{r.grams:.0f} g</span></div>'
            + (f'<div class="food-sub">{sub_html}</div>' if sub_html else "")
            + f'<div class="food-macros">{macros}</div></div>')


def meal_head(slot, items, meals):
    """Icon, meal name, its macros and calories, and a bar of where the calories come from."""
    def total(m):
        return sum(getattr(i, m) for i in items) if isinstance(items, list) else items[m].sum()
    kcal, p, c, f = (total(m) for m in MACROS)
    parts = [p * 4, c * 4, f * 9]
    split = "".join(f'<span style="width:{x / (sum(parts) or 1) * 100:.1f}%;background:{col}"></span>'
                    for x, col in zip(parts, MACRO_COLORS.values()))
    return (f'<div class="meal-head"><span class="meal-icon">{svg(SLOT_SVG.get(slot, "bowl"), 22, 1.7)}</span>'
            f'<div><div class="meal-name">{slot_label(slot, meals)}</div><div class="meal-macros">P {p:.0f} g · C {c:.0f} g · '
            f'F {f:.0f} g</div></div><span class="meal-kcal">{kcal:.0f}<small>kcal</small></span></div>'
            f'<div class="split" aria-hidden="true">{split}</div>')


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


def cancel_proposal():
    st.session_state.pop("proposal", None)


def _day_error(day_totals, daily):
    """Mean % distance of a day's calories and macros from the targets."""
    return sum(abs(day_totals[m] - daily[m]) / daily[m] for m in MACROS) / len(MACROS) * 100


def _version_column(title, meals, others, daily, best, key, n_meals):
    """One side of the comparison: the meals, then the day's totals with this version."""
    totals = {m: others[m] + sum(getattr(i, m) for items in meals.values() for i in items) for m in MACROS}
    with st.container(key=key + ("_best" if best else "")):
        st.markdown(f'<div class="cmp-head">{title}' + ('<span class="cmp-best">Closer to your targets</span>' if best else "")
                    + "</div>", unsafe_allow_html=True)
        for slot, items in meals.items():
            st.markdown(f'<div class="cmp-meal">{meal_head(slot, items, n_meals)}{"".join(food_html(i) for i in items)}</div>',
                        unsafe_allow_html=True)
        cells = []
        for m, label in zip(MACROS, ["kcal", "protein", "carbs", "fat"]):
            diff = totals[m] - daily[m]
            near = abs(diff) <= 0.05 * daily[m]
            cells.append(f"<div><b>{totals[m]:,.0f}{'' if m == 'calories' else ' g'}</b>{label} "
                         f'<span class="{"near" if near else "far"}">({diff:+.0f})</span></div>')
        st.markdown(f'<div class="cmp-totals"><b style="color:var(--ink)">Day with this version</b>'
                    f'<div class="cmp-grid">{"".join(cells)}</div></div>', unsafe_allow_html=True)
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


def delta_words(diff, unit=""):
    n = abs(round(diff))
    return "on target" if n == 0 else f"{n:,}{unit} {'over' if diff > 0 else 'under'}"


def show_week(plan, items):
    targets = plan["targets"]
    micros = plan_micro_targets(plan)
    nutrients = day_micros(items, load_resources()[0])
    preferences_text = (db.get_user(plan["user_id"]) or {}).get("preferences_text", "")
    st.subheader(f"Week {plan['week_number']} meal plan")
    tabs = st.tabs(DAYS)
    for d, tab in enumerate(tabs):
        day = items[items["day_index"] == d]
        with tab:
            for i, slot in enumerate(plan_slots(day)):
                if i % 2 == 0:
                    cols = st.columns(2)  # one row per pair, in time order: breakfast | lunch, dinner | snack
                meal = day[day["slot"] == slot]
                with cols[i % 2].container(key=f"mealcard_{plan['id']}_{d}_{slot}"):
                    st.markdown(meal_head(slot, meal, meals_of(plan)), unsafe_allow_html=True)
                    # Fixed height so every meal card is the same size (a long meal scrolls inside its card).
                    with st.container(height=MEAL_BODY_PX, border=False):
                        st.markdown("".join(food_html(r) for r in meal.itertuples()), unsafe_allow_html=True)
                    swap_col, new_col = st.columns(2, vertical_alignment="center", gap="small")
                    with swap_col:
                        swap_food(plan, meal, day, d, slot, preferences_text)
                    if new_col.button("New meal", key=f"newmeal_{plan['id']}_{d}_{slot}", width="stretch",
                                      icon=":material/autorenew:",
                                      help="See a different meal next to this one, and keep the one you prefer."):
                        propose(plan, items, d, slot, preferences_text)
                    key = f"eat_{plan['id']}_{d}_{slot}"
                    eaten = bool(meal["eaten"].fillna(0).max()) if not meal.empty else False
                    st.checkbox("Eaten", value=eaten, key=key, on_change=on_check, args=(plan["id"], d, slot, key),
                                width="stretch")

            with st.container(key=f"card_day_{d}"):
                left, right = st.columns([2.6, 1.3], vertical_alignment="center")
                left.markdown(f'<div class="card-title">{DAYS[d]} totals</div>', unsafe_allow_html=True)
                if right.button("New version of this day", key=f"newday_{plan['id']}_{d}", width="stretch",
                                icon=":material/autorenew:",
                                help="See a different plan for this whole day next to the current one, and keep the one you prefer."):
                    propose(plan, items, d, None, preferences_text)
                st.markdown(day_totals_table(day[MACROS].sum(), targets, nutrients.loc[d] if d in nutrients.index else None, micros),
                            unsafe_allow_html=True)


@st.cache_data(show_spinner="Preparing the PDF...", max_entries=20)
def plan_pdf(plan_info, items, user_name):
    """Printable PDF of the week (Feature #8); rebuilt only when the plan or its ticks change."""
    foods, _ = load_resources()
    return week_pdf(plan_info, items, user_name, foods)


def show_overview(plan, items):
    slots = plan_slots(items)
    overview = (items.groupby(["day_index", "day", "slot"])["name"].apply(lambda s: " + ".join(n.split(",")[0] for n in s))
                .unstack("slot")[slots].reset_index(level=0, drop=True))
    with st.container(key="card_overview"):
        left, b1, b2 = st.columns([2.4, 1.3, 1], vertical_alignment="center")
        left.markdown('<div class="card-title">Week at a glance</div><div class="small-muted">The PDF is printable: every day '
                      'with tick boxes, plus the shopping list.</div>', unsafe_allow_html=True)
        name = (db.get_user(plan["user_id"]) or {}).get("name", "")
        info = {k: plan[k] for k in ("week_number", "targets", "prefs")}
        # Files are made only when a button is clicked (a function instead of the data). Otherwise the browser fetches
        # every file on each page load, and download managers save the PDF on their own at sign-in.
        b1.download_button("Download plan (PDF)", partial(plan_pdf, info, items, name), on_click="ignore",
                           file_name=f"meal_plan_week{plan['week_number']}.pdf", mime="application/pdf", type="primary",
                           width="stretch", icon=":material/download:",
                           help="Printable: every day with tick boxes, plus the shopping list.")
        b2.download_button("Download CSV", partial(items.drop(columns=["id", "plan_id"]).to_csv, index=False),
                           on_click="ignore", file_name="meal_plan.csv", mime="text/csv", width="stretch")
        head = "".join(f"<th scope='col'>{slot_label(s, meals_of(plan))}</th>" for s in slots)
        rows = "".join(f"<tr><th scope='row'>{day}</th>" + "".join(f"<td>{html.escape(str(v)) if pd.notna(v) else ''}</td>"
                                                                  for v in row) + "</tr>"
                       for day, row in overview.iterrows())
        st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead><tr><th scope="col">Day</th>{head}</tr></thead>'
                    f'<tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)


def page_plan(plan, items, user):
    show_hello(user, plan, items)
    show_targets(plan["targets"], plan_micro_targets(plan))
    show_understood(plan["prefs"])
    show_week(plan, items)
    show_overview(plan, items)


# --- Nutrients page (Feature #7) --------------------------------------------------------------------------------
def page_nutrients(plan, items, foods):
    """The week's fiber, sugars, saturated fat, sodium, potassium, calcium and iron."""
    page_header(f"Week {plan['week_number']}", "Nutrients this week",
                "Daily averages across your seven days, next to the targets for adults. Calories and macros are on the "
                "Plan page.")
    s = nutrient_summary(items, foods, plan_micro_targets(plan))
    rows, groups = [], {"ok": [], "close": [], "off": []}
    for r in s.itertuples():
        t = {"value": r.target, "kind": r.kind}
        st_ = status(r.average, t)
        words, dot, bg, fg = STATUS[st_]
        if st_ != "info":
            groups["off" if st_ in ("low", "high") else st_].append(r.label)
            pct = min(r.average / r.target, 1) * 100 if r.target else 0
            bar = f'<span class="bar"><span style="width:{pct:.0f}%;background:{dot}"></span></span>'
        else:
            bar = ""
        rows.append(f'<tr><th scope="row">{r.label}</th><td><span class="chip" style="background:{bg};color:{fg};font-weight:600">'
                    f'<span class="dot" style="background:{dot}"></span>{words}</span></td>'
                    f'<td><div class="barline"><b>{fmt(r.average, r.unit)}</b>{bar}</div></td>'
                    f'<td>{aim_text(t, r.unit) or "no limit (see note)"}</td>'
                    f'<td style="text-align:right">{f"{int(r.days_met)} / {r.days}" if pd.notna(r.days_met) else "-"}</td></tr>')
    n_target = sum(len(v) for v in groups.values())
    tiles = [("On target", STATUS["ok"][1], f'{len(groups["ok"])}<small>of {n_target} with a target</small>', groups["ok"]),
             ("Close", STATUS["close"][1], str(len(groups["close"])), groups["close"]),
             ("A little under or over", STATUS["low"][1], str(len(groups["off"])), groups["off"])]
    st.markdown('<div class="tiles">' + "".join(
        f'<div class="tile" style="background:var(--card);border:1px solid var(--line);padding:18px 20px"><div class="tile-label">'
        f'<span class="dot" style="background:{c}"></span>{l}</div><div class="tile-value">{v}</div>'
        f'<div class="tile-note">{", ".join(names) or "None"}</div></div>' for l, c, v, names in tiles) + "</div>",
        unsafe_allow_html=True)
    with st.container(key="card_nutrients"):
        st.markdown('<div class="card-title">Daily average vs target</div><div class="tbl-wrap"><table class="tbl"><thead><tr>'
                    '<th scope="col">Nutrient</th><th scope="col">Status</th><th scope="col" style="width:32%">Daily average</th>'
                    '<th scope="col">Target per day</th><th scope="col" style="text-align:right">Days on target</th></tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)
    with st.container(key="card_nutrient_notes"):
        st.markdown('<div class="card-title" style="font-size:1.05rem">About these numbers</div>', unsafe_allow_html=True)
        st.markdown("- Targets for adults: fiber 14 g per 1,000 kcal and saturated fat under 10% of calories (Dietary "
                    "Guidelines for Americans). Sodium, potassium, calcium and iron from the US National Academies.\n"
                    "- Sugars include the natural sugar of fruit and milk, so no limit is applied unless the diabetes mode is on.\n"
                    "- Sodium counts the foods only, not the salt you add when cooking (1 teaspoon of salt is about 2,300 mg).\n"
                    "- A few values missing in USDA are estimated from similar foods.")


# --- Shopping page (Feature #3) ----------------------------------------------------------------------------------
def page_shopping(plan, items, foods):
    """The week's shopping list, grouped by shop section, with tick boxes and downloads."""
    head, d1, d2 = st.columns([3, 1, 1], vertical_alignment="bottom")
    with head:
        page_header(f"Week {plan['week_number']}", "Shopping list",
                    "Everything for this week's plan. Rice, pasta, beans and meat show the approximate raw or dry amount to "
                    "buy. Ticks last while this page is open.")
    skip = st.session_state.get(f"shop_skip_{plan['id']}", False)
    shop = shopping_list(items, foods, skip_eaten=skip)
    title = f"NutriPlan shopping list - week {plan['week_number']}"
    if not shop.empty:
        d1.download_button("Text file", partial(to_text, shop, title), on_click="ignore", file_name="shopping_list.txt",
                           mime="text/plain", width="stretch", icon=":material/download:")
        csv = shop.assign(amount=shop["grams"].map(format_amount)).drop(columns=["plate_g"])
        d2.download_button("CSV", partial(csv.to_csv, index=False), on_click="ignore", file_name="shopping_list.csv",
                           mime="text/csv", width="stretch", icon=":material/download:")

    def item_key(r):
        return f"shopitem_{plan['id']}_{r.item}_{r.label}"

    with st.container(key="card_basket"):
        left, right = st.columns([2, 1], vertical_alignment="center")
        if not shop.empty:
            ticked = sum(bool(st.session_state.get(item_key(r))) for r in shop.itertuples())
            left.progress(ticked / len(shop), text=f"**{ticked} of {len(shop)} in the basket** · {len(shop) - ticked} left")
        right.toggle("Leave out meals already eaten", key=f"shop_skip_{plan['id']}")
    if shop.empty:
        st.info("Nothing left to buy: every meal is ticked as eaten.")
        return

    cols, filled = st.columns(3), [0, 0, 0]
    for section, rows in shop.groupby("section", sort=False):
        i = filled.index(min(filled))  # next section goes to the shortest column
        filled[i] += len(rows) + 2
        with cols[i].container(key=f"shopcard_{plan['id']}_{SECTIONS.index(section)}"):
            name = re.sub(r"^\W+\s*", "", section)  # section names start with an emoji
            st.markdown(f'<div class="row-between" style="align-items:center"><span class="card-title" style="font-size:1.05rem">'
                        f'{name}</span><span class="chip chip-less" style="font-weight:600">{len(rows)} items</span></div>',
                        unsafe_allow_html=True)
            for r in rows.itertuples():
                extra = " · ".join(x for x in (r.label, r.note) if x)
                st.checkbox(f"**{r.item}** · {format_amount(r.grams)}" + (f"  \n:gray[{extra}]" if extra else ""),
                            key=item_key(r))


# --- Progress page: finish the week, weight (Feature #6), what was learned ------------------------------------------
def profile_of(user):
    return UserProfile(**{k: user[k] for k in db.PROFILE_FIELDS})


def weight_chart(history, expected):
    """Logged weights (green line + points) and the weight the plan should give (grey dashed line)."""
    line, ring = chart_color(), "#161E1A" if is_dark() else "#FFFFFF"
    hist = history.assign(day=pd.to_datetime(history["day"]))
    base = alt.Chart(hist).encode(x=alt.X("day:T", title=None), y=alt.Y("weight_kg:Q", title="kg", scale=alt.Scale(zero=False)),
                                  tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("weight_kg:Q", title="Weight (kg)")])
    actual = (base.mark_line(color=line, strokeWidth=3)
              + base.mark_point(color=line, filled=True, size=80, stroke=ring, strokeWidth=2, opacity=1))
    plan_line = alt.Chart(expected).mark_line(color="#8E9A95", strokeDash=[6, 5], strokeWidth=2).encode(
        x="day:T", y="expected_kg:Q", tooltip=[alt.Tooltip("day:T", title="Date"),
                                                alt.Tooltip("expected_kg:Q", title="Expected (kg)", format=".1f")])
    return (plan_line + actual).properties(height=260)


def show_weight(plan, user_id, foods, emb):
    user = db.get_user(user_id)
    history = db.weight_history(user_id)
    t = plan["targets"]
    expected = expected_weekly_change(t["calories"], t["tdee"])
    expected = expected if abs(expected) >= 0.005 else 0.0  # "maintain" shows 0.00, not -0.00
    rate = weekly_rate(history)
    now, start = history["weight_kg"].iloc[-1], history["weight_kg"].iloc[0]
    since = pd.to_datetime(history["day"].iloc[0]).strftime("%d %b")

    with st.container(key="card_weight"):
        st.subheader("Weight progress")
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
            st.markdown('<div class="legend" style="font-size:.82rem"><span><span class="dot" style="background:var(--green)">'
                        '</span>Your weight</span><span><span class="dot" style="background:#8E9A95"></span>What the plan '
                        'expects</span></div>', unsafe_allow_html=True)
            st.altair_chart(weight_chart(history, expected_line(history, start, expected)), width="stretch")

        with st.form("log_weight", border=False):
            c1, c2, c3 = st.columns([2, 2, 2], vertical_alignment="bottom")
            weight = c1.number_input("Weight (kg)", 30.0, 300.0, float(now), step=0.1, format="%.1f", key="log_weight_kg")
            day = c2.date_input("Date", value=date.today(), max_value=date.today(), format="DD/MM/YYYY", key="log_weight_day")
            logged = c3.form_submit_button("Log weight", type="primary", width="stretch", icon=":material/add:")
        st.caption("Weigh yourself once a week, at the same time of day (for example in the morning). Water, salt and "
                   "muscle move the number from day to day.")
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
                    f"(this week's plan uses {t['calories']:,} kcal). Next week's plan uses it automatically, or you can "
                    "rebuild this week now.")
            if st.button("Rebuild this week with my new targets", key="rebuild_week", icon=":material/autorenew:"):
                with st.spinner("Building your 7-day plan..."):
                    new_week_plan(user_id, profile_of(user), foods, emb)
                st.rerun()


RATING_LABELS = {1: "1 - not at all", 2: "2", 3: "3 - okay", 4: "4", 5: "5 - very much"}


def rate_week(plan):
    """Optional 30-second survey about the week (real-user evaluation, Feature #2)."""
    saved = db.get_survey(plan["id"]) or {}
    with st.expander("Rate this week (optional, 30 seconds, helps evaluate the app)", expanded=False,
                     icon=":material/star:"):
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
            if st.form_submit_button("Save my rating"):
                db.save_survey(plan["id"], liked, portions, easy, use, comment)
                st.success("Thank you, your rating is saved." + (" (updated)" if saved else ""))


def show_feedback(plan, items, foods, emb):
    done, total = meals_done(items)
    with st.container(key="card_finish"):
        left, right = st.columns([1.7, 1], vertical_alignment="center")
        with left:
            st.subheader("Finish the week")
            st.caption("When the week is over, press the button. Unticked meals count as skipped: next week shows those "
                       "foods (and similar ones) less often, and more of what you ate.")
        clicked = right.button("Finish week and plan next week", type="primary", disabled=done == 0, width="stretch",
                               icon=":material/check:",
                               help="Ends this week and builds the next one." if done else "Tick at least one meal you ate first.")
        st.progress(done / total, text=f"Meals eaten this week: {done} of {total}")
        rate_week(plan)
        if clicked:
            with st.spinner("Learning from this week and building the next one..."):
                finish_week(plan, items, foods, emb)
            st.rerun()


def show_learning(user_id):
    """What the system learned from the previous weeks, and acceptance over time."""
    history = db.user_history(user_id)
    if history.empty:
        return
    last_week = history[history["week_number"] == history["week_number"].max()]
    swaps, choices = db.user_swaps(user_id), db.user_choices(user_id)
    week = last_week["week_number"].max()
    s = week_summary(last_week, swaps=swaps[swaps["week_number"] == week], choices=choices[choices["week_number"] == week])

    def box(title, note, rows):
        lis = "".join(f"<li><span>{html.escape(str(n))}</span><b>{k}</b></li>" for n, k in rows)
        return (f'<div class="pref-box"><div class="pref-head" style="margin:0">{title}</div><div class="small-muted" '
                f'style="margin-bottom:8px">{note}</div><ul class="learn-list">{lis or "<li class=empty>Nothing yet</li>"}</ul></div>')

    with st.container(key="card_learning"):
        st.subheader("What I learned from your previous weeks")
        st.caption(f"Based on week {week}: what you ate, skipped, swapped and picked.")
        st.markdown('<div class="pref-grid">'
                    + box("Showing more like", "Eaten last week", [(n, f"x{k}") for n, k in s["most_eaten"]])
                    + box("Showing less often", "Skipped last week", [(n, f"x{k}") for n, k in s["most_skipped"]])
                    + box("Swapped away or not chosen", "Shown less often",
                          [(n, f"x{k}") for n, k in s["swapped_away"]] + [(n, "not chosen") for n, _ in s["not_chosen"]])
                    + "</div>", unsafe_allow_html=True)
        rates = pd.DataFrame([{"Week": f"Week {w}", "Meals eaten (%)": round(100 * acceptance_rate(g), 1)}
                              for w, g in history.groupby("week_number")])
        st.bar_chart(rates, x="Week", y="Meals eaten (%)", height=220, color=chart_color())


def page_progress(plan, items, user_id, foods, emb):
    history = db.weight_history(user_id)
    intro = "Finish the week, log your weight and see what your plan has learned."
    if len(history) > 1:
        change = history["weight_kg"].iloc[-1] - history["weight_kg"].iloc[0]
        since = pd.to_datetime(history["day"].iloc[0]).strftime("%d %B")
        intro = (f"{abs(change):.1f} kg {'down' if change < 0 else 'up'} since {since}. Small, steady steps." if abs(change) >= 0.05
                 else f"Holding steady since {since}.")
    page_header(f"Week {plan['week_number']}", "Your progress", intro)
    show_feedback(plan, items, foods, emb)
    if not history.empty:
        show_weight(plan, user_id, foods, emb)
    show_learning(user_id)


def new_week_plan(user_id, profile, foods, emb):
    """Build and save a new week for a user from their profile, keeping what earlier weeks taught us."""
    taste, penalties = learned_inputs(user_id, foods)
    plan, targets, prefs, seed, taste = build_plan(profile, foods, emb, taste, penalties)
    db.save_plan(user_id, plan, targets, prefs, seed, taste)


def no_plan_yet():
    with st.container(key="card_empty"):
        st.markdown('<div class="card-title">Let\'s build your first week</div><p class="page-intro">Check your details on '
                    'the Profile page and press <b>Generate my weekly plan</b>. It takes a few seconds.</p>',
                    unsafe_allow_html=True)
        st.button("Go to my profile", type="primary", key="to_profile", on_click=go_to, args=("Profile",),
                  icon=":material/arrow_forward:")


def save_profile(submitted, foods, emb):
    """Create the user (new) or update the profile, then build this week's plan and open it."""
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
        go_to("Plan")
        st.rerun()
    except ValueError as e:
        st.error(str(e))


def main():
    foods, emb = load_resources()
    db.init_db()
    st.html(CSS + (DARK_CSS if is_dark() else ""))
    if "apply_theme" in st.session_state:  # the Dark mode switch was flipped: save the choice and reload
        st.html(THEME_JS % st.session_state.pop("apply_theme"), unsafe_allow_javascript=True)

    user_id = current_user_id()
    user = db.get_user(user_id) if user_id else None
    if user is None:
        submitted = signed_out_page()
        if submitted is not None:
            save_profile(submitted, foods, emb)
        st.markdown(FOOTER, unsafe_allow_html=True)
        return

    if not user.get("consented_at"):
        consent_page(user)
        st.markdown(FOOTER, unsafe_allow_html=True)
        return

    plan = db.latest_plan(user_id)
    page = top_bar(user, plan)
    if page == "Profile":
        page_header("Your profile", "Your details", "Change anything here and generate a new plan. Your name stays the same.")
        with st.container(key="page_profile"):
            submitted = profile_form(user, plan["targets"] if plan else None)
            delete_account(user)
        if submitted is not None:
            save_profile(submitted, foods, emb)
    elif plan is None:
        no_plan_yet()
    else:
        items = db.plan_items(plan["id"])
        if page == "Plan":
            page_plan(plan, items, user)
        elif page == "Nutrients":
            page_nutrients(plan, items, foods)
        elif page == "Shopping":
            page_shopping(plan, items, foods)
        else:
            page_progress(plan, items, user_id, foods, emb)
        if st.session_state.get("proposal", {}).get("plan_id") == plan["id"]:
            compare_dialog()
    st.markdown(FOOTER, unsafe_allow_html=True)


main()
