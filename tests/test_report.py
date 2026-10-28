"""Feature #8 tests: the printable PDF of the week's plan.

Run:  python -m pytest tests -q
"""
import re
from dataclasses import asdict

import pandas as pd
import pytest

from src.calculator import UserProfile
from src.config import FOODS_CLEAN
from src.planner import generate_week_plan
from src.report import latin1, week_pdf


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


@pytest.fixture(scope="module")
def week(foods):
    user = UserProfile("Salma", 28, "female", 62, 165, 2, "lose", "I'm allergic to peanuts and I love lentils")
    plan, targets, prefs = generate_week_plan(user, foods=foods, seed=4)
    info = {"week_number": 3, "targets": asdict(targets), "prefs": prefs.summary()}
    return info, plan.to_dataframe().assign(eaten=None)


def text_of(pdf_bytes):
    """Visible text of an uncompressed fpdf2 file (strings drawn with the Tj operator)."""
    raw = pdf_bytes.decode("latin-1")
    return " ".join(m.replace("\\(", "(").replace("\\)", ")") for m in re.findall(r"\((.*?)(?<!\\)\) Tj", raw))


def test_pdf_has_every_day_and_the_shopping_list(foods, week):
    info, items = week
    pdf = week_pdf(info, items, "Salma", foods, compress=False)
    assert pdf.startswith(b"%PDF")
    text = text_of(pdf)
    for word in ["NutriPlan AI", "Week 3 meal plan for Salma", "Daily targets", "peanut", "lentil",
                 "Monday", "Sunday", "Breakfast", "Snack", "Shopping list", "Vegetables", "Cooking oil"]:
        assert word in text, word
    assert text.count("[   ]") >= 28                      # one box per meal + one per shopping item
    pages = len(re.findall(rb"/Type /Page\b", pdf))
    assert 4 <= pages <= 9                                 # two days per page + shopping list


def test_eaten_meals_are_pre_ticked(foods, week):
    info, items = week
    items = items.copy()
    items.loc[(items["day_index"] == 0) & (items["slot"] == "lunch"), "eaten"] = 1
    assert text_of(week_pdf(info, items, "Salma", compress=False)).count("[ x ]") == 1


def test_without_foods_no_shopping_page(week):
    info, items = week
    assert "Shopping list" not in text_of(week_pdf(info, items, "Salma", compress=False))


def test_non_latin_text_does_not_break_the_pdf(foods, week):
    info, items = week
    assert latin1("≈ 4 eggs — 🥚 ok") == "~ 4 eggs - ok"
    text = text_of(week_pdf(info, items, "محمد 🥗", foods, compress=False))
    assert "plan for you" in text                            # name in another script falls back to "you"
