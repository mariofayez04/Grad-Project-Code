"""Phase 2 tests. Expected values worked out by hand from the Mifflin-St Jeor formula.

Run:  python -m pytest tests -q
"""
import pytest

from src.calculator import (
    UserProfile, adjust_for_goal, apply_activity, apply_floor, calculate_bmr, calculate_targets, macro_split,
)


def test_bmr_male():
    # 10*80 + 6.25*180 - 5*25 + 5 = 1805
    assert calculate_bmr("male", 25, 80, 180) == pytest.approx(1805)


def test_bmr_female():
    # 10*60 + 6.25*165 - 5*30 - 161 = 1320.25
    assert calculate_bmr("female", 30, 60, 165) == pytest.approx(1320.25)


@pytest.mark.parametrize("level,expected", [(1, 2166.0), (2, 2797.75), (3, 3113.625)])
def test_activity(level, expected):
    assert apply_activity(1805, level) == pytest.approx(expected)


@pytest.mark.parametrize("goal,expected", [("maintain", 2000), ("lose", 1600), ("gain", 2200), ("Lose weight", 1600)])
def test_goal_adjustment(goal, expected):
    assert adjust_for_goal(2000, goal) == pytest.approx(expected)


def test_floor():
    assert apply_floor(1100, "female") == (1200, True)
    assert apply_floor(1400, "male") == (1500, True)
    assert apply_floor(1800, "male") == (1800, False)


def test_macro_split_maintain():
    # 2000 kcal: 25% protein = 500/4 = 125 g, 45% carbs = 900/4 = 225 g, 30% fat = 600/9 = 66.7 g
    assert macro_split(2000, "maintain") == {"protein_g": 125, "carbs_g": 225, "fat_g": 67}


@pytest.mark.parametrize("goal", ["maintain", "lose", "gain"])
def test_macros_add_up_to_calories(goal):
    m = macro_split(2500, goal)
    total = 4 * m["protein_g"] + 4 * m["carbs_g"] + 9 * m["fat_g"]
    assert total == pytest.approx(2500, abs=15)  # small rounding error only


def test_full_pipeline_male_lose():
    t = calculate_targets(UserProfile("A", 25, "male", 80, 180, 2, "lose"))
    assert t.tdee == pytest.approx(2797.8, abs=0.1)
    assert t.calories == 2238  # 2797.75 * 0.8
    assert not t.floor_applied
    # 35% protein would be 196 g (2.45 g/kg) -> capped at 2.2 * 80 = 176 g, the extra 20 g goes to carbs
    assert t.protein_g == 176
    assert t.carbs_g == 216


def test_protein_cap_keeps_calories():
    m = macro_split(2238, "lose", weight_kg=80)
    assert 4 * m["protein_g"] + 4 * m["carbs_g"] + 9 * m["fat_g"] == pytest.approx(2238, abs=10)


def test_protein_cap_not_applied_when_below():
    assert macro_split(2000, "maintain", weight_kg=80) == macro_split(2000, "maintain")


def test_full_pipeline_floor_and_warning():
    # small, older, sedentary woman trying to lose: 1057.75 * 1.2 * 0.8 = 1015 -> raised to 1200
    t = calculate_targets(UserProfile("B", 50, "F", 50, 155, 1, "lose weight"))
    assert t.calories == 1200 and t.floor_applied
    assert any("safe minimum" in w for w in t.warnings)


def test_underweight_lose_warning():
    t = calculate_targets(UserProfile("C", 22, "female", 45, 170, 2, "lose"))
    assert any("underweight" in w for w in t.warnings)


def test_per_meal_sums_to_daily():
    t = calculate_targets(UserProfile("D", 30, "male", 75, 175, 2, "maintain"))
    assert sum(m["calories"] for m in t.per_meal().values()) == pytest.approx(t.calories, abs=2)


@pytest.mark.parametrize("field,value", [("age", 10), ("weight_kg", 10), ("height_cm", 300), ("activity_level", 4)])
def test_invalid_input(field, value):
    p = UserProfile("E", 30, "male", 75, 175, 2, "maintain")
    setattr(p, field, value)
    with pytest.raises(ValueError):
        calculate_targets(p)


def test_invalid_sex_and_goal():
    with pytest.raises(ValueError):
        calculate_targets(UserProfile("F", 30, "other", 75, 175, 2, "maintain"))
    with pytest.raises(ValueError):
        calculate_targets(UserProfile("G", 30, "male", 75, 175, 2, "bulk"))
