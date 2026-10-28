"""Feature #6: weight tracking - expected vs actual weekly change, and pace warnings.

Expected change on a plan (rule of thumb used in the report):
    kg per week = (daily calorie target - TDEE) * 7 / 7,700      (about 7,700 kcal per kg of body fat)
Actual pace = slope of a straight line through the last 4 weeks of logged weights (needs >= 2 entries 7+ days apart).
Safe pace (common clinical guidance): losing at most about 1% of body weight a week; gaining at most about 0.5%.

Run a demo:  python -m src.progress
"""
from datetime import date

import numpy as np
import pandas as pd

KCAL_PER_KG = 7700
RATE_WINDOW_DAYS = 28        # actual pace is measured over the last 4 weeks
MIN_SPAN_DAYS = 7            # ... once the entries cover at least a week
MAX_LOSS_PER_WEEK = 0.01     # share of body weight
MAX_GAIN_PER_WEEK = 0.005


def expected_weekly_change(calories, tdee):
    """kg per week the plan should give (negative = losing)."""
    return (calories - tdee) * 7 / KCAL_PER_KG


def weekly_rate(history):
    """Actual kg per week from logged weights (columns day, weight_kg), or None if there isn't a week of data."""
    if history is None or len(history) < 2:
        return None
    days = pd.to_datetime(history["day"])
    recent = history[days >= days.max() - pd.Timedelta(days=RATE_WINDOW_DAYS)]
    span = (pd.to_datetime(recent["day"]).max() - pd.to_datetime(recent["day"]).min()).days
    if len(recent) < 2 or span < MIN_SPAN_DAYS:
        return None
    x = (pd.to_datetime(recent["day"]) - pd.to_datetime(recent["day"]).min()).dt.days.to_numpy(float)
    slope = np.polyfit(x, recent["weight_kg"].to_numpy(float), 1)[0]  # kg per day
    return float(slope * 7)


def pace_warning(rate, weight_kg):
    """A warning when the actual pace is faster than the safe guide, else None."""
    if rate is None:
        return None
    if rate < -MAX_LOSS_PER_WEEK * weight_kg:
        return (f"You are losing about {abs(rate):.1f} kg a week - faster than the usual safe pace (up to about "
                f"{MAX_LOSS_PER_WEEK * weight_kg:.1f} kg, 1% of your weight). Consider eating a bit more and talk to a "
                "doctor or dietitian.")
    if rate > MAX_GAIN_PER_WEEK * weight_kg:
        return (f"You are gaining about {rate:.1f} kg a week - faster than the usual pace for muscle gain (up to "
                f"about {MAX_GAIN_PER_WEEK * weight_kg:.1f} kg). A smaller surplus keeps more of it muscle.")
    return None


def expected_line(history, start_weight, rate_per_week, until=None):
    """Weight the plan should give on each day from the first entry to `until` (for the chart)."""
    if history is None or history.empty:
        return pd.DataFrame(columns=["day", "expected_kg"])
    start = pd.to_datetime(history["day"]).min()
    end = pd.Timestamp(until or max(pd.to_datetime(history["day"]).max(), pd.Timestamp(date.today())))
    days = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"day": days, "expected_kg": start_weight + rate_per_week * (days - start).days / 7})


if __name__ == "__main__":
    hist = pd.DataFrame({"day": ["2026-10-01", "2026-10-08", "2026-10-15", "2026-10-22"], "weight_kg": [80, 79.4, 78.9, 78.3]})
    print("expected on 2,200 kcal with TDEE 2,700:", round(expected_weekly_change(2200, 2700), 2), "kg/week")
    print("actual:", round(weekly_rate(hist), 2), "kg/week;", pace_warning(weekly_rate(hist), 78.3))
