"""Feature #8: printable PDF of the week's plan (fpdf2, built-in Helvetica font, A4 portrait).

  Page 1      name, week, daily targets, what the parser understood, notes
  Then        one block per day: every meal with its foods, grams and macros, an empty box to tick when eaten
              (already-eaten meals are pre-ticked), and the day's totals vs the targets
  Last page   the shopping list (src/shopping.py) with tick boxes

Printer-friendly: white paper, dark text, red accents. Built-in fonts cover Latin-1 only, so emoji and a few
symbols are replaced (e.g. "≈" -> "~").

Run a demo:  python -m src.report        (writes data/plan_demo.pdf; git ignores it)
"""
import re
from datetime import date

from fpdf import FPDF
from fpdf.enums import TableCellFillMode, XPos, YPos
from fpdf.fonts import FontFace

from src.dishes import load_recipes, recipe_line
from src.nutrients import HEALTH_MODES, LABELS, MICROS, day_micros, fmt, parse_modes, status, target_text
from src.planner import DAYS, MACROS, plan_slots, slot_label, split_name
from src.shopping import format_amount, shopping_list

RED = (230, 57, 70)
INK = (29, 29, 29)
MUTED = (115, 115, 115)
LIGHT = (246, 246, 246)
LINE = (210, 210, 210)
MACRO_LABELS = {"calories": "kcal", "protein_g": "Protein", "carbs_g": "Carbs", "fat_g": "Fat"}
GROUP_NAMES = {"milk": "milk / dairy", "tree_nut": "tree nuts", "meat": "meat (vegetarian)", "meat_not_fish": "meat except fish"}
HEAD = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=RED)
WHITE = FontFace(fill_color=(255, 255, 255))  # meal-name and tick-box cells are not striped
SYMBOLS = {"≈": "~", "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "×": "x", "…": "..."}


def latin1(text):
    """Make text printable with the built-in fonts (Latin-1): swap common symbols, drop emoji."""
    text = "".join(SYMBOLS.get(ch, ch) for ch in str(text))
    return re.sub(r"\s{2,}", " ", text.encode("latin-1", "ignore").decode("latin-1")).strip()


def fmt_macro(m, value):
    return f"{value:,.0f}" if m == "calories" else f"{value:.0f} g"


class PlanPDF(FPDF):
    def __init__(self, title):
        super().__init__(orientation="portrait", unit="mm", format="A4")
        self.doc_title = latin1(title)
        self.set_title(self.doc_title)
        self.set_author("NutriPlan AI")
        self.set_margins(14, 14, 14)
        self.set_auto_page_break(auto=True, margin=16)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, self.doc_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*RED)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, "NutriPlan AI - for education only, not medical advice.")
        self.cell(0, 5, f"Page {self.page_no()} / {{nb}}", align="R")

    # --- small building blocks --------------------------------------------------------------------------
    def heading(self, text, size=15, space=2):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(*INK)
        self.cell(0, size * 0.5, latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(space)

    def note(self, text, size=8.5):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.2, latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def stat_boxes(self, values):
        """Row of boxes: label on top, big value below (like the app's metric cards)."""
        gap, n = 3, len(values)
        w = (self.epw - gap * (n - 1)) / n
        x0, y0 = self.l_margin, self.get_y()
        for i, (label, value) in enumerate(values):
            x = x0 + i * (w + gap)
            self.set_fill_color(*LIGHT)
            self.set_draw_color(*LINE)
            self.rect(x, y0, w, 17, style="DF")
            self.set_fill_color(*RED)
            self.rect(x, y0, w, 1.2, style="F")
            self.set_xy(x + 3, y0 + 3)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*MUTED)
            self.cell(w - 6, 4, latin1(label))
            self.set_xy(x + 3, y0 + 8)
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*INK)
            self.cell(w - 6, 7, latin1(value))
        self.set_xy(x0, y0 + 21)


def _day_block(pdf, day_name, day, targets, meals_per_day, nutrients=None, recipes=None):
    """One day: title + totals, then a table (meal | food | amount | kcal | P | C | F | eaten box)."""
    rows = len(day) + 1
    if pdf.will_page_break(21 + rows * 6.4):  # keep a day on one page
        pdf.add_page()
    tot = day[MACROS].sum()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*RED)
    pdf.cell(40, 7, day_name)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 7, latin1("Day total: " + " · ".join(
        f"{MACRO_LABELS[m]} {fmt_macro(m, tot[m])} ({tot[m] - targets[m]:+.0f})" for m in MACROS)),
        align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if nutrients is not None and targets.get("micros"):
        marks = {"ok": "ok", "close": "nearly", "low": "low", "high": "high", "info": ""}
        parts = []
        for m in MICROS:
            label, unit = LABELS[m]
            mark = marks[status(nutrients[m], targets["micros"][m])]
            parts.append(f"{label} {fmt(nutrients[m], unit)}" + (f" ({mark})" if mark else ""))
        pdf.set_font("Helvetica", "", 7.5)
        pdf.cell(0, 4.5, latin1(" · ".join(parts)), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    pdf.set_draw_color(*LINE)
    pdf.set_fill_color(255, 255, 255)  # plain rows are white (fpdf2 reuses the last fill colour otherwise)
    pdf.set_text_color(*INK)
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(col_widths=(19, 98, 14, 12, 10, 10, 10, 13), line_height=4.2, padding=(1.1, 1.4), markdown=True,
                   text_align=("LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "CENTER"),
                   headings_style=HEAD, cell_fill_color=LIGHT, cell_fill_mode=TableCellFillMode.ROWS,
                   borders_layout="HORIZONTAL_LINES", v_align="MIDDLE") as table:
        table.row(["Meal", "Food", "Amount", "kcal", "P", "C", "F", "Eaten"])
        for slot in plan_slots(day):
            meal = day[day["slot"] == slot]
            if meal.empty:
                continue
            eaten = bool(meal["eaten"].fillna(0).max()) if "eaten" in meal else False
            for i, r in enumerate(meal.itertuples()):
                row = table.row()
                if i == 0:
                    row.cell(f"**{slot_label(slot, meals_per_day)}**\n{meal['calories'].sum():.0f} kcal", rowspan=len(meal),
                             style=WHITE)
                title, sub = split_name(r.name)
                title, sub = latin1(title).replace("*", ""), latin1(sub).replace("*", "")
                line = recipe_line(recipes, r.fdc_id, r.grams) if recipes is not None else ""
                if line:  # an Egyptian dish: what is in this portion
                    sub = f"{sub}\nIngredients: {latin1(line)}"
                row.cell(f"**{title}**" + (f"  {sub}" if sub else ""))
                row.cell(f"{r.grams:.0f} g")
                for m in MACROS:
                    row.cell(f"{getattr(r, m):.0f}")
                if i == 0:
                    row.cell("[ x ]" if eaten else "[   ]", rowspan=len(meal), style=WHITE)
    pdf.ln(5)


def _shopping_page(pdf, items, foods):
    pdf.add_page()
    pdf.heading("Shopping list", 15, 1)
    pdf.note("Everything for the week. Rice, pasta, beans and meat show the approximate raw / dry amount to buy.")
    pdf.ln(2)
    shop = shopping_list(items, foods)
    for section, rows in shop.groupby("section", sort=False):
        if pdf.will_page_break(10 + 5.6 * (len(rows) + 1)):
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(*INK)
        pdf.cell(0, 6, latin1(section), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_draw_color(*LINE)
        pdf.set_fill_color(255, 255, 255)
        with pdf.table(col_widths=(10, 92, 26, 54), line_height=4.4, padding=1.1, headings_style=HEAD,
                       cell_fill_color=LIGHT, cell_fill_mode=TableCellFillMode.ROWS,
                       text_align=("CENTER", "LEFT", "RIGHT", "LEFT"), borders_layout="HORIZONTAL_LINES") as table:
            table.row(["", "Item", "Amount", "Note"])
            for r in rows.itertuples():
                table.row(["[   ]", latin1(r.item), format_amount(r.grams),
                           latin1(" · ".join(x for x in (r.label, r.note) if x))])
        pdf.ln(3)


def week_pdf(plan, items, user_name, foods=None, compress=True):
    """PDF bytes for one saved week. plan: db.latest_plan(...) dict; items: db.plan_items(plan id).
    foods: the food table, needed for the shopping-list page (left out when None).
    compress=False keeps the text readable in the file (used by the tests)."""
    targets, prefs = plan["targets"], plan.get("prefs", {})
    user_name = latin1(user_name) or "you"  # e.g. an Arabic name can't be drawn with the built-in fonts
    pdf = PlanPDF(f"NutriPlan AI - week {plan['week_number']} plan for {user_name}")
    pdf.set_compression(compress)

    pdf.add_page()
    pdf.set_fill_color(*RED)
    pdf.rect(pdf.l_margin, pdf.get_y(), 2, 15, style="F")
    pdf.set_x(pdf.l_margin + 5)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*INK)
    pdf.cell(0, 9, "NutriPlan AI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(pdf.l_margin + 5)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, latin1(f"Week {plan['week_number']} meal plan for {user_name} · printed {date.today():%d %b %Y}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    pdf.heading("Daily targets", 12, 1)
    pdf.stat_boxes([("Calories", f"{targets['calories']:,} kcal"), ("Protein", f"{targets['protein_g']} g"),
                    ("Carbs", f"{targets['carbs_g']} g"), ("Fat", f"{targets['fat_g']} g")])
    pdf.note(f"BMR {targets['bmr']:,.0f} kcal · TDEE {targets['tdee']:,.0f} kcal · BMI {targets['bmi']}")
    modes = parse_modes(targets.get("health_modes"))
    if modes:
        pdf.note("Diet mode: " + " · ".join(HEALTH_MODES[m] for m in sorted(modes)))
    if targets.get("micros"):
        pdf.note("Also every day: " + " · ".join(f"{LABELS[m][0]} {target_text(targets['micros'][m], LABELS[m][1])}"
                                                 for m in MICROS if targets["micros"][m]["value"] is not None))
    for w in targets.get("warnings", []):
        pdf.note("Note: " + w)
    pdf.ln(3)

    never = [GROUP_NAMES.get(g, g.replace("_", " ")) for g in prefs.get("block_allergen_groups", []) + prefs.get("block_diet_flags", [])]
    never += [t for t in prefs.get("block", []) if t not in never]
    pdf.heading("What I understood from your preferences", 12, 1)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*INK)
    for label, values in (("Never included", never), ("Less often", prefs.get("avoid", [])), ("More often", prefs.get("prefer", []))):
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(32, 5.5, label)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(0, 5.5, latin1(", ".join(sorted(set(values))) or "-"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for clause in prefs.get("unclear", []):
        pdf.note(latin1(f'Please check: "{clause}". I could not tell what this means, so nothing was left out because of it.'))
    pdf.ln(3)
    pdf.note("How to use it: tick the [ ] box of every meal you eat, then tick the same meals in the app before "
             "pressing \"Finish week\" - next week's plan learns from what you ate and skipped. Grams are cooked / "
             "as-served weights.")
    pdf.ln(4)

    recipes = load_recipes()
    for d, day_name in enumerate(DAYS):
        day = items[items["day_index"] == d]
        if not day.empty:
            nut = day_micros(day, foods).iloc[0] if foods is not None else None
            _day_block(pdf, day_name, day, targets, targets.get("meals_per_day"), nut, recipes)

    if foods is not None:
        _shopping_page(pdf, items, foods)
    return bytes(pdf.output())


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from dataclasses import asdict

    import pandas as pd

    from src.calculator import UserProfile
    from src.config import DATA_DIR, FOODS_CLEAN
    from src.planner import generate_week_plan
    foods = pd.read_csv(FOODS_CLEAN)
    user = UserProfile("Demo", 30, "male", 75, 175, 2, "maintain", "I'm allergic to shrimp and I love chicken")
    week, targets, prefs = generate_week_plan(user, foods=foods, seed=1)
    plan = {"week_number": 1, "targets": asdict(targets), "prefs": prefs.summary()}
    out = DATA_DIR / "plan_demo.pdf"
    out.write_bytes(week_pdf(plan, week.to_dataframe().assign(eaten=None), user.name, foods))
    print(f"Saved {out}")
