---
title: NutriPlan AI
emoji: 🥗
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
short_description: AI personalized weekly meal plans from USDA food data
---

# NutriPlan AI: Personalized Nutrition Recommendation System

Graduation project, Faculty of Computing. Supervisor: Dr Eid Emary.

A web app that turns a short profile and a free-text sentence like
*"I hate mushrooms and I'm allergic to shrimp"* into a 7-day meal plan that hits the user's calorie and
macro targets, and learns from the meals the user actually eats.

> ⚠️ For education only, not medical advice.

## What it does

| Input | Output |
|---|---|
| Name + PIN (private sign-in), age, biological sex, weight, height, activity level (1-3), goal, meals per day (3-5), optional diet mode (diabetes-friendly, low-salt) | Daily calorie target and macro split (g), plus fiber, sugars, saturated fat, sodium, potassium, calcium and iron |
| Free-text preferences and allergies | 7-day plan with 3, 4 or 5 meals a day and gram portions, incl. 20 Egyptian dishes (koshari, ful, molokhia...) |
| "Eaten" tick under every meal | Next week's plan adapted to what was eaten and skipped |
| "Log weight" (Progress page) | Weight chart vs the plan's expected trend, pace per week, targets that follow your weight |
| "Download plan (PDF)" (Plan page) | Printable week: every day with tick boxes, plus the shopping list |
| "New meal" / "New version of this day" | A different meal or day shown next to the current one; you keep the one you prefer and next week learns from it |
| "Swap a food" under every meal | Similar foods (embeddings + macro profile) 3 at a time with "Show more", or type what you would like; sized to keep the calories; swapped-away foods appear less next week |
| "Dark mode" switch (top bar) | The whole app in a dark theme; the choice is remembered in the browser |

The app has a top menu with five pages: **Plan** (targets, the week's meals, week at a glance), **Nutrients**,
**Shopping** (list with tick boxes and downloads), **Progress** (finish the week, rating, weight, what it learned) and
**Profile**.

## How it works

```
profile ──► calculator.py ──► daily kcal + macros (Mifflin-St Jeor, activity, goal, safety floor, protein cap)
free text ─► parser.py ─────► block / avoid / prefer   (FoodBaseBERT-NER + rules + allergen dictionary)
USDA SR Legacy ─► clean_data.py ─► foods_clean.csv (3,184 foods + 20 Egyptian dishes, allergen + diet tags, meal roles)
foods ──► embeddings.py ────► 384-d vectors (all-MiniLM-L6-v2), computed once
            planner.py ─────► 7-day plan: score candidates, combine 2-4 foods, solve grams (bounded least squares)
                              + swap a food: same meal component, nearest in embedding + macro space
checkboxes ► feedback.py ───► taste vector + food / family penalties for next week
            app.py ─────────► Streamlit UI, SQLite storage
```

**Where AI is used (and where it isn't):**
- Classical: BMR/TDEE formulas, macro split, filtering, portion optimisation.
- AI: food-entity extraction from free text (BERT NER), semantic food similarity (sentence embeddings),
  a per-user taste vector learned from weekly feedback (vector arithmetic on pre-trained embeddings, no training).
- Safety: allergies never rely on the model alone; a deterministic allergen dictionary always double-checks.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Rebuild the data from the raw USDA download (path in `src/config.py`):

```bash
python -m src.clean_data      # Phase 1: foods_clean.csv
python -m src.embeddings      # Phase 4: embeddings.npy
```

## Run with Docker

```bash
docker compose up -d --build     # first build takes ~5-10 min (downloads PyTorch + both models)
docker compose logs -f           # watch the logs (Ctrl+C to stop watching)
docker compose down              # stop
```

Open http://localhost:8501. The project's `data/` folder is mapped into the container, so the
database (`data/app.db`) stays on your PC across restarts and is shared with a local `streamlit run`.
The port is bound to `127.0.0.1`, so only this PC can reach the app.

## Tests and evaluation

```bash
python -m pytest tests -q     # 281 tests (~5 min)
python -m src.evaluate        # writes data/evaluation/results.md (~10-15 min)
python -m src.simulate        # feedback-loop experiment only
python -m src.heldout         # held-out sentences only
```

See [data/evaluation/results.md](data/evaluation/results.md) for the numbers used in the report.

### Testing with real people (sections 11-12 of the results)

1. **Held-out sentences (no app needed, ~10 min each).** Run `python -m src.heldout --template` and send
   `data/evaluation/heldout_sentences.xlsx` to 5-10 people who did not build the app. They write how they would
   describe their likes, dislikes and allergies, and what the app should understand (Instructions tab). Put the
   filled sheets' rows into that one file in `data/evaluation/` (keep the header), then run `python -m src.evaluate`.
2. **One-week pilot (the app).** Each tester makes a profile (name + PIN), follows the plan for a week, ticks the
   meals they eat, opens **"Rate this week"** in the "Finish the week" section of the Progress page, then presses
   **Finish week**.
   A second week shows whether the plan learned. `python -m src.evaluate` then reports meals eaten per week, swaps,
   New meal / New day choices, weight entries and the ratings - aggregated, without names (comments go to
   `pilot_comments.csv` without names).
3. Tell testers the app is for education only, ask for their OK to use their anonymous answers in the report, and
   run it on one PC (or the local Docker app) so everyone's data is in the same `data/app.db`.

## Host on Vercel

The project is ready for Vercel's Container Runtime (`Dockerfile`, `vercel.json`). Vercel doesn't keep files on its
servers, so accounts and plans need a hosted Postgres database. Without one, every restart deletes them.

1. Create a free Postgres database, for example at neon.tech, in a European region (the app runs in Frankfurt), and
   copy its connection string.
2. Import this GitHub repository in Vercel and set these environment variables:
   - `PORT` = `8501`
   - `DATABASE_URL` = the connection string
3. Deploy, and use the production address (preview deployments shut down after 30 seconds without traffic).

With `DATABASE_URL` set the app creates its tables by itself and uses Postgres. Without it, it uses `data/app.db`.

Note: Vercel's free plan closes the live connection after 5 minutes and stops idle servers. The page reconnects and
stays signed in, but it can take about 20 seconds when a new server has to start, and a half-filled form is lost.

To run the tests on Postgres, point them at a throwaway database (every test wipes it):
`TEST_DATABASE_URL=postgresql://... python -m pytest tests -q`

## Deploy for free (Hugging Face Spaces)

1. Create a free account at huggingface.co and a new **Space** → SDK **Docker** → blank template.
2. Create an access token (Settings → Access Tokens, type *Write*) and log in once
   (the `hf` tool is already installed in the project's virtual environment):
   ```bash
   hf auth login
   ```
3. Upload the project to the Space (the upload tool stores binary files such as `embeddings.npy` correctly;
   a plain `git push` of binaries is rejected by Hugging Face):
   ```bash
   hf upload <your-username>/<space-name> . --repo-type space --exclude ".venv/*" --exclude ".git/*" --exclude "data/app.db" --exclude "notebooks/*" --exclude "*__pycache__*"
   ```
4. The Space builds the `Dockerfile` (about 5-10 minutes) and the app appears at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

Note: free Spaces have no persistent disk, so the SQLite database resets when the Space restarts.
That is fine for a demo; for the defense, running locally keeps all history.

## Project structure

```
app.py                Streamlit web app
src/config.py         paths, model names, nutrient IDs
src/explore_usda.py   prints the raw USDA table structure
src/allergens.py      deterministic allergen + diet tagging
src/clean_data.py     Phase 1: raw USDA -> foods_clean.csv
src/calculator.py     Phase 2: calories and macros
src/parser.py         Phase 3: free-text preferences and allergies
src/embeddings.py     Phase 4: food embeddings
src/planner.py        Phase 5: weekly meal planner
src/db.py             Phase 6: SQLite storage
src/feedback.py       Phase 7: weekly learning loop
src/simulate.py       Phase 7: simulated users for evaluation
src/shopping.py       weekly shopping list (raw / dry amounts to buy)
src/report.py         printable PDF of the week (fpdf2)
src/nutrients.py      fiber, sodium, calcium... reference values and plan totals
src/progress.py       weight tracking: expected vs actual weekly change
src/dishes.py         20 Egyptian dishes as recipes of USDA ingredients
src/heldout.py        held-out parser evaluation sheet (real people's sentences)
src/evaluate.py       Phase 8: all evaluation metrics
tests/                pytest suite
data/processed/       foods_clean.csv, embeddings, cleaning report
data/evaluation/      evaluation results
```

## Data and models (all free)

- USDA FoodData Central, SR Legacy (April 2018), public domain.
- [Dizex/FoodBaseBERT-NER](https://huggingface.co/Dizex/FoodBaseBERT-NER) for food entity recognition.
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for embeddings.
