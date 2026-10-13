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
| Name, age, biological sex, weight, height, activity level (1-3), goal | Daily calorie target and macro split (g) |
| Free-text preferences and allergies | 7-day plan: breakfast, lunch, dinner, snack with gram portions |
| "Eaten" checkbox under every meal | Next week's plan adapted to what was eaten and skipped |

## How it works

```
profile ──► calculator.py ──► daily kcal + macros (Mifflin-St Jeor, activity, goal, safety floor, protein cap)
free text ─► parser.py ─────► block / avoid / prefer   (FoodBaseBERT-NER + rules + allergen dictionary)
USDA SR Legacy ─► clean_data.py ─► foods_clean.csv (3,184 foods, allergen + diet tags, meal roles)
foods ──► embeddings.py ────► 384-d vectors (all-MiniLM-L6-v2), computed once
            planner.py ─────► 7-day plan: score candidates, combine 2-4 foods, solve grams (bounded least squares)
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
python -m pytest tests -q     # 128 tests
python -m src.evaluate        # writes data/evaluation/results.md
python -m src.simulate        # feedback-loop experiment only
```

See [data/evaluation/results.md](data/evaluation/results.md) for the numbers used in the report.

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
   hf upload <your-username>/<space-name> . --repo-type space --exclude ".venv/*" --exclude ".git/*" --exclude "data/app.db" --exclude "foods.csv" --exclude "notebooks/*" --exclude "*__pycache__*"
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
src/evaluate.py       Phase 8: all evaluation metrics
tests/                pytest suite
data/processed/       foods_clean.csv, embeddings, cleaning report
data/evaluation/      evaluation results
```

## Data and models (all free)

- USDA FoodData Central, SR Legacy (April 2018), public domain.
- [Dizex/FoodBaseBERT-NER](https://huggingface.co/Dizex/FoodBaseBERT-NER) for food entity recognition.
- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for embeddings.
