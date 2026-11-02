# AI Personalized Nutrition Recommendation System: Full Project Summary

Everything decided and discussed in this chat, in one place. Sections 1–6 are decisions and reasoning, 7–11 are the build reference, 12–15 are extras (other models, gotchas, report tips, scope control).

> **Current status (2026-10-28):** all 8 phases and 10 extra features are built and tested locally (235 tests pass); evaluation regenerated (section 17.7). Section 16 is the checklist, section 17 the full progress log: what was built, the results, what changed from this plan, and what is left (online deploy + report).

---

## 1. Your project at a glance

- **Title:** AI Personalized Nutrition Recommendation System
- **Context:** graduation project, Faculty of Computing, proposal dated 26/3/2026, supervisor Dr Eid Emary
- **Planned approach (from your proposal):** Python + scikit-learn; compute calories/macros from user inputs; content-based recommender (KNN) over a food nutrition dataset; filter by allergies/preferences; weekly meal plans; web app
- **Planned deliverables:** trained recommendation model, processed nutrition dataset, web app, weekly diet plan generator, report, source code, demo
- **Hard constraint:** everything must be free (no paid APIs, no paid hosting)

## 2. Decisions you made

| Item | Choice |
|---|---|
| Food dataset | USDA FoodData Central (SR Legacy recommended, see section 7) |
| Preference-parsing model | `Dizex/FoodBaseBERT-NER` |
| Embedding model (recommended) | `sentence-transformers/all-MiniLM-L6-v2` |
| Web framework | Streamlit (or Gradio, see gotchas in section 13) |
| Storage for checkbox state | SQLite |
| Hosting | Hugging Face Spaces or Streamlit Community Cloud (free) |

## 3. Input / output spec

**Inputs (menu):**
1. Name
2. Age
3. **Biological sex (add this, see below)**
4. Weight (kg)
5. Height (cm)
6. Activity level (1–3)
7. Goal: maintain weight / lose weight / gain muscle
8. Free-text box for preferences & allergies, e.g. "I hate mushrooms and I'm allergic to shrimp"

**Outputs:**
1. Daily calorie target (from activity level + goal)
2. Macro split in grams (protein / carbs / fat), depending on goal
3. 7-day meal plan hitting those calorie and macro targets
4. A checkbox under every meal (eaten / not eaten)
5. Next week's plan improved based on unchecked meals and preferences

**Gap fixed:** the Mifflin-St Jeor BMR formula needs biological sex. It's a required term, so the form needs that field.

## 4. Feedback on FoodBaseBERT-NER

- It's a good free, small choice. It finds *which words are foods* in a sentence.
- It does **not** tell you how the user feels about each food. "I hate mushrooms and I'm allergic to shrimp" gives you two FOOD entities and nothing else. You need a small rule layer on top (section 8).
- `Dizex/InstaFoodRoBERTa-NER` is a sibling model trained on informal social-media text. Not clearly better for a form field, so keep FoodBaseBERT-NER.
- **Safety point for your report:** for allergies, do not rely on the ML model alone. A missed detection on a real allergy is a safety problem. Always cross-check against a deterministic allergen dictionary (section 7) as the primary safety net. Use the NER model for open-ended dislikes and likes.

## 5. Can this be done without AI?

Mostly yes, and say so plainly in your report (committees ask this).

| Part | Needs AI? |
|---|---|
| Calorie target, macro split | No. Plain formulas |
| Picking foods near the macro targets | No. Filter + sort by distance. "KNN" here is just nearest-neighbor search with no training |
| Weekly plan, checkboxes, saving state | No. Normal software engineering |
| Reading free-text preferences | Benefits from NLP. Regex alone breaks on phrases like "I'm not really into shellfish these days" |
| Smart substitutions for rejected meals | Embeddings add real value (below) |
| Learning taste over weeks | A lightweight recommender (below) |

## 6. The three places AI genuinely adds value

### 6.1 Free-text preference parsing
**Simply:** the user types a normal sentence. A small model picks out the foods, then simple rules decide what to do with each one.
- "allergic to / allergy" → **block** completely
- "hate / dislike / don't like" → **avoid** (soft)
- "love / favorite / prefer" → **prefer** (boost)

**Upgrades:** handle multiple sentences and typos; add a category dictionary so "dairy", "nuts", "seafood", "gluten" expand into specific foods (section 8).

### 6.2 Semantic substitution when a meal is rejected
**Simply:** a naive system swaps a skipped meal for anything with the same calories (you might get a chocolate bar replacing grilled chicken). Embeddings turn each food into numbers that capture *what it's like*, so "grilled chicken with rice" ends up near "grilled fish with rice".
- Embed every food once with `all-MiniLM-L6-v2` and save the vectors.
- Replacement must be close in **both** macros **and** embedding space to foods the user accepted before.

**Upgrade:** compare against the *average* of several accepted meals, not one.

### 6.3 Learning preferences over time (stretch feature)
**Simply:** keep one running "taste vector" per user. Nudge it toward accepted meals and away from skipped ones each week, then rank next week's candidates by closeness to it. It's like a taste profile that updates itself.
- This is arithmetic on vectors from a pre-trained model. **No training or GPU needed.**
- It fits your recommender-systems coursework and gives you something to evaluate (acceptance rate over weeks).

**Upgrade:** weight recent weeks more heavily so the profile follows changing tastes.

---

## 7. Data: USDA FoodData Central

**Download page:** https://fdc.nal.usda.gov/download-datasets

**Which file:** **SR Legacy (CSV).** It's the cleaned set of generic/whole foods (roughly 7,800 items, 2018 release), much smaller and cleaner than Branded, FNDDS, or Full downloads. Branded data is huge and full of messy product names. (Confirm the exact file name and size on the page.)

**Important: the raw download is several linked tables, not one file.** Expect files such as `food.csv`, `food_nutrient.csv`, `nutrient.csv`, `food_category.csv`, `food_portion.csv`. Calories and macros are stored as rows in `food_nutrient.csv` that you pivot into columns. The nutrient IDs you'll want (verify against `nutrient.csv`):

| Nutrient | ID |
|---|---|
| Energy (kcal) | 1008 |
| Protein | 1003 |
| Total fat | 1004 |
| Carbohydrate (by difference) | 1005 |

Values are per 100 g. Portion sizes live in `food_portion.csv`.

**Final columns to keep:**

| Column | Purpose |
|---|---|
| `fdc_id` | unique ID |
| `description` | food name |
| `food_category` | grouping |
| `calories` | macro calc |
| `protein_g` | macro calc |
| `carbs_g` | macro calc |
| `fat_g` | macro calc |
| `serving_size` / `serving_unit` | realistic portions |

**Columns to add:**

| Column | Purpose |
|---|---|
| `allergens` | comma-separated allergen tags found by keyword-matching `description` |
| `is_allergen_free` | `True` only when `allergens` is empty |

Drop rows missing calories or any macro. Save as `foods.csv`.

**Starter allergen dictionary (the 9 groups):**

| Group | Trigger words |
|---|---|
| milk | milk, cheese, butter, cream, yogurt, yoghurt, whey, casein, lactose, ghee, curd, custard |
| egg | egg, eggs, mayonnaise, meringue, albumen |
| peanut | peanut, peanuts, groundnut |
| tree_nut | almond, walnut, cashew, pistachio, pecan, hazelnut, macadamia, pine nut, chestnut |
| shellfish | shrimp, prawn, crab, lobster, crayfish, clam, mussel, oyster, scallop, squid, octopus |
| soy | soy, soybean, tofu, tempeh, miso, edamame |
| wheat | wheat, flour, bread, pasta, couscous, semolina, bulgur, cracker, noodle |
| fish | fish, salmon, tuna, cod, tilapia, sardine, anchovy, mackerel, trout, herring |
| sesame | sesame, tahini |

**Matching tips:**
- Use **word-boundary** regex so "eggplant" isn't tagged as egg.
- Over-flagging is the safe direction for allergies (e.g. "peanut butter" may also get a milk tag). Accept that, or refine later.

---

## 8. Calculator and parser reference

### 8.1 Calories
- **BMR (Mifflin-St Jeor):** `10 × weight_kg + 6.25 × height_cm − 5 × age + 5` (male) or `− 161` (female)
- **Activity multiplier (suggested mapping, adjust if you cite a different one):** level 1 → 1.2, level 2 → 1.55, level 3 → 1.725
- **TDEE** = BMR × multiplier
- **Goal adjustment (starting values):** lose ≈ −15 to −20%, maintain = 0, gain muscle ≈ +10 to +15%
- **Safety floor:** don't output below roughly 1,200 kcal (women) / 1,500 kcal (men). Add a short "not medical advice" note in the app.

### 8.2 Macro split (starting percentages of calories; cite a source in your report)
| Goal | Protein | Carbs | Fat |
|---|---|---|---|
| Maintain | 25% | 45% | 30% |
| Lose weight | 35% | 35% | 30% |
| Gain muscle | 30% | 45% | 25% |

Convert to grams: protein and carbs = 4 kcal/g, fat = 9 kcal/g.

### 8.3 Preference parser rules
1. Run `Dizex/FoodBaseBERT-NER` (e.g. `pipeline("ner", model="Dizex/FoodBaseBERT-NER", aggregation_strategy="simple")`) to get food entities.
2. Look at the words just before each entity and classify it. **Check negative phrases first** ("don't like" before "like"):
   - **block:** allergic to, allergy, can't eat, cannot eat, intolerant
   - **avoid:** hate, dislike, don't like, do not like, not into, avoid
   - **prefer:** love, like, favorite, prefer, enjoy
3. Expand category words:
   - dairy → milk group
   - nuts → peanut + tree_nut
   - seafood → fish + shellfish
   - gluten → wheat
4. Cross-check every **block** against the `allergens` column and the dictionary. Never rely on the model alone.

---

## 9. Implementation plan (8 phases)

**Phase 1: Data**
1. Download SR Legacy CSV.
2. Join and pivot the tables into one row per food with the columns in section 7.
3. Drop rows missing calories/macros.
4. Build the allergen dictionary and tag each row.
5. Save `foods.csv`.

**Phase 2: Calorie and macro calculator**
1. `calculate_bmr(sex, age, weight, height)`
2. `apply_activity(bmr, level)`
3. `adjust_for_goal(tdee, goal)`
4. `macro_split(calories, goal)` → grams

**Phase 3: Preference and allergy reader**
1. Load the NER model.
2. Extract food entities from the free text.
3. Classify block / avoid / prefer with the rules above.
4. Cross-check blocks against the allergen column.

**Phase 4: Food embeddings (one-time)**
1. Load `all-MiniLM-L6-v2` (384-dimension vectors).
2. Encode every `description`.
3. Save vectors with `fdc_id` (e.g. `.npy`) so you never recompute.

**Phase 5: Meal-picking engine**
1. Remove blocked/allergen foods.
2. Score the rest by closeness to per-meal macro targets.
3. Add preference boosts/penalties.
4. Build 7 days × meal slots with a variety rule (no constant repeats).

**Phase 6: Web app**
1. Input form with all fields.
2. Calorie/macro summary.
3. Weekly plan table with a checkbox per meal.
4. Save checkbox state (SQLite).

**Phase 7: Weekly learning loop**
1. At week's end, read checked vs skipped meals.
2. Update the user's taste vector.
3. Re-rank next week's candidates partly by closeness to it.

**Phase 8: Deploy and evaluate**
1. Deploy free (Hugging Face Spaces or Streamlit Community Cloud).
2. Measure the metrics in section 14.

### 9.1 Suggested formulas for Phases 5 and 7 (starting points)
- **Candidate score:** `score = w1 × macro_fit + w2 × cosine(food_vector, taste_vector) + w3 × preference_adjustment`
- **Taste vector update:** `taste_new = normalize(taste_old + α × mean(accepted_vectors) − β × mean(skipped_vectors))`
- Start with something like α = 0.3, β = 0.15 and tune. Use cosine similarity for vectors.

---

## 10. Prompt to give Gemini (Google Colab)

Paste this as one message. After each phase Gemini will stop and wait; reply "continue".

```
You are helping me build my university graduation project: an AI-powered
personalized nutrition recommendation system. Everything must run on free
tools only (no paid APIs) inside Google Colab.

PROJECT SPEC
Input: name, age, biological sex, weight (kg), height (cm), activity level
(1-3), goal (maintain / lose weight / gain muscle), and a free-text field
for preferences/allergies (e.g. "I hate mushrooms and I'm allergic to
shrimp").

Output: daily calorie target, macro split in grams, a 7-day meal plan built
from real food data, a checkbox per meal (eaten / not eaten), and a weekly
feedback loop that improves suggestions based on which meals were skipped.

DATA SOURCE
Use the USDA FoodData Central SR Legacy CSV download
(https://fdc.nal.usda.gov/download-datasets). The download is several
linked tables (for example food.csv, food_nutrient.csv, nutrient.csv,
food_category.csv, food_portion.csv). FIRST print the file names and column
headers so we both see the real structure. Then join/pivot them into one
row per food. Nutrient IDs I expect (verify in nutrient.csv): 1008 energy
(kcal), 1003 protein, 1004 total fat, 1005 carbohydrate. Values are per
100 g.

Final columns: fdc_id, description, food_category, calories, protein_g,
carbs_g, fat_g, serving_size, serving_unit. Drop rows missing calories or
any macro. Add two columns:
- allergens: comma-separated tags from milk, egg, peanut, tree_nut,
  shellfish, soy, wheat, fish, sesame, found by WORD-BOUNDARY keyword
  matching on description (so "eggplant" is not tagged egg). Build a
  trigger-word dictionary per group. Over-flagging is acceptable.
- is_allergen_free: True only if allergens is empty.
Save as foods.csv.

WORKING STYLE
Build one phase at a time, in this order:
1. Data download + cleaning (above)
2. Calorie/macro calculator: Mifflin-St Jeor (needs sex), activity
   multiplier (levels 1/2/3 -> 1.2/1.55/1.725), goal adjustment (lose
   about -15 to -20%, gain about +10 to +15%), calorie safety floor, and
   macro split by goal (maintain 25/45/30, lose 35/35/30, gain 30/45/25
   as protein/carb/fat percent)
3. Free-text preference/allergy parser using Hugging Face model
   Dizex/FoodBaseBERT-NER to extract foods, plus keyword rules (check
   negative phrases first) to classify each as block / avoid / prefer,
   category expansion (dairy, nuts, seafood, gluten), and a cross-check of
   every block against the allergens column
4. Food embeddings with sentence-transformers/all-MiniLM-L6-v2 over
   description, computed once and saved
5. Meal-selection engine: filter blocked foods, score by macro distance
   plus preference adjustment, build a 7-day plan. Note: SR Legacy items
   are individual foods, so each "meal" must be a combination of 2-3 foods
   with gram portions scaled to hit the meal's calories/macros
6. Web interface (Gradio in Colab, or Streamlit for deployment) with the
   input form, summary, weekly table, and a checkbox per meal
7. Weekly taste-vector feedback loop (explain it to me before coding it)
8. Evaluation: macro-adherence error, allergen-filter accuracy

After each phase: show the code, explain it in plain language, and STOP.
Wait for me to say "continue". Do not skip ahead or combine phases.
```

---

## 11. Free tools checklist

| Need | Tool |
|---|---|
| Data | USDA FoodData Central |
| NER model | `Dizex/FoodBaseBERT-NER` (Hugging Face, run locally) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (run locally) |
| Code environment | Google Colab |
| Web app | Streamlit or Gradio |
| Database | SQLite |
| Hosting | Hugging Face Spaces / Streamlit Community Cloud |

---

## 12. Other Hugging Face models and datasets found

**Datasets**
- `datahiveai/recipes-with-nutrition`: recipes with macro/micronutrient profiles and diet labels. Good if you want real *meals* instead of single foods. https://huggingface.co/datasets/datahiveai/recipes-with-nutrition
- `adarshzolekar/foods-nutrition-dataset`: small structured set for prototyping. https://huggingface.co/datasets/adarshzolekar/foods-nutrition-dataset
- `ai5labsOfficial/open-food-calories`: ~5k rows, MIT license. https://huggingface.co/datasets/ai5labsOfficial/open-food-calories
- `mmathys/food-nutrients`: ~3k photographed plates (Nutrition5k) with macros; only useful for a photo feature. https://huggingface.co/datasets/mmathys/food-nutrients

**Models**
- `sentence-transformers/all-MiniLM-L6-v2`: small, fast text embeddings (your semantic-matching model). https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- `Dizex/FoodBaseBERT-NER`: your chosen food-entity extractor. https://huggingface.co/Dizex/FoodBaseBERT-NER
- `Dizex/InstaFoodRoBERTa-NER`: informal-text sibling. https://huggingface.co/Dizex/InstaFoodRoBERTa-NER
- `sgarbi/bert-fda-nutrition-ner`: finds nutrition components in label-style text (niche). https://huggingface.co/sgarbi/bert-fda-nutrition-ner
- `meta-llama/Llama-3.1-8B-Instruct` (gated): a paper on LLM meal planning (arXiv 2502.20601) reported Llama 3.1 8B landing close to calorie targets when grounded in USDA data. Best used only to *explain* the plan in friendly language, not to choose foods (LLMs are unreliable at exact calorie/macro arithmetic). https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- `sanil08/food-llm`: small on-device food-log-to-nutrition pipeline; a reference design more than a drop-in. https://huggingface.co/sanil08/food-llm
- Photo food logging (optional bonus): `nateraw/food`, `ashaduzzaman/vit-finetuned-food101`, `eslamxm/vit-base-food101` (ViT models fine-tuned on Food-101). Food-101 skews toward Western dishes, so expect weaker results on Egyptian/Middle Eastern food unless you add data.

---

## 13. Practical gotchas and open design decisions

1. **SR Legacy is ingredient-level, not meal-level.** "Chicken breast, cooked" is a food, not a meal. Your engine must build each meal from 2–3 foods (protein + carb + vegetable/fat) with gram amounts scaled to hit the meal's targets. Decide early; the alternative is a recipes dataset like `datahiveai/recipes-with-nutrition`.
2. **Streamlit doesn't run natively inside Colab.** Develop Phases 1–5 in Colab. For the interface, either use Gradio (works in Colab with a share link, and also deploys to Hugging Face Spaces) or put the Streamlit app in a repo and deploy it.
3. **Free Hugging Face Spaces storage is temporary.** A SQLite file can reset when the Space restarts. Fine for a demo; for the defense, demo locally or keep state in session/JSON export.
4. **Allergen tagging will over-flag.** That's the safe direction. Mention it as a deliberate design choice.
5. **Filter odd USDA categories** (baby foods, fast-food items, game meats, etc.) so the plans look like normal meals.
6. **Calorie safety floor and a "not medical advice" note** are worth including in the app.

---

## 14. Evaluation and report talking points

**Metrics**
- Macro-adherence error: how close daily totals land to the targets
- Allergen filter precision/recall on a small hand-labeled test set of sentences and foods
- Parser accuracy: correct block/avoid/prefer labels on ~30–50 test sentences
- Acceptance rate of suggestions across a few simulated weeks (if Phase 7 is built)
- Optional ablation: free-text parsing vs a plain dropdown form

**Points to make in the defense**
- Be upfront about what's classical math (BMR, macros, filtering, nearest-neighbor) vs what's genuinely AI (NER parsing, embedding-based substitution, taste-vector learning).
- Allergies use a deterministic safety net alongside the model, by design.
- Embeddings and the taste vector need **no training of your own**: pre-trained model plus vector arithmetic. That's legitimate, lightweight, and free.

---

## 15. If time gets tight

Phases 1–6 give you a complete working project. **Phase 7 (taste vector) is the safest thing to cut or simplify.** A simpler fallback: when a meal is skipped, just penalize that food and pick the embedding-nearest replacement that fits the macros.

## 16. Next steps checklist

The project was built locally in VS Code / Python (not Colab), so the Gemini prompt in section 10 was not used.

- [x] Download SR Legacy CSV from the USDA page (kept outside the project in `..\Dataset\`)
- [x] Phase 1: clean food table working (`data/processed/foods_clean.csv`, 3,184 foods)
- [x] Phase 2: calculator tested (unit tests against hand-calculated Mifflin-St Jeor values)
- [x] Phase 3: parser tested on 32 hand-labelled sentences
- [x] Phase 4: embeddings saved
- [x] Phase 5: a full 7-day plan generating correctly
- [x] Phase 6: interface with checkboxes
- [x] Phase 7: taste-vector loop
- [x] Phase 8: evaluation done, Docker image ready
- [x] Extra: "Swap a food" feature, new dark theme (section 17.3)
- [x] Extra: private users with name + PIN sign-in (section 17.7)
- [x] Extras: all 10 features of the roadmap in section 17.7
- [ ] Phase 8: put the site online on Hugging Face Spaces (needs `hf auth login` with your token first)
- [ ] Phase 8: write the report
- [ ] Collect held-out sentences from 5-10 people + a one-week pilot (tools ready, see 17.7 #2)

---

## 17. Progress log

### 17.1 Timeline

| Date | Step |
|---|---|
| 2026-10-12 | Phase 0: virtual environment, packages, both models downloaded |
| 2026-10-12 | Phase 1: USDA data cleaning + allergen tagging |
| 2026-10-12 | Phase 2: calorie and macro calculator |
| 2026-10-12 | Phase 3: preference and allergy parser |
| 2026-10-12 | Phase 4: food embeddings |
| 2026-10-12 | Phase 5: weekly meal planner |
| 2026-10-12 | Phase 6: Streamlit web app + SQLite |
| 2026-10-13 | Phase 7: weekly feedback loop + simulation |
| 2026-10-13 | Phase 8: evaluation, Dockerfile, README, Docker Compose |
| 2026-10-27 | Small fixes, "Swap a food" feature, design changes (section 17.3) |
| 2026-10-27 | Roadmap #1 private users (PIN), #3 shopping list, #5 New meal / New day with before-after choice (section 17.7) |
| 2026-10-28 | #8 PDF, #11 meals per day, bug fix (PDF downloading at sign-in), #7 extra nutrients, #10 diet modes + nutrient caps, #6 weight tracking, #9 Egyptian dishes, #2 evaluation + real-user tools, dishes in feedback learning |
| 2026-10-28 | All of the above committed, merged into `main` and pushed to GitHub (section 17.8) |
| 2026-10-28 | UI redesign (light calm theme, top-menu pages) and a Dark mode switch, merged into `main` and pushed (section 17.9) |

### 17.2 What each phase built

**Phase 1: Data** (`src/clean_data.py`, `src/allergens.py`)
- Joins the raw USDA tables into one row per food: 4 nutrients per 100 g plus one realistic household portion.
- 7,793 foods → **3,184** kept. Removed: baby/fast/restaurant foods, spices, drinks, brand names, organs, game meat, raw versions of foods eaten cooked, flours, sauces, dry mixes, and foods whose calories don't match their macros.
- Allergen tags for the 9 groups + gluten, word-boundary matching, with exceptions (eggplant, peanut butter, coconut milk, water chestnut, buckwheat, corn tortillas...). Over-flagging on purpose.
- Diet flags: vegetarian, pork, alcohol. A `meal_role` per food (protein, carb, vegetable, fruit, fat, dairy, legume, snack, mixed).
- The old `foods.csv` from an earlier attempt (unreliable allergen tags) was deleted.

**Phase 2: Calculator** (`src/calculator.py`)
- Mifflin-St Jeor BMR → activity ×1.2 / 1.55 / 1.725 → goal −20% / 0 / +10% → floor 1,500 kcal (men) / 1,200 kcal (women).
- Macro split from section 8.2. Protein capped at 2.2 g per kg body weight (spare calories go to carbs).
- BMI warnings (underweight + lose, obese + gain). Daily split: breakfast 25%, lunch 35%, dinner 30%, snack 10%.

**Phase 3: Parser** (`src/parser.py`)
- FoodBaseBERT-NER **plus** a food-word dictionary find the foods; cue phrases (longest first) label each one block / avoid / prefer, including cues after the food ("shrimp allergy", "gluten-free").
- Category words expand (dairy, nuts, seafood, gluten...). Vegetarian, vegan, halal and pescatarian are rules, not the model.
- Safety net: any allergen word after an allergy cue is blocked even if no food was found.

**Phase 4: Embeddings** (`src/embeddings.py`)
- Descriptions are cleaned (USDA filler removed) and the category added, then encoded with all-MiniLM-L6-v2 → 384 numbers per food, saved in `embeddings.npy`.

**Phase 5: Planner** (`src/planner.py`)
- Each meal is built from components: lunch / dinner = protein + carb + vegetable + optional cooking oil; breakfast = main + carb + fruit; snack = fruit + extra.
- Candidates are scored: how well they do their job + closeness to the taste vector + stated likes / dislikes − repeat penalties + a little randomness.
- The top 5 per component are combined and the grams solved with bounded least squares. Each meal aims at what is still left of the day.

**Phase 6: Web app** (`app.py`, `src/db.py`)
- Streamlit: profile form, daily targets, "What I understood", 7 day tabs with an "Eaten" checkbox per meal, week overview, CSV download. The user is kept in the URL (`?user=ID`).
- SQLite (`data/app.db`): users, plans, plan_items, swaps.

**Phase 7: Feedback loop** (`src/feedback.py`, `src/simulate.py`)
- Taste vector: `normalize(old + 0.3 × mean(eaten) − 0.15 × mean(skipped))`.
- Food penalties: skipped +0.8, eaten −0.2, older weeks count 0.7× less each week.
- Family penalty: a food family (e.g. all beef) skipped more than the user's average is pushed down as a whole.
- Unticked meal = skipped. Cooking oil is ignored (not a taste signal). Swapped-away foods count as skipped (17.3).

**Phase 8: Evaluation and deploy** (`src/evaluate.py`, `Dockerfile`, `compose.yaml`, `README.md`)
- All numbers in `data/evaluation/results.md` (summary in 17.4).
- Docker image for Hugging Face Spaces (CPU-only PyTorch, models downloaded at build time). `docker compose up -d --build` runs it locally with `data/` kept on the PC.

### 17.3 Changes on 2026-10-27

Small fixes
- `scipy==1.18.1` added to `requirements.txt` (the planner imports it directly).
- Old `foods.csv` deleted; planner description corrected (meals are 2–4 foods); harmless warning in `simulate.py` removed.
- Docker image: Streamlit file watcher turned off (it filled the logs with a harmless `torchvision` error).

"Swap a food" (new feature, under every meal)
1. Pick a food of the meal → 3 suggestions that fill the same part of the meal, are close in **embedding space and macro profile**, avoid your allergies, and are sized to keep the same calories.
2. Suggestions are varied: at most one per base food, no repeated names, nothing already on that day's plan (fixed a "Pasta, cooked" shown twice bug).
3. **Show more options**: the next 3, up to about 10 pages, then back to the start.
4. **Type what you'd like** (e.g. "brown rice"): embedding search among foods of the same kind, with a bonus when every typed word is in the name. If nothing is close (score < 0.6) the app says so.
5. **Learning**: every swap is saved (`swaps` table). When the week is finished, the swapped-away food counts as skipped in the taste vector, food penalties and family penalties, and is listed under "Swapped away" in "What I learned".
- New code: `component_foods`, `component_of`, `swap_options`, `search_swap`, `servable_foods` (planner); `replace_item`, `user_swaps` (db); `swaps` argument in `food_penalties`, `next_week_inputs`, `week_summary` (feedback).

Design (replaced by the redesign of 2026-10-28, section 17.9)
- Dark theme (`.streamlit/config.toml` + CSS at the top of `app.py`): **black** background, **dark-gray** meal cards, totals boxes and profile sidebar, **red** buttons, red accents (card top strip, calorie labels, grams).
- All meal cards have the same height; "Swap a food" and "Eaten" sit on one row at the bottom.
- Each food shows as: **bold title + grams** (e.g. "Turkey 100 g"), then the details in medium text, then `(kcal : 139), (P : 30), (F : 2), (C : 0)`.

Tests: 128 → **135** (swap variety, typed search, swap learning, swap storage).

### 17.4 Results (from `data/evaluation/results.md`)

| What | Result |
|---|---|
| Daily calories vs target | 1.2% mean error, 100% of days within ±10% |
| Protein / carbs / fat | 3.7% / 2.6% / 3.6% mean error |
| Allergens served to 20 users with allergies | **0** of 1,803 items |
| Allergen tagger (32 hand-labelled foods) | 100% precision, 100% recall |
| Parser, NER + dictionary (32 sentences) | 100% labels correct (NER alone 95.9%, dictionary alone 95.9%) |
| Embeddings, 5 nearest neighbours with same category | 98.2% (raw USDA text: 89.4%) |
| Feedback loop, meals eaten (simulated hidden dislikes) | ~80% in week 1 → 98–100% from week 2 (no feedback: stays ~80–85%) |
| Time to build a week's plan | ~2.2 s |

Limitations to state in the report: the tagger and parser test cases were used while writing the rules (scores are optimistic); allergens are found from food names only; the feedback test uses simulated users. The swap feature is not in the evaluation yet.

### 17.5 How to run

```bash
.venv\Scripts\python.exe -m streamlit run app.py      # the web app at http://localhost:8501
.venv\Scripts\python.exe -m pytest tests -q            # 235 tests (~5 min)
.venv\Scripts\python.exe -m src.evaluate               # regenerates data/evaluation/results.md (~3-5 min)
docker compose up -d --build                           # same app in Docker
```

If a test fails with `httpx.ConnectTimeout` (the model loader checking Hugging Face), run the tests offline:
`set HF_HUB_OFFLINE=1` and `set TRANSFORMERS_OFFLINE=1` (PowerShell: `$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1`).
In Claude's preview, `.claude/launch.json` has a second entry, "nutriplan-8502", on port 8502 with a scratch database,
so testing never touches `data/app.db`.

### 17.6 What is left

1. **Put the site online**: run `.venv\Scripts\hf.exe auth login` (paste your Hugging Face Write token), choose private or public, then upload (README, "Deploy for free"). The free Space resets its database when it restarts.
2. **Write the report**: use sections 5, 6, 14 and 17.4; be clear about what is classical math vs AI.
3. Optional: a held-out test set written by other people; add the swap feature to the evaluation.

### 17.7 Feature roadmap (started 2026-10-27)

Built one at a time, each reviewed before the next:

| # | Feature | Status |
|---|---|---|
| 1 | Private users: name + PIN sign-in | ✅ done |
| 3 | Shopping list for the week | ✅ done |
| 5 | Regenerate one meal or one day, compare before / after, learn from the choice | ✅ done |
| 8 | Printable PDF of the week's plan | ✅ done |
| 11 | Choose 3, 4 or 5 meals a day | ✅ done |
| 7 | More nutrients (fiber, sugar, sodium, calcium, iron) | ✅ done |
| 10 | Diet modes for health conditions (diabetes-friendly, low-salt) | ✅ done |
| 6 | Weight tracking + targets that follow your weight | ✅ done |
| 9 | Egyptian dishes (ful, koshari, molokhia, ta'ameya...) | ✅ done |
| 2 | Real-user evaluation tools (held-out sentences, friends' feedback) + updated evaluation | ✅ done |

**#1 Private users (done)**
- The sidebar no longer lists everyone's name. Signed out: **Sign in** (name + PIN) or **New user** (profile + PIN, typed twice).
- PINs are 4–8 digits, stored only as a salted PBKDF2-SHA256 hash (200,000 rounds). After 5 wrong PINs the profile is locked for 5 minutes. A wrong name and a wrong PIN give the same message, so names can't be probed.
- A new user can't take an existing name any more (before, "New user" with the same name overwrote that person's profile).
- Reloading keeps you signed in through a signed link (`?user=ID&t=...`, HMAC with a server secret in `data/secret.key`, which git and Docker ignore). Typing `?user=1` alone opens nothing; changing the PIN ends old links. **Sign out** button.
- Profiles made before PINs (e.g. Marco, Ramez Alex) are asked to create a PIN the first time they sign in. Until then anyone who knows the name could claim them, so each owner should sign in once soon.
- The database is upgraded automatically (new columns added on start). 7 new tests (142 total).

**#3 Shopping list (done)** (`src/shopping.py`, section "🛒 Shopping list" in the app)
- Adds up every food of the week and groups it by shop section (meat & poultry, fish, eggs & dairy, bread, grains, beans, vegetables, fruits, nuts, oils), one card per section in the dark style, with a tick box per item.
- Converts plate weights into what to buy (approximate): cooked rice ×0.35 → dry, pasta/noodles ×0.45, bulgur/couscous/quinoa/barley ×0.4, oats cooked with water ×0.17, cooked beans ×0.4, fresh pasta ×0.8, cooked meat/fish ×1.35 → raw. Canned, cured and smoked foods are bought as they are.
- Merges variants into one line: all eggs → "Eggs ≈ N eggs", all oils → "Cooking oil ≈ N ml, any kind", "Milk, reduced fat" with and without added vitamins → one line; plate words (cooked, boiled, raw...) are dropped from names.
- "Leave out meals already eaten" switch for shopping mid-week. Downloads: text list with [ ] boxes, and CSV.
- 15 new tests (157 total).

**#5 Regenerate a meal or a day, with a before / after choice (done)**
- Every meal card has **🎲 New meal**; every day has **🎲 New version of this day**. A window shows **Current** and **New** side by side: the meals, the day's totals with each version and the difference from the targets, and a "🎯 closer to your targets" badge. Buttons: **Keep current** / **Use the new one** (closing the window decides nothing).
- The new version is really different: the current foods are left out (for a whole day also other versions of them, e.g. another cheese), and foods already on that day are not repeated. The rest of the week counts for variety. A new meal aims at the day's target minus the other meals, so the day stays on target (tested within 10%). Allergies and diet rules still apply.
- **Learning (weekly):** every choice is saved (`choices` table). When the week is finished, the foods of the version you did not keep count as half a skip and the ones you kept as half an eaten meal, in the taste vector, food penalties and family penalties (`CHOICE_WEIGHT = 0.5`; a choice is a softer signal than skipping a meal). Cooking oil is ignored. "What I learned" lists them under "Swapped away or not chosen".
- **Same week:** a food you swapped away or did not choose is not offered right back in the next new versions (penalty 1.5, and 0.5 for other versions of the same food). The full weekly plan only changes when the week is finished.
- Planner refactor: the one-day loop is shared by `plan_week`, `replan_meal` and `replan_day` (existing plans unchanged). Card buttons now sit on two rows ("🔄 Swap" + "🎲 New meal", then "Eaten") so they fit narrow windows. `use_container_width` (being removed from Streamlit) replaced by `width="stretch"`.
- 9 new tests (166 total), including the full app flow: New meal → Use the new one, New day → Keep current.

**#8 Printable PDF (done)** (`src/report.py`, red "🖨️ Download plan (PDF)" button under "Week at a glance")
- Page 1: name, week, daily targets (four boxes), BMR / TDEE / BMI, warnings, what the parser understood, how to use the sheet.
- One block per day, two days per A4 page: every meal with its foods (bold name + details), grams, kcal / protein / carbs / fat, an empty `[   ]` box per meal to tick with a pen (meals already ticked in the app print as `[ x ]`), and the day's totals vs the targets.
- Last pages: the shopping list (same as the app) with a `[   ]` box per item.
- Printer-friendly: white paper, red headers, light-grey stripes; footer "for education only, not medical advice" + page numbers. About 7 pages, ~30 KB.
- New dependency `fpdf2==2.8.9` (small, pure Python, no system packages, works in Docker). Built-in fonts cover Latin letters only: emoji and symbols are swapped or dropped, and a name in another script (e.g. Arabic) prints as "you". The PDF is cached in the app, so ticking boxes stays fast.
- `split_name` (bold title + details) moved from app.py to src/planner.py so the app and the PDF share it. Shopping names now merge regardless of capital letters ("Cheese, Swiss" = "Cheese, swiss").
- 4 new tests (170 total), checking the text of the PDF: every day, the allergy, the shopping list, pre-ticked meals, non-Latin names.

**#11 Meals per day (done)**
- New "Meals per day" choice in the profile: **3** (breakfast 30%, lunch 40%, dinner 30%), **4** (unchanged: 25 / 35 / 30 / snack 10%) or **5** (breakfast 25%, morning snack 10%, lunch 30%, afternoon snack 10%, dinner 25%).
- Meals are shown and planned in time order; with 3 or 5 meals dinner is planned last and corrects the day's small errors (sample week: calories 0.2–0.8% off, protein 0.3–0.4% off).
- The two snacks use the same rules as before and never repeat a food on the same day. The week view, "Week at a glance", shopping list, swap, New meal / New day, the comparison window and the PDF all follow the plan's own meals ("Morning snack", "Afternoon snack").
- Saved in the profile (`users.meals_per_day`) and in each plan's targets. Existing profiles and plans were upgraded automatically and keep 4 meals.
- 11 new tests (181 total), including a 5-meal plan built through the app (35 "Eaten" boxes).

**Bug fix: the PDF downloaded by itself at sign-in (fixed)**
- Cause: every download button sent its file (PDF, 2 CSVs, shopping text) to the browser on each page load, without a click. Download managers (e.g. Internet Download Manager) and some browser settings catch any `.pdf` the page fetches and save it.
- Fix: the buttons now get a function instead of the file, so the file is made only when the button is clicked (`on_click="ignore"`: no page reload either). Checked in the browser: after signing in with Enter, no file is fetched. Sign-in is also faster (the PDF isn't built in advance).

**#7 More nutrients (done)** (`src/nutrients.py`, `clean_data.fill_micros`)
- The food table now has 7 more nutrients per 100 g from USDA: fiber, total sugars, saturated fat, sodium, potassium, calcium, iron (IDs in `config.MICRO_IDS`). Rows and old columns unchanged, so the embeddings stay valid.
- Missing USDA values are estimated and marked in a new `estimated` column: 0 when the parent macro is ~0 (no carbs -> no sugar / fiber; no fat -> no saturated fat), otherwise the median of the same kind of food (same category + first word, then category), capped at the parent macro. Estimated: sugar 24.1% of foods, fiber 6.8%, saturated fat 2.9%, potassium 0.9%, calcium / iron 0.2%, sodium 1 food (cleaning report).
- Daily reference values per person (saved with each plan): fiber >= 14 g / 1,000 kcal and saturated fat < 10% of kcal (DGA 2020-2025), sodium <= 2,300 mg, potassium >= 3,400 / 2,600 mg, calcium >= 1,000 mg (1,200 women 51+, men 71+), iron >= 8 mg (18 mg women 19-50) (US National Academies). Sugars are shown without a limit: USDA gives total sugars (incl. fruit and milk), the limits are for added sugars.
- App: the targets line under "Your daily targets"; a row of nutrient pills under each day's totals (✓ met, ≈ nearly at 90%+, ↓ low, ↑ too high); new section "🧪 Nutrients this week" (daily average, target, days on target). PDF: targets on page 1 and a nutrient line under each day.
- Finding for the report: without being told, the planner already meets fiber, saturated fat, sodium and potassium on most days, but calcium is often short (sample woman 776 mg / 1,000 mg, 0 of 7 days) and iron is borderline for women (18 mg needed). Making the planner aim for these belongs to #10.
- 14 new tests (195 total).

**#10 Diet modes + nutrient-aware planning (done)** (`src/nutrients.py`, planner, calculator)
- Profile "Health focus (optional)", both can be ticked:
  - **Diabetes-friendly:** carbs <= 40% of calories (the rest goes to fat), total sugars <= 10% of calories, stronger fiber bonus, sugar penalty; sugary foods left out (dried / sweetened / flavoured foods, syrups, sweet baked goods).
  - **Low-salt / blood pressure (DASH-style):** sodium <= 1,500 mg, stronger potassium and calcium bonus, sodium penalty; foods with > 600 mg sodium per 100 g and cured / pickled / salted foods left out.
  - A clear note: general guidance, not a treatment; follow your doctor (medicines may need adjusting). Shown in the app and the PDF.
- **For everyone, nutrient-aware planning:** every food gets a nutrient score added to its ranking (`W_NUT = 1.0`): a bonus for fiber, calcium, iron, potassium it brings above its "fair share" (daily target / daily calories), a penalty for sodium / saturated fat above it.
- **Pacing:** while a day is planned, a nutrient stops earning a bonus once the day already has enough of it for the meals so far.
- **Daily caps:** iron 45 mg and calcium 2,500 mg (2,000 mg 51+) - official safe upper limits (NASEM) - fiber at 2x its target (comfort), plus every "max" target (sodium, saturated fat, sugars in diabetes mode). A meal combination that would push the day over a cap gets a heavy penalty (`W_CAP = 20`); the planner aims at 95% of the cap (portion rounding) and a meal may use only its share of the cap + 35% (so breakfast can't use the whole day's iron). Found while testing: even before this feature, fortified cereals / breads gave some days 60+ mg of iron.
- Swaps, typed search, New meal / New day all respect the modes. The salt you add when cooking is not counted (said in the app).
- **Tuning (report material)** - 4 people x 2 weeks per setting; days on target out of 7:

| Setting | Macro error | Foods / week | Fiber | Calcium | Iron | Sodium under limit | Sodium avg | Sugar avg |
|---|---|---|---|---|---|---|---|---|
| No modes, before (W_NUT 0) | 2.3% | 87 | 3.5 | 2.1 | 5.4 | 4.4 | 2,144 mg | 55 g |
| No modes, final | 3.2% | 86 | 7.0 | 6.4 | 6.8 | 7.0 | 1,173 mg | 58 g |
| Diabetes, final | 3.5% | 85 | 7.0 | 4.9 | 6.5 | 7.0 | 1,071 mg | 33 g (was 46) |
| Low-salt, final | 3.5% | 83 | 7.0 | 6.2 | 6.6 | 7.0 | 475 mg (was 1,764) | 59 g |
| Both, final | 3.5% | 81 | 7.0 | 4.4 | 6.8 | 7.0 | 393 mg | 33 g |

  W_NUT 2-3 met calcium on slightly more days but cut variety to 60-76 different foods a week, so 1.0 was kept.
- **Caps (336 planned days, 4 people x 4 settings x 3 weeks):** iron, calcium, sodium, saturated fat over their cap on 0 days (highest iron 44 mg); fiber over its comfort cap on 1.8% of days (at most +10%); sugar (diabetes) 0.6% (at most +4%).
- Calcium is still the hardest (diabetes mode leaves out flavoured yogurts): 4.4-6.4 of 7 days.
- Note: `data/evaluation/results.md` was made before #7 / #10; it is regenerated in #2.
- 13 new tests (208 total).

**#6 Weight tracking (done)** (`src/progress.py`, section "⚖️ Weight progress" at the bottom of the app)
- Log a weight with a date (no future dates, 30-300 kg); one entry per day - logging again the same day replaces it. New profiles start with their profile weight; existing profiles (Marco, Mario, Ramez Alex) got their profile weight as a first entry automatically. Changing the weight in the profile form also adds an entry.
- Four numbers: current weight (+ change since the first entry), **expected change per week** on this plan = (calorie target - TDEE) x 7 / 7,700 kcal per kg, **your actual pace** (straight line through the last 4 weeks; needs entries a week apart), BMI.
- Chart (Altair): your weights (red line + points) next to the weight the plan should give (grey dashed line).
- **Targets follow the weight:** the latest entry becomes the profile weight, so next week's plan (Finish week) uses it. If it changes the calorie target by 50+ kcal, the app shows the new target and a "🔄 Rebuild this week with my new targets" button.
- **Safety:** a warning when losing faster than ~1% of body weight a week, or gaining faster than ~0.5% (common clinical guidance).
- New table `weights`; `src/db.py`: `log_weight`, `weight_history`, `delete_weight` (a typo can be fixed by logging the same date again).
- 8 new tests (216 total), incl. the app flow: log 88 kg -> profile updated -> new-target message -> rebuild -> week 2 with the new calorie target.

**#9 Egyptian dishes (done)** (`src/dishes.py`, data step 7 in `clean_data.py`, planner `DISH_TEMPLATES`)
- 20 dishes as recipes of USDA SR Legacy ingredients (one serving "as served", grams per ingredient, no added salt):
  - breakfast: ful medames, ta'ameya, shakshuka, gebna with tomatoes, bessara;
  - lunch / dinner: koshari, molokhia with rice and chicken, mahshi, fattah, bamya with rice, kofta, hawawshi, sayadiya, fasolia with rice, shorbet ads (lentil soup), mesaqa'a, torly, ferakh bel roz;
  - snacks: halabessa, roz bel laban. Desserts (basbousa, konafa, om ali) left out on purpose.
- Each dish's nutrients per 100 g = gram-weighted sum of its ingredients (incl. the 7 extra nutrients); allergens and vegetarian / pork / alcohol flags come from the ingredients with the same deterministic rules (e.g. koshari -> wheat, ta'ameya -> sesame, molokhia -> milk from ghee, halabessa not vegetarian because of chicken broth). Ingredients come from the whole USDA table (cumin, lemon juice and broth are not in foods_clean). Dishes are added to foods_clean.csv (category "Egyptian Dishes", meal_role "dish", ids from 9,000,001) and the recipes to dish_ingredients.csv; embeddings rebuilt (3,204 foods).
- Planner: a meal can also be **one dish + optional sides** (bread / rice, salad, extra protein, fruit; 0 g = not needed). For every meal both kinds are tried and the better fit wins. Breakfast dishes up to 350 g (room for bread), lunch / dinner up to 650 g, snacks up to 300 g.
- Profile setting "🥘 Egyptian dishes": **often** (~12 dish meals a week) / **sometimes** (default, ~7 a week = about one a day) / **never**. Tuned: without a handicap dish meals won 16-20 of 28 meals (the optional sides make them flexible), so the dish meal's value gets -0.7 (sometimes) or -0.3 (often). Macro error unchanged (~3%).
- Everything follows: allergies and diets (tested: no wheat / sesame dishes when allergic, only vegetarian dishes for vegetarians), diet modes, "I love koshari", Swap (a dish swaps for another dish of that meal), typed search, New meal / New day.
- App: dish cards show the Arabic name (kept in place with `<bdi>`) and the ingredients of that portion in gold ("fava beans 240 g, spinach 55 g, ..."); the PDF prints the ingredients too. Shopping list: a dish is split into its ingredients (scaled to the portion), with shop-friendly names ("Fava beans (ful)", "Molokhia leaves", "Tahini") and a new "🧂 Herbs, spices & sauces" section.
- Also fixed on the way: a portion at its maximum wasn't rounded to 5 g (e.g. 493.6875 g); "sweet potato leaves" could be served as a carb.
- Speed: a week takes ~3.2 s (was ~2.1 s) because each meal tries two kinds; dish sides keep only their 2 best candidates (same accuracy, 25% faster than 3).
- Existing profiles got "sometimes": their next plans include about one Egyptian dish a day (set "never" to switch off).
- 11 new tests (227 total).

**#2 Evaluation: regenerated + real-user tools (done)**
- `python -m src.evaluate` (~10-15 min) regenerates `data/evaluation/results.md` with the final planner and 5 new sections: 8 nutrients and safety caps (nutrient-aware vs macros only), 9 diet modes, 10 Egyptian dishes, 11 held-out sentences, 12 real-user pilot. CSVs for each next to it.
- Headline numbers (results.md, 2026-10-28):
  - Macro adherence (30 users x 7 days): calories 2.3% mean error (98.1% of days within ±10%), protein 4.3% (92.9%), carbs 3.0%, fat 4.8%.
  - Allergy safety: 20 users with 1-3 allergies, 1,652 items incl. 145 Egyptian dish servings -> **0** items with a stated allergen; parser found 100% of the allergies.
  - Nutrients, nutrient-aware vs macros only (84 days): days on target - calcium 38% -> 82%, fiber 55% -> 98%, iron 68% -> 98%, potassium 69% -> 98%, sodium 54% -> 100%; days over a cap - sodium 46% -> 0%, iron 1.2% -> 0%, calcium 2.4% -> 0%.
  - Diet modes: low-salt 530 mg sodium / day, diabetes-friendly 32 g sugars (limit met 100% of days) and carbs 40.5% of kcal; 0 left-out foods served.
  - Egyptian dishes: sometimes 6.2 dish meals / week (5.4 different), often 10.5 (8.9 different), never 0.
  - Feedback loop (simulated hidden dislikes): meals eaten with feedback 77-87% in week 1 -> 96-100% from week 2 (after the fix below); without feedback no improvement.
  - Speed: 3.2 s per week (max 6.2 s).
- **Honest trade-offs for the report:**
  - Macro accuracy is a little lower than in the first evaluation (calories 1.2% -> 2.3%, protein days within ±10% 97.6% -> 92.9%): the planner now also chases nutrients and keeps under safety caps (same users: 3.1% macros only vs 4.2% nutrient-aware). Much better nutrition for a small macro cost.
  - (Fixed the same day - see "Dishes count their ingredients in feedback learning" below.) Learning from skipped meals had become a bit slower once dishes were added, because a dish's family was only the dish itself.
- **Real-user tools:**
  - Held-out sentences: `python -m src.heldout --template` makes `data/evaluation/heldout_sentences.xlsx` (Instructions tab + 3 grey examples). Volunteers who didn't build the app write sentences + what the app should understand (never / less often / more often). Scored per label (recall, precision) and allergy recall (safety) - section 11.
  - In the app: "📝 Rate this week" (optional) in "Finish the week": liked the meals, realistic portions, easy to follow (1-5), would keep using it (yes / maybe / no), comment. Table `surveys`, one per week.
  - Pilot summary (section 12) from `data/app.db`, without names: finished weeks, meals eaten per week of use, swaps, New meal / day choices, weight entries, ratings; comments exported without names. How to run the test with people: README "Testing with real people".
- Small fixes: "-0.00 kg / week" on maintain now shows 0.00; "by 1 people" -> "by 1 person".
- 5 new tests (232 total).

**Follow-up: dishes count their ingredients in feedback learning (done)** (`src/feedback.py`)
- `food_families`: a food belongs to its own family (base name, e.g. "beef"); an Egyptian dish also to the families of its main ingredients - at least 10% of the serving, ignoring oils, broth, spices (`MAIN_INGREDIENT_SHARE`, `NEUTRAL_CATEGORIES`). E.g. bamya -> beef, okra, rice; sayadiya -> fish; shakshuka -> egg; koshari -> lentils, pasta, rice.
- Skipping a dish now counts against its ingredients' families, and a family skipped more than the user's average now pushes down the dishes made with it too (a disliked beef no longer comes back hidden in bamya, kofta or hawawshi). The "only more than your usual skip rate" rule still protects families like rice.
- Simulation (section 6 of results.md, regenerated), with feedback, before -> after:

| Hidden dislike | Meals eaten week 2 / 3 / 4 | Disliked foods served week 2 / 3 / 4 |
|---|---|---|
| beef & lamb | 92.9 / 97.6 / 97.6% -> **96.4 / 100 / 98.8%** | 2.0 / 0.7 / 0.7 -> **1.0 / 0.0 / 0.3** |
| eggs & cottage cheese | 92.9 / 95.3 / 94.1% -> **96.4 / 96.4 / 98.8%** | 2.0 / 1.3 / 1.7 -> **1.0 / 1.0 / 0.3** |
| fish & seafood | 98.8 / 100 / 100% -> **100 / 100 / 100%** | 0.3 / 0.0 / 0.0 -> **0.0 / 0.0 / 0.0** |

  ("No feedback" rows unchanged.)
- 3 new tests (235 total).

**Roadmap complete (2026-10-28).** Still open: put the site online (needs `hf auth login`), collect held-out sentences + a one-week pilot with 5-10 people, write the report.

### 17.8 Saved to git and GitHub (2026-10-28)

All the work of 2026-10-27 and 2026-10-28 (sections 17.3 and 17.7) is in commit `5a39318`, made on the branch `feature-roadmap`, then merged into `main` (fast-forward) and pushed to GitHub (`origin` = github.com/mariofayez04/Grad-Project-Code), together with the earlier unpushed Docker Compose commit. The merged branch was deleted.
- Not in git on purpose (`.gitignore`): `data/app.db` (people's profiles and plans), `data/secret.key` (sign-in link secret), `data/plan_demo.pdf` (demo output), `.venv/`.
- Before committing, a test was found writing its fake survey comment into `data/evaluation/pilot_comments.csv`; the test now writes to a temporary folder and the file was removed.
- 235 tests pass at this commit.
- Left out of the commits on purpose: an uncommitted change to `.streamlit/config.toml` (a light "sage green" theme) made after the commit, outside this work. It was kept as it is in the working folder; commit it when it's finished. (Superseded: the redesign in 17.9 replaced the theme file and is committed.)

### 17.9 UI redesign and dark mode (2026-10-28)

Goal: a more premium, calm look for a health app (motivational, not noisy, comfortable), with no feature removed, no
"AI-looking" fonts, styles or copy, and no long dashes.

**1. Competitor research (what people say)**

| App | What users like | What users complain about | What we did |
|---|---|---|---|
| MyFitnessPal | Big food database | The April 2026 "Today" redesign made logging slower, micronutrients buried in menus and endless scrolling; heavy ads; complex interface | Every feature is one click from the top menu; nutrients are in plain view (Plan page pills + Nutrients page); no ads or upsell banners |
| Lifesum | Described as "beautiful, clean, calming" | Food ratings feel judgmental; ads | Kept the calm look; replaced red warnings with neutral words: "On target", "Close", "A little under or over" |
| Cronometer (web) | Accuracy, custom recipes | Web version looks like a stretched phone app, too much empty space | Layouts use the full desktop width (two meal cards side by side, 4 target tiles in a row) |
| Eat This Much | Desktop multi-day view (CNN Underscored top pick) | Free tier limited to one day | Kept "Week at a glance" as a clear table |
| Mealime | Low effort, simple grocery list | Less control | Shopping list with tick boxes, a basket progress bar and sections in columns |

Sources: mwm.ai (MyFitnessPal v26.16.0 redesign), unstar.app (MyFitnessPal ads and paywall), nutrola.app and
kimola.com (Lifesum reviews), forums.cronometer.com, CNN Underscored (best meal planning apps), plantoeat.com
(Mealime and Eat This Much reviews). Several sources are written by competing apps, so their numbers are treated
loosely.

**2. Design first, then code**
- The design was drawn first as a canvas with 9 screens: welcome and sign in, profile (new user and edit), Plan,
  Nutrients, Shopping, Progress, the Swap popover, the New meal compare dialog, and a phone view.
  Canvas: https://claude.ai/artifact/JDqexXsBckX3sr5yvFcpfh (private until shared from its Share menu).
- Then it was built into `app.py` and `.streamlit/config.toml`.

**3. Look**
- Light: soft sage background `#F4F6F2`, white cards with a thin border and 20 px corners, deep green accent `#1E5B48`,
  ink text `#15201C`. Macro colours: protein blue, carbs orange, fat lavender.
- Fonts (Google Fonts, set in the theme): Hanken Grotesk for text, Schibsted Grotesk for headings. The usual
  AI-template fonts (Inter, Poppins, Space Grotesk) were avoided on purpose.
- Emoji replaced by simple line icons (inline SVG in the page, Material icons on buttons). No gradients or glow.
- Copy is plain and warm, e.g. "27 meals to go this week. Steady beats perfect."; nutrient statuses never say "bad".

**4. New layout (no sidebar)**
- A top bar with the brand, the page menu (**Plan, Nutrients, Shopping, Progress, Profile**), the **Dark mode** switch
  and the user's name (menu: Signed in as..., Edit profile, Sign out).
- The open page is kept in the address (`?p=shopping`), so a reload stays on the same page and keeps the user signed in
  (`?user=ID&t=...` as before).
- **Welcome (signed out):** headline, short intro and the 4 "How it works" steps on the left; a card on the right with a
  Sign in / New user switch. New user shows the full profile form in the card. The PIN flows are unchanged (wrong PIN,
  locked for 5 minutes, profiles from before PINs asked to create one).
- **Plan:** greeting with the date and week, a ring with meals eaten this week and a "Finish the week" shortcut; daily
  targets as 4 tiles (with % of calories), BMR / TDEE / BMI, the diet mode badge and the "Also every day" nutrient line;
  "What I understood" as chips (Never included / Less often / More often) with "Edit preferences"; day tabs drawn as
  pills; per day: totals vs targets in words ("30 under", "14 g over"), nutrient pills with a coloured dot and a legend,
  "New version of this day"; meal cards with an icon, macros, kcal and a protein/carbs/fat bar, the foods (Arabic name and
  "In this portion: ..." for Egyptian dishes), **Swap a food**, **New meal** and an **Eaten** pill that turns green;
  "Week at a glance" as a table with **Download plan (PDF)** and **Download CSV**.
- **Nutrients:** summary tiles (on target / close / a little under or over, with the nutrient names), a table with
  status chips, daily average with a bar towards the target, target per day ("at least 29 g", "under 1,500 mg") and days
  on target, then the notes about the numbers.
- **Shopping:** text and CSV downloads at the top, a "N of M in the basket" progress bar, the "Leave out meals already
  eaten" switch, section cards in 3 columns; ticked items are struck through.
- **Progress:** Finish the week (progress bar, explanation, "Rate this week" survey, the Finish button), Weight progress
  (4 numbers with help, the chart with a legend, Log weight, the "Rebuild this week with my new targets" offer) and
  What I learned (3 lists + meals eaten per week chart). The intro sums up the weight change ("1.3 kg down since ...").
- **Profile:** the same fields as before, grouped in cards (Account, About you, Goal and activity, Your meals,
  Preferences and allergies). Biological sex, goal, activity, meals per day and Egyptian dishes are tappable choice cards
  with a short note each (e.g. "Lose weight: 20% under your daily need"); Health focus is two checkboxes. A side card
  shows this week's target and the "Generate my weekly plan" button.
- **Dialogs:** the Swap popover and the "Which one do you prefer?" comparison keep their behaviour; the version closer to
  the targets gets a green border and a "Closer to your targets" badge.
- Phone width: the menu wraps, cards stack, wide tables scroll inside their box; no sideways page scroll.

Every feature from before is still there; only where it lives changed (e.g. weight and Finish week are on Progress, the
profile form is on Profile).

**5. Dark mode switch**
- `.streamlit/config.toml` now has `[theme.light]` and `[theme.dark]` colours (shared fonts and corners in `[theme]`).
  Dark: background `#0E1412`, cards `#161E1A`, text `#E4EBE6`, buttons `#2E7D5E`, lighter green accent `#7CC6A4`.
- All colours in `app.py` are CSS variables; when the dark theme is in use (`st.context.theme.type`), a dark set of the
  same variables is added, so cards, chips, tables, icons and charts follow.
- The **Dark mode** switch (top bar and welcome screen) saves the choice where Streamlit's own Settings > Theme menu keeps
  it (browser storage, key `stActiveTheme-<page path>-v2`) and reloads the page; the URL keeps the user signed in and on
  the same page. The switch also shows the right position when the theme was changed in Settings.
- Limitation: the switch relies on that Streamlit storage key, which is fine for the pinned `streamlit==1.65.0`; a
  future Streamlit upgrade may change it (Settings > Theme would still work).

**6. Tests and checks**
- 8 app test files updated for the new layout: tests open other pages with `at.session_state["goto"] = "<Page>"`; labels
  changed ("Eaten ✅" -> "Eaten", "➕ Log weight" -> "Log weight"); meals per day and Egyptian dishes are now radios,
  Health focus a checkbox; the Egyptian dish check looks for "In this portion".
- Full suite: 235 passed after the redesign; the 96 app-level tests passed again after dark mode.
- Checked in the browser on a scratch database: sign up, sign out / in, reload keeps the user, every page, Eaten tick,
  Swap suggestions, New meal comparison, phone width, both themes; no console errors. The `torchvision` messages in the
  local server log come from Streamlit's file watcher scanning the transformers package (already off in Docker).

**7. Git**
- Commit `c3330fc` "Redesign the app UI: light calm theme and top-menu pages" and `516f23d` "Add a Dark mode switch with a
  matching dark palette", made on the branch `ui-redesign`, merged into `main` (fast-forward) and pushed to GitHub.
- Files: `app.py`, `.streamlit/config.toml`, `.claude/launch.json`, 8 test files. The old dark/red theme is still in
  git history (commit `46e78dd`) if it is ever needed.

### 17.10 Hosting on Vercel: test deploy prepared (2026-11-01)

Decision: try Vercel first with a test deploy, before moving the database to Postgres. Railway (about $12-15 a month) stays the fallback if Streamlit doesn't work well there.

Why a test first: the app needs a running server and a live connection. Vercel now runs Docker images (Container Runtime) and WebSockets, but both are in beta. Vercel's limits that matter:
- Instances scale down after 5 minutes without traffic (30 seconds on preview deployments), so test on the production deployment.
- Data on the instance is temporary and not shared between instances, so `data/app.db` can't be kept there. Real use needs a hosted database (for example Neon Postgres) and the sign-in secret as a setting.
- Live connections close at the function's maximum duration: 5 minutes on the free Hobby plan, up to 30 minutes on Pro. The reconnect can reach another instance. The signed sign-in link should survive this, but open forms could reset. Only a real deploy can show how bad this is.

Changes for the test (not yet on GitHub):
- `Dockerfile`: listens on `$PORT` (default 8501 for Docker Compose and other hosts) and never contacts Hugging Face at runtime (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`), since the models are in the image.
- `vercel.json`: Container preset, Frankfurt region (`fra1`, the closest common region to Egypt).

Tested locally with the Vercel free plan's limits (Docker, 2 GB memory, 1 CPU, throwaway database):
- Image builds: 3.52 GB, within Vercel's limits (15 GB, 2 million files; the Python packages are about 44,000 files).
- Server answers within 1 second; models ready in about 20 seconds on 1 CPU. So a visitor after an idle period waits about 20 seconds here; Vercel may differ.
- Sign-up and a full 7-day plan work. Peak memory 0.46 GB of the 2 GB limit, but that plan had no preferences text, so the embedding model wasn't loaded. With preferences text the peak was 0.96 GB on Windows, not measured in the container.
- Not tested in the container: reopening the sign-in link, PDF download.

Settings the Vercel project needs: `PORT=8501` and, for the test only, `NUTRI_DB_PATH=/tmp/nutriplan/app.db`. Test data is thrown away whenever the app restarts: no real profiles on the test site.
(Replaced by `DATABASE_URL`, see 17.11.)

Still to check on the real deploy: does Streamlit's live connection work, what happens at the 5-minute disconnect, how long the wait after idle really is, do downloads (PDF) work. If it passes, next step: move the database to Postgres.

### 17.11 Permanent accounts on the hosted site, totals table, equal choice cards (2026-11-01)

The test deploy went live at `grad-project-code.vercel.app`. Feedback from using it:

**Bug: signed out while scrolling, then "Wrong name or PIN." with the right PIN.**
Cause: on Vercel the database file and the sign-in secret were on the server's temporary disk. When Vercel started a new server, the account was gone. On the user's PC no account had a failed PIN attempt, which fits: the failure was on the hosted site only.

Fix, in `src/db.py`:
- With the `DATABASE_URL` setting the app uses Postgres (`psycopg`, with a small connection pool). Without it, SQLite as before, so running on the PC doesn't change.
- The queries are shared. Changes for Postgres: `RETURNING id` instead of `lastrowid`, column names instead of positions, `lower(name)` so names still ignore case, and one `_read_sql` helper instead of `pd.read_sql_query`. The schema is translated by `_postgres_schema()` (`BIGSERIAL`, `BIGINT`, `DOUBLE PRECISION`, `BYTEA`).
- The sign-in link secret comes from the database URL when hosted, so every server shares it and a restart doesn't sign anyone out. No extra setting is needed. Changing the database password signs everyone out once.
- `tests/conftest.py`: `TEST_DATABASE_URL` runs the whole test suite on Postgres.

Tested on a throwaway Postgres in Docker: sign-up with preferences text, 7-day plan, Eaten tick, New meal, New version of this day; then the app and the database restarted. The same PIN still signs in (also with the name in lower case), the old sign-in link still works, and the tick is still there.

Still needs one step by the user: create a free Postgres database and add `DATABASE_URL` in Vercel. Until then the hosted site keeps losing accounts. The hosted site itself is not retested yet.

What this can't change: Vercel's free plan closes the live connection after 5 minutes and stops idle servers. After the fix that means a short reconnect and staying signed in, but up to about 20 seconds if a new server has to start, and a half-filled form or an open comparison is lost.

**Choice buttons (profile and sign-up form).** Cards had different sizes, and the text was squeezed by a large indent ("Exercis-e"). Now every card in a group is the same size, with the description under the title:
- Three equal columns; one column when the form is narrow. This uses a container query, because the sign-up card can be narrow on a wide screen.
- Cards with a description are one height (100 px, 72 px in one column); cards without one (Biological sex, Egyptian dishes) share a shorter height.
- The Health focus boxes are the same width, and action buttons are all 44 px high. The top menu stays at 40 px.

**Day totals.** Moved from above the meals to below them, as a table: Nutrient, This day, Target, Result. Calories and macros come first, then the other nutrients. The dot is green within 5% of the target (the same rule as the comparison window) and yellow otherwise; the other nutrients keep their existing status. "New version of this day" moved with it.

### 17.12 Simulated testers and the allergy safety fix (2026-11-02)

Hosting: the Neon free Postgres database `neon-aureolin-bridge` is connected to the Vercel project `grad-project-code` through the Vercel integration, so `DATABASE_URL` is set. After the redeploy the user signed up, signed out and signed in again on the live site: it works.

**Simulated testers (`src/testers.py`, results in `data/evaluation/simulated_testers.md`).** 24 scripted users x 4 weeks x learning on and off, using the app's own database, planner and feedback functions on a throwaway database. These are not real people, and they produce no satisfaction ratings. The report must say so.

**Bug found by testing on new wording.** The parser's 100% on its own test sentences did not hold on sentences it was not built with:

| Sentences (written by an AI assistant, not volunteers) | Allergy items blocked, before | After |
|---|---|---|
| Set 1, 61 sentences, used to build the fix | 18 / 27 | 27 / 27 |
| Set 2, 40 sentences, failures seen while fixing | 7 / 30 | 30 / 30 |
| Set 3, 40 sentences, written after the rules were frozen, scored once | not measured | **22 / 32 (68.8%)** |

Set 3 is the fair number; sets 1 and 2 are optimistic. Of set 3's 10 misses, the app now asks about 7 and misses 3 with no warning (9.4% of items). The 4 testers with messy wording were served 185 allergen foods before the fix and 0 after, but they were chosen because their wording was known to fail, so that shows the effect of the fix, not a rate.

**Fix (`src/parser.py`):** spelling correction (uses the NER model's vocabulary and the food names, so real words stay as they are); reaction wording, conditions (celiac, favism) and doctor's advice count as a block; `unclear` makes the app ask when it can't tell what was meant (in the app and the PDF). The rule of asking was added after seeing set 3, and doesn't change its score. No phrase list can cover every wording, so asking is part of the safety design.

Known limits: likes and dislikes in indirect wording are often missed ("Salmon is amazing"); "I'd prefer to skip the mushrooms" still reads as a like.

Tests: 238 -> 267. Report files are in `report/` (`build_report.py`, `content.py`, `make_diagrams.py`, `make_charts.py`, `make_code_figures.py`).

### 17.13 Consent and deleting an account (2026-11-02)

Why: Egypt's Law No. 151 of 2020 treats health data as sensitive, and summaries of it describe explicit consent as the general rule. This comes from law-firm summaries, not a legal review.

- **Sign-up:** a required box. It says what is stored, that data may be kept outside Egypt, and that the account can be deleted. Nothing is saved until it's ticked.
- **Accounts from before:** asked at their next sign-in. They can agree, sign out or delete; nothing else opens until they answer. (`users.consented_at`)
- **Delete my account:** at the bottom of the Profile page. It asks for the PIN and a confirmation, then removes the profile, plans, foods, swaps, choices, weights and ratings (`db.delete_user`, `ON DELETE CASCADE`). Wrong PINs count towards the lock.
- **Hosted database upgrade:** `init_db` didn't add new columns on Postgres, so this column could have crashed the live site. It now runs `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. Tested on a throwaway Postgres built from the schema of the deployed version: the column is added, the user is kept, and starting twice is fine.
- Tests: 267 -> 270. A database provider may keep backups for a while after a delete; this was not checked for Neon.

### 17.14 Allergy checklist (2026-11-02)

Why: free text blocked only 68.8% of allergy items on sentences kept apart (17.12), and no list of phrases can cover every wording.

- **Profile:** ten tick boxes (milk, egg, peanut, tree nuts, fish, shellfish, soy, wheat, gluten, sesame) or "I have no food allergies". One answer is required. Saved in `users.allergies`.
- **Rule:** a ticked group is always blocked, with no language model involved (`parse_preferences(text, allergies=...)`). Plans, swaps, typed swap searches and new meals all respect it. Free text stays for likes, dislikes and other allergies.
- **Older accounts:** answer once at the next sign-in. A ticked allergy rebuilds their plan; "no allergies" keeps it.
- **Measured** (`python -m src.testers --checklist`, `data/evaluation/checklist_results.md`): 15 people who only ticked boxes, 1 week each: 0 of 1,239 items contained a ticked allergen, 1.9% calorie error. With the matching box ticked, set 3 goes from 22 / 32 to 32 / 32, which is true by design and assumes people tick correctly. The kiwi allergy in set 3 is outside the list and still depends on the text.
- Report screenshots retaken with a neutral test account.
- Tests: 270 -> 281.

### 17.15 Realistic portions and the pilot kit (2026-11-02)

Why: plans hit their numbers with meals nobody would eat (a lunch with 305 g of tofu and 355 g of French beans; 410 g of lentils).

- **Limits (`src/planner.py`):** lower maximum portions per part of a meal (protein 220 g, carbohydrate 280 g, beans and lentils 220 g, vegetables 200 g, dishes 450 g). Bigger eaters get at most +20% (was +60%), then denser foods win. A meal over `MEAL_GRAMS` loses value (`W_BULK`).
- **Swaps:** a food that can't reach the calories it replaces within 12% now comes after the foods that can.
- **Measured (`python -m src.portions`), 30 users x 7 days, before -> after:**

| Measure | Before | After |
|---|---|---|
| Foods over a generous serving | 18.6% | 12.9% (2.5% for targets under 2,800 kcal) |
| Largest single food | 560 g | 335 g |
| Main meals over 800 g | 21.7% | 11.9% |
| Heaviest main meal | 1,480 g | 1,160 g |
| Calorie error | 2.3% | 3.2% |
| Days within 10% of calories | 98.1% | 95.7% |
| Protein error | 4.3% | 5.6% |

  The serving limits are a judgment, not official values. Whether foods go together isn't measured yet, and people with targets of 2,800 kcal or more still get many large portions (40.1%).
- All evaluation results were regenerated, and the report's numbers updated.
- **Pilot kit:** `data/evaluation/pilot_message.md` (message for friends, English and Arabic) and `meal_rating_sheet.xlsx` (20 real planned meals for a qualified person to rate, `python -m src.portions --sheet`). Both are unused so far: the report must not claim any real-user or expert result until they come back.
- The fasting, Arabic input, budget, "why this meal", leftovers, family and photo features were suggested and declined by the user, so none was built.
- Tests: 281 -> 284.
