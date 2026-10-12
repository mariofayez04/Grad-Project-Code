# AI Personalized Nutrition Recommendation System: Full Project Summary

Everything decided and discussed in this chat, in one place. Sections 1–6 are decisions and reasoning, 7–11 are the build reference, 12–15 are extras (other models, gotchas, report tips, scope control).

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

- [ ] Download SR Legacy CSV from the USDA page
- [ ] Paste the Gemini prompt (section 10) into a new Colab session
- [ ] Phase 1: get `foods.csv` working
- [ ] Phase 2: calculator tested against an online TDEE calculator
- [ ] Phase 3: parser tested on at least 10 sample sentences
- [ ] Phase 4: embeddings saved
- [ ] Phase 5: a full 7-day plan generating correctly
- [ ] Phase 6: interface with checkboxes
- [ ] Phase 7: taste-vector loop (if time allows)
- [ ] Phase 8: deploy, evaluate, write the report
