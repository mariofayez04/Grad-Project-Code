"""Phase 0 sanity check: libraries, dataset location, and both models.

Run from the project root:  .venv\\Scripts\\python.exe -m tests.check_setup
"""
import importlib

from src.config import USDA_DIR, NER_MODEL, EMBED_MODEL

for lib in ["pandas", "numpy", "sklearn", "torch", "transformers", "sentence_transformers", "streamlit"]:
    mod = importlib.import_module(lib)
    print(f"[ok] {lib} {getattr(mod, '__version__', '')}")

needed = ["food.csv", "food_nutrient.csv", "nutrient.csv", "food_category.csv", "food_portion.csv", "measure_unit.csv"]
missing = [f for f in needed if not (USDA_DIR / f).exists()]
print("[ok] USDA dataset found" if not missing else f"[!!] missing USDA files: {missing}")

from transformers import pipeline
ner = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
print("[ok] NER:", [(e["word"], e["entity_group"]) for e in ner("I hate mushrooms and I'm allergic to shrimp")])

from sentence_transformers import SentenceTransformer
emb = SentenceTransformer(EMBED_MODEL).encode(["grilled chicken breast"])
print("[ok] embeddings shape:", emb.shape)
