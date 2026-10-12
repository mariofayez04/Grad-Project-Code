"""Central paths and settings used by every module."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original USDA SR Legacy download (kept outside the project, never modified)
USDA_DIR = PROJECT_ROOT.parent / "Dataset" / "FoodData_Central_sr_legacy_food_csv_2018-04" / "FoodData_Central_sr_legacy_food_csv_2018-04"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FOODS_CLEAN = PROCESSED_DIR / "foods_clean.csv"
EMBEDDINGS = PROCESSED_DIR / "embeddings.npy"
EMBEDDING_IDS = PROCESSED_DIR / "embedding_ids.npy"  # fdc_id of each row in EMBEDDINGS
DB_PATH = DATA_DIR / "app.db"

# Hugging Face models
NER_MODEL = "Dizex/FoodBaseBERT-NER"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# USDA nutrient IDs (per 100 g)
NUTRIENT_IDS = {"calories": 1008, "protein_g": 1003, "fat_g": 1004, "carbs_g": 1005}
