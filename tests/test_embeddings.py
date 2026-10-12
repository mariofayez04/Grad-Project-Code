"""Phase 4 tests: saved vectors are aligned with foods_clean.csv and behave sensibly.

Requires the vectors to be built first:  python -m src.embeddings
"""
import numpy as np
import pandas as pd
import pytest

from src.config import EMBEDDINGS, FOODS_CLEAN
from src.embeddings import FoodEmbeddings, embed_text, neighbor_agreement

pytestmark = pytest.mark.skipif(not EMBEDDINGS.exists(), reason="run `python -m src.embeddings` first")


@pytest.fixture(scope="module")
def emb():
    return FoodEmbeddings(pd.read_csv(FOODS_CLEAN))


def test_embed_text_removes_noise():
    t = embed_text("Chicken, broilers or fryers, breast, meat only, cooked, roasted", "Poultry Products")
    assert t == "Chicken, breast, cooked, roasted (Poultry)"


def test_shape_and_normalised(emb):
    assert emb.vectors.shape == (len(emb.foods), 384)
    assert np.allclose(np.linalg.norm(emb.vectors, axis=1), 1.0, atol=1e-4)


def test_neighbours_share_role(emb):
    assert neighbor_agreement(emb.vectors, emb.foods["meal_role"]) > 0.9


def _first(emb, pattern):
    return emb.foods.loc[emb.foods["description"].str.contains(pattern, case=False, regex=True), "fdc_id"].iloc[0]


def test_chicken_neighbours_are_poultry(emb):
    res = emb.similar_to(_first(emb, r"^Chicken, broiler.*breast.*grilled"), k=5)
    assert (res["meal_role"] == "protein").all()


def test_search_finds_the_food(emb):
    for query, word in [("broccoli", "Broccoli"), ("greek yogurt", "Yogurt"), ("brown rice", "Rice")]:
        assert word in emb.search(query, k=1)["description"].iloc[0]


def test_masked_substitution(emb):
    """Replacement search limited to allowed foods (as the planner does after filtering allergens)."""
    mask = (emb.foods["food_category"] != "Poultry Products").to_numpy()
    res = emb.similar_to(_first(emb, r"^Chicken, broiler.*breast.*grilled"), k=5, mask=mask)
    assert (res["food_category"] != "Poultry Products").all()


def test_mean_vector_normalised(emb):
    ids = emb.foods["fdc_id"].iloc[:10].tolist()
    assert np.linalg.norm(emb.mean_vector(ids)) == pytest.approx(1.0, abs=1e-5)
