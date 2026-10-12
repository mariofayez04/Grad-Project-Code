"""Phase 4: food embeddings with sentence-transformers/all-MiniLM-L6-v2.

Every food description becomes a 384-number vector that captures "what the food is like", so
"grilled chicken breast" ends up close to "roasted turkey breast" and far from "chocolate bar".
Vectors are computed once, L2-normalised (so cosine similarity = dot product) and saved to disk.

Build:  python -m src.embeddings
"""
import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import EMBED_MODEL, EMBEDDING_IDS, EMBEDDINGS, FOODS_CLEAN

# USDA wording that says nothing about what the food is like.
NOISE = [
    r"\([^)]*\)", r"broilers? or fryers", r"separable lean only", r'trimmed to [\d/]+" fat', r"all grades",
    r"\bchoice\b", r"\bselect\b", r"\bprime\b", r"without salt( added)?", r"no salt added", r"mature seeds",
    r"with added [^,]*", r"\b(un)?enriched\b", r"commercially prepared", r"drained solids", r"solids and liquids?",
    r"\bnfs\b", r"mixed species", r"all classes", r"meat only", r"\bboneless\b", r"\bskinless\b", r"\bdrained\b",
    r"\bfluid\b", r"\bregular\b", r"\bimported\b", r"\bdomestic\b", r"\bfarmed\b", r"\bwild\b", r"composite of [^,]*",
    r"\bready-to-(serve|eat|heat)\b", r"\bprepared\b", r"from recipe", r"\bplain\b", r"\bfresh\b",
]
NOISE_RE = re.compile("|".join(NOISE), re.I)


def embed_text(description, category=None):
    """Clean a USDA description into the text we embed, e.g.
    'Chicken, broilers or fryers, breast, meat only, cooked, roasted' -> 'Chicken, breast, cooked, roasted (Poultry)'."""
    text = NOISE_RE.sub("", description)
    parts = [p.strip(" -") for p in text.split(",")]
    text = ", ".join(p for p in parts if p)
    if category:
        text += f" ({category.replace(' Products', '').replace(' and Vegetable', '')})"
    return text


@lru_cache(maxsize=1)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def encode(texts, batch_size=128, show_progress=False):
    """Encode a list of strings into normalised float32 vectors."""
    vecs = load_model().encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=show_progress, convert_to_numpy=True)
    return vecs.astype(np.float32)


def build(foods=None):
    foods = foods if foods is not None else pd.read_csv(FOODS_CLEAN)
    texts = [embed_text(d, c) for d, c in zip(foods["description"], foods["food_category"])]
    vecs = encode(texts, show_progress=True)
    EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS, vecs)
    np.save(EMBEDDING_IDS, foods["fdc_id"].to_numpy())
    return vecs


class FoodEmbeddings:
    """Saved food vectors plus similarity helpers used by the planner (Phase 5) and feedback loop (Phase 7)."""

    def __init__(self, foods=None):
        foods = foods if foods is not None else pd.read_csv(FOODS_CLEAN)
        if not EMBEDDINGS.exists():
            raise FileNotFoundError(f"{EMBEDDINGS} not found - run: python -m src.embeddings")
        self.vectors = np.load(EMBEDDINGS)
        self.ids = np.load(EMBEDDING_IDS)
        if not np.array_equal(self.ids, foods["fdc_id"].to_numpy()):
            raise ValueError("Embeddings are out of date with foods_clean.csv - run: python -m src.embeddings")
        self.foods = foods.reset_index(drop=True)
        self.index = {fid: i for i, fid in enumerate(self.ids)}

    def vector(self, fdc_id):
        return self.vectors[self.index[fdc_id]]

    def mean_vector(self, fdc_ids):
        """Average of several foods, re-normalised (e.g. the 'taste' of all accepted meals)."""
        v = self.vectors[[self.index[i] for i in fdc_ids]].mean(axis=0)
        return v / (np.linalg.norm(v) or 1.0)

    def similarity(self, query_vec, fdc_ids=None):
        """Cosine similarity of a vector to every food (or to the given foods)."""
        mat = self.vectors if fdc_ids is None else self.vectors[[self.index[i] for i in fdc_ids]]
        return mat @ query_vec

    def similar_to(self, fdc_id, k=5, mask=None):
        """Top-k most similar foods to one food. `mask` (bool array) limits the candidates."""
        sims = self.similarity(self.vector(fdc_id))
        sims[self.index[fdc_id]] = -np.inf
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        top = np.argsort(-sims)[:k]
        return self.foods.iloc[top].assign(similarity=sims[top])

    def search(self, text, k=5):
        """Foods closest to free text, e.g. a preference like 'grilled chicken'."""
        sims = self.similarity(encode([text])[0])
        top = np.argsort(-sims)[:k]
        return self.foods.iloc[top].assign(similarity=sims[top])


def neighbor_agreement(vectors, labels, k=5):
    """Share of each food's k nearest neighbours with the same label (meal_role or category).
    Used in the report to compare embedding-text variants (raw USDA text: 0.90, cleaned + category: 0.97)."""
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :k]
    labels = np.asarray(labels)
    return float(np.mean(labels[nn] == labels[:, None]))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    vecs = build()
    print(f"Saved {vecs.shape[0]} vectors of size {vecs.shape[1]} -> {EMBEDDINGS}")
    emb = FoodEmbeddings()
    queries = sys.argv[1:] or ["grilled chicken", "salmon", "brown rice", "broccoli", "greek yogurt", "lentils"]
    for q in queries:
        res = emb.search(q, k=5)
        print(f"\n'{q}':")
        for d, s in zip(res["description"], res["similarity"]):
            print(f"   {s:.2f}  {d[:90]}")
