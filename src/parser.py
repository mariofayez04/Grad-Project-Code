"""Phase 3: read free-text preferences and allergies.

  "I hate mushrooms and I'm allergic to shrimp"
     -> avoid: mushroom      block: shrimp (+ shellfish group)

How it works
 1. Find food mentions: Dizex/FoodBaseBERT-NER  +  a dictionary scan (safety net for anything the model misses).
 2. Split the text into clauses and classify each food by the nearest cue phrase:
       block  (hard filter):  allergic to, intolerant, can't eat, don't eat, no ..., ... free
       avoid  (soft penalty): hate, dislike, don't like, not into ...
       prefer (boost):        love, like, enjoy, favorite ...
 3. Expand category words (dairy -> milk, nuts -> peanut + tree_nut, seafood -> fish + shellfish ...).
 4. Diet words (vegetarian, vegan, halal) are handled by rules, never by the model.
 5. Every blocked word is cross-checked with the deterministic allergen dictionary (allergens.py).

Run a demo:  python -m src.parser "I hate mushrooms and I'm allergic to shrimp"
"""
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from src.allergens import ALLERGEN_TRIGGERS, GLUTEN_EXTRA, tag_allergens
from src.config import FOODS_CLEAN, NER_MODEL

# --- cue phrases ------------------------------------------------------------------------------------
# Matched longest-first, so "don't like" wins over "like" and "not into" wins over "into".
CUES = {
    "block": [
        "allergic to", "allergic", "allergy to", "allergies to", "allergy", "allergies", "intolerant to", "intolerant",
        "intolerance", "can't eat", "cant eat", "cannot eat", "can not eat", "don't eat", "dont eat", "do not eat",
        "never eat", "not allowed", "can't have", "cannot have", "can't tolerate", "no", "without", "free of",
        "avoid completely", "must avoid", "react to", "reaction to",
    ],
    "avoid": [
        "hate", "hates", "dislike", "dislikes", "don't like", "dont like", "do not like", "don't love", "don't enjoy",
        "do not enjoy", "not a fan of", "not into", "not really into", "not big on", "not fond of", "can't stand",
        "cant stand", "avoid", "rather not", "prefer not", "would rather not", "don't want", "do not want", "less",
        "fewer", "sick of", "tired of", "disgusting", "gross", "not keen on",
    ],
    "prefer": [
        "love", "loves", "like", "likes", "really like", "enjoy", "enjoys", "favorite", "favourite", "favorites",
        "prefer", "prefers", "fan of", "big fan of", "into", "want", "more", "crave", "adore", "fond of", "keen on",
    ],
}
# Cues that come right AFTER the food ("shrimp allergy", "gluten-free", "chicken is my favorite").
POST_CUES = {
    "block": ["allergy", "allergies", "intolerance", "free", "-free"],
    "prefer": ["is my favorite", "is my favourite", "are my favorite", "are my favourite"],
    "avoid": ["is disgusting", "are disgusting", "is gross", "are gross"],
}

# Category words -> allergen groups.
CATEGORY_GROUPS = {
    "dairy": {"milk"}, "lactose": {"milk"}, "milk": {"milk"}, "nuts": {"peanut", "tree_nut"},
    "nut": {"peanut", "tree_nut"}, "tree nuts": {"tree_nut"}, "tree nut": {"tree_nut"}, "seafood": {"fish", "shellfish"},
    "shellfish": {"shellfish"}, "fish": {"fish"}, "gluten": {"gluten"}, "wheat": {"wheat", "gluten"},
    "eggs": {"egg"}, "egg": {"egg"}, "soy": {"soy"}, "sesame": {"sesame"}, "peanuts": {"peanut"},
    "peanut": {"peanut"}, "crustaceans": {"shellfish"}, "shrimp": {"shellfish"},
}
# Words -> diet flags understood by the planner.
FLAG_WORDS = {"pork": {"pork"}, "bacon": {"pork"}, "ham": {"pork"}, "meat": {"meat"}, "alcohol": {"alcohol"}}
DIETS = {
    r"\bvegetarian\b|\bveggie\b": {"flags": {"meat"}},
    r"\bvegan\b|\bplant[- ]based\b": {"flags": {"meat"}, "groups": {"milk", "egg"}, "terms": {"honey"}},
    r"\bhalal\b|\bmuslim\b": {"flags": {"pork", "alcohol"}},
    r"\bpescatarian\b|\bpescetarian\b": {"flags": {"meat_not_fish"}},
}
# Words the NER model sometimes tags as food but aren't.
NOT_FOOD = {"intolerant", "vegetarian", "vegan", "veg", "halal", "allergic", "allergy", "allergies", "food", "foods",
            "meal", "meals", "breakfast", "lunch", "dinner", "snack", "snacks", "diet", "free", "please", "intolerance"}

CLAUSE_SPLIT_RE = re.compile(r"[.!?;\n]+|\bbut\b|\bhowever\b|\bwhereas\b|\balthough\b|\bthough\b", re.I)
GENERIC_BASE_NAMES = {"snacks", "snack", "candies", "soup", "beverages", "cereals", "cereals ready-to-eat", "sauce",
                      "spices", "fast foods", "restaurant", "babyfood", "frozen novelties", "toppings", "fat",
                      "oil", "salad dressing", "breakfast bar", "crackers", "nuts", "seeds", "mollusks", "crustaceans"}


def _cue_regex(phrases):
    alts = "|".join(re.escape(p).replace(r"\ ", r"\s+").replace("'", "['’]?") for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"(?<![\w-])(?:{alts})(?![\w])", re.I)


_PRE_CUE_RE = {label: _cue_regex(p) for label, p in CUES.items()}
_ALL_PRE_RE = _cue_regex([p for ps in CUES.values() for p in ps])
_LABEL_OF = {p.lower().replace("'", ""): label for label, ps in CUES.items() for p in ps}
_POST_CUE_RE = {label: re.compile(r"^\s*(?:" + "|".join(re.escape(p) for p in ps) + r")\b", re.I) for label, ps in POST_CUES.items()}


# --- helpers -----------------------------------------------------------------------------------------
def singular(word):
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("oes", "ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


MODIFIERS = {"severe", "mild", "slight", "bad", "serious", "really", "very", "any", "all", "some", "a", "an", "the",
             "my", "of", "kind", "kinds", "type", "types", "fresh", "much", "lot", "lots"}


def normalize_term(text):
    words = [w for w in re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower())]
    while words and words[0] in MODIFIERS:
        words = words[1:]
    return " ".join(words[:-1] + [singular(words[-1])]) if words else ""


def term_regex(term):
    """Regex matching every word of a term (any order), each allowing a simple plural."""
    parts = [rf"(?=.*\b{re.escape(w)}(?:s|es)?\b)" for w in term.split()]
    if term.endswith("y"):
        parts[-1] = rf"(?=.*\b{re.escape(term.split()[-1][:-1])}(?:y|ies)\b)"
    return re.compile("".join(parts), re.I)


@lru_cache(maxsize=1)
def food_vocabulary():
    """Dictionary of known food words: allergen triggers + first words of the cleaned USDA descriptions."""
    vocab = {w for words in ALLERGEN_TRIGGERS.values() for w in words} | set(GLUTEN_EXTRA) | set(CATEGORY_GROUPS)
    vocab |= set(FLAG_WORDS) | {"honey", "mushroom", "onion", "garlic", "broccoli", "spinach", "eggplant", "chicken",
                                "beef", "lamb", "turkey", "rice", "lentil", "oatmeal", "oats", "potato", "tomato"}
    try:
        base = pd.read_csv(FOODS_CLEAN, usecols=["base_name"])["base_name"].dropna().unique()
        vocab |= {singular(b) for b in base if b not in GENERIC_BASE_NAMES and len(b.split()) <= 2 and len(b) > 2}
    except FileNotFoundError:
        pass
    vocab = {normalize_term(v) for v in vocab if v} - NOT_FOOD - {""}
    return vocab


@lru_cache(maxsize=1)
def _vocab_regex():
    alts = "|".join(re.escape(v).replace(r"\ ", r"\s+") for v in sorted(food_vocabulary(), key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})(?:s|es)?\b", re.I)


@lru_cache(maxsize=1)
def load_ner():
    from transformers import pipeline
    return pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")


def _expand_to_words(text, start, end):
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "-"):
        start -= 1
    while end < len(text) and (text[end].isalnum() or text[end] == "-"):
        end += 1
    return start, end


def _merge_spans(text, spans):
    """Merge overlapping spans and spans separated only by spaces ("grilled" + "chicken")."""
    merged = []
    for s, e, src in sorted(spans):
        if merged and (s <= merged[-1][1] or text[merged[-1][1]:s].isspace()):
            ps, pe, psrc = merged[-1]
            merged[-1] = (ps, max(pe, e), psrc if psrc == src else "ner+dict")
        else:
            merged.append((s, e, src))
    return merged


def find_food_spans(text, use_ner=True, use_dict=True):
    """Return [(start, end, source)] of food mentions in text."""
    spans = []
    if use_ner:
        for ent in load_ner()(text):
            s, e = _expand_to_words(text, ent["start"], ent["end"])
            spans.append((s, e, "ner"))
    if use_dict:
        spans += [(m.start(), m.end(), "dict") for m in _vocab_regex().finditer(text)]
    spans = _merge_spans(text, spans)

    cleaned = []
    for s, e, src in spans:
        # Drop non-food words the model sometimes tags (e.g. "lactose intolerant" -> "lactose").
        words = [w for w in re.findall(r"[\w'-]+", text[s:e]) if w.lower() not in NOT_FOOD and not _ALL_PRE_RE.fullmatch(w)]
        if not words:
            continue
        first, last = text.find(words[0], s), text.rfind(words[-1], s, e)
        cleaned.append((first, last + len(words[-1]), src))
    return cleaned


# --- classification ----------------------------------------------------------------------------------
@dataclass
class Mention:
    term: str
    label: str      # block / avoid / prefer / unknown
    cue: str
    source: str     # ner / dict / ner+dict / diet
    groups: set = field(default_factory=set)
    flags: set = field(default_factory=set)


@dataclass
class Preferences:
    block_terms: set = field(default_factory=set)
    block_groups: set = field(default_factory=set)
    block_flags: set = field(default_factory=set)
    avoid_terms: set = field(default_factory=set)
    avoid_groups: set = field(default_factory=set)
    prefer_terms: set = field(default_factory=set)
    mentions: list = field(default_factory=list)

    def summary(self):
        return {
            "block": sorted(self.block_terms), "block_allergen_groups": sorted(self.block_groups),
            "block_diet_flags": sorted(self.block_flags), "avoid": sorted(self.avoid_terms),
            "avoid_groups": sorted(self.avoid_groups), "prefer": sorted(self.prefer_terms),
        }


def _clauses(text):
    """Yield (start, end) of each clause."""
    start = 0
    for m in CLAUSE_SPLIT_RE.finditer(text):
        yield start, m.start()
        start = m.end()
    yield start, len(text)


def classify(clause, s, e):
    """Label a food span inside a clause by the nearest cue. Returns (label, cue)."""
    after = clause[e:e + 25]
    for label, rx in _POST_CUE_RE.items():
        m = rx.match(after)
        if m:
            return label, m.group(0).strip()
    before = [m for m in _ALL_PRE_RE.finditer(clause[:s])]
    if before:
        cue = before[-1].group(0)
        key = re.sub(r"\s+", " ", cue.lower()).replace("'", "").replace("’", "")
        return _LABEL_OF.get(key, "unknown"), cue
    later = _ALL_PRE_RE.search(clause, e)
    if later:  # e.g. "mushrooms, I hate them"
        key = re.sub(r"\s+", " ", later.group(0).lower()).replace("'", "").replace("’", "")
        return _LABEL_OF.get(key, "unknown"), later.group(0)
    return "unknown", ""


def parse_preferences(text, use_ner=True, use_dict=True):
    prefs = Preferences()
    if not text or not text.strip():
        return prefs

    for pattern, rule in DIETS.items():
        if re.search(pattern, text, re.I):
            m = Mention(re.search(pattern, text, re.I).group(0).lower(), "block", "diet", "diet",
                        set(rule.get("groups", set())), set(rule.get("flags", set())))
            prefs.mentions.append(m)
            prefs.block_groups |= m.groups
            prefs.block_flags |= m.flags
            prefs.block_terms |= set(rule.get("terms", set()))

    for cs, ce in _clauses(text):
        clause = text[cs:ce]
        for s, e, src in find_food_spans(clause, use_ner, use_dict):
            term = normalize_term(clause[s:e])
            if not term:
                continue
            if term.endswith("-free"):  # "gluten-free", "dairy-free"
                term, label, cue = term[:-5], "block", "-free"
            else:
                label, cue = classify(clause, s, e)
            groups = set(CATEGORY_GROUPS.get(term, set()))
            flags = set(FLAG_WORDS.get(term, set()))
            if label == "block":
                # Safety cross-check: the deterministic dictionary decides the allergen groups too.
                groups |= set(tag_allergens(term))
            m = Mention(term, label, cue, src, groups, flags)
            prefs.mentions.append(m)
            if label == "block":
                prefs.block_terms.add(term)
                prefs.block_groups |= groups
                prefs.block_flags |= flags
            elif label == "avoid":
                prefs.avoid_terms.add(term)
                prefs.avoid_groups |= groups
            elif label == "prefer":
                prefs.prefer_terms.add(term)

    # Allergy safety net: any allergen word after a block cue is blocked, even if no food span was found.
    # The scanned segment runs from the cue to the next avoid/prefer cue (or the end of the clause).
    for cs, ce in _clauses(text):
        clause = text[cs:ce]
        for cue in _PRE_CUE_RE["block"].finditer(clause):
            nxt = [m.start() for label in ("avoid", "prefer") for m in _PRE_CUE_RE[label].finditer(clause, cue.end())]
            segment = clause[cue.end():min(nxt, default=len(clause))]
            extra = set(tag_allergens(segment)) | {g for w, gs in CATEGORY_GROUPS.items()
                                                    if re.search(rf"\b{re.escape(w)}\b", segment, re.I) for g in gs}
            if extra - prefs.block_groups:
                prefs.mentions.append(Mention(segment.strip(), "block", "safety-net", "dict", extra - prefs.block_groups))
                prefs.block_groups |= extra
    if "wheat" in prefs.block_groups:
        prefs.block_groups.add("gluten")
    if "gluten" in prefs.block_groups:
        prefs.block_groups.add("wheat")
    return prefs


# --- applying preferences to the food table -----------------------------------------------------------
def match_term(desc, term):
    """Foods whose description contains the term; "grilled chicken" falls back to "chicken" if nothing matches."""
    hits = desc.str.contains(term_regex(term))
    if not hits.any() and " " in term:
        hits = desc.str.contains(term_regex(term.split()[-1]))
    return hits


def apply_to_foods(foods, prefs):
    """Return (allowed_foods, blocked_foods). allowed_foods gets a `pref_score` column (+ prefer / - avoid)."""
    df = foods.copy()
    allergens = df["allergens"].fillna("").str.split("|").map(set)
    desc = df["description"]

    blocked = allergens.map(lambda a: bool(a & prefs.block_groups))
    for term in prefs.block_terms:
        blocked |= match_term(desc, term)
    if "pork" in prefs.block_flags:
        blocked |= df["contains_pork"].astype(bool)
    if "alcohol" in prefs.block_flags:
        blocked |= df["contains_alcohol"].astype(bool)
    if "meat" in prefs.block_flags:
        blocked |= ~df["is_vegetarian"].astype(bool)
    if "meat_not_fish" in prefs.block_flags:
        blocked |= ~df["is_vegetarian"].astype(bool) & (df["food_category"] != "Finfish and Shellfish Products")

    score = pd.Series(0.0, index=df.index)
    for term in prefs.prefer_terms:
        score += match_term(desc, term).astype(float)
    for term in prefs.avoid_terms:
        score -= match_term(desc, term).astype(float)
    score -= allergens.map(lambda a: len(a & prefs.avoid_groups)).astype(float)
    df["pref_score"] = score
    return df[~blocked].copy(), df[blocked].copy()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    text = " ".join(sys.argv[1:]) or "I hate mushrooms and I'm allergic to shrimp"
    p = parse_preferences(text)
    print("Input:", text)
    for m in p.mentions:
        print(f"  {m.label:7} {m.term!r:24} cue={m.cue!r:16} source={m.source:9} groups={sorted(m.groups)} flags={sorted(m.flags)}")
    print("Summary:", p.summary())
    foods = pd.read_csv(FOODS_CLEAN)
    allowed, blocked = apply_to_foods(foods, p)
    print(f"Foods allowed: {len(allowed)}  blocked: {len(blocked)}")
    print("Top preferred:", allowed.sort_values("pref_score", ascending=False)["description"].head(5).tolist())
