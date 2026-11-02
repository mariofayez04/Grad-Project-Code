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

Safety layer (added after testing on wording the rules were not built with):
 - Spelling mistakes are corrected first ("alergic to penuts"), and the app shows what it corrected.
 - A body reaction ("shellfish gives me hives"), a condition ("I have celiac disease") or a doctor's advice counts as
   a block, even with no allergy word. Over-blocking is the safe direction.
 - If health wording is found but no food could be blocked, or a food is named with no way to tell what the person
   means ("eating cod is not safe for me"), the clause goes in `unclear`, so the app asks instead of staying silent.
   No list of phrases can cover every wording: on sentences kept apart from the rules, 68.8% of allergy items were
   blocked (data/evaluation/simulated_testers.md), so asking is part of the safety design.

The allergy checklist in the profile (argument `allergies`) needs no language model at all: a ticked group is always
blocked. Free text remains for likes, dislikes and allergies the checklist doesn't list.

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
        "avoid completely", "must avoid", "react to", "reaction to", "stay away from", "keep away from",
        "keep me away from", "stay off", "cut out", "sensitive to", "sensitivity to", "can't touch", "cant touch",
        "can't go near", "cannot go near", "not supposed to eat", "not supposed to have", "have to avoid", "told to avoid",
        "should not eat", "shouldn't eat", "must not eat", "must never eat", "cant have", "can not have", "cant tolerate",
        "cannot tolerate",
    ],
    "avoid": [
        "hate", "hates", "dislike", "dislikes", "don't like", "dont like", "do not like", "don't love", "don't enjoy",
        "do not enjoy", "not a fan of", "not into", "not really into", "not big on", "not fond of", "can't stand",
        "cant stand", "avoid", "rather not", "prefer not", "would rather not", "don't want", "do not want", "less",
        "fewer", "sick of", "tired of", "disgusting", "gross", "not keen on", "not a big fan of", "except", "apart from",
    ],
    "prefer": [
        "love", "loves", "like", "likes", "really like", "enjoy", "enjoys", "favorite", "favourite", "favorites",
        "prefer", "prefers", "fan of", "big fan of", "into", "want", "more", "crave", "adore", "fond of", "keen on",
    ],
}
# Cues that come right AFTER the food ("shrimp allergy", "gluten-free", "chicken is my favorite").
POST_CUES = {
    "block": ["allergy", "allergies", "intolerance", "sensitivity", "free", "-free", "is off limits", "are off limits"],
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
            "meal", "meals", "breakfast", "lunch", "dinner", "snack", "snacks", "diet", "free", "please", "intolerance",
            "allergist", "doctor", "dietitian", "dietician", "nutritionist", "hospital", "emergency", "epipen",
            "hive", "hives", "rash", "throat", "stomach", "lip", "lips"}

# Health wording. A body reaction blocks the food it is said about, whatever else the sentence says.
REACTION_RE = re.compile(
    r"\b(?:hives|rash(?:es)?|swell\w*|swollen|anaphyla\w*|epipen|throat|breath\w*|itch\w*|vomit\w*|throw\w*\s+up|nause\w*"
    r"|sick(?!\s+of)|ill|stomach|bloat\w*|diarrh\w*|cramps?|reacts?|reaction|emergency|hospital\w*|dangerous|deadly|unsafe"
    r"|poison\w*|(?:doesn['’]?t|don['’]?t|does\s+not|do\s+not)\s+agree\s+with|problem\s+for\s+me)\b", re.I)
# Medical advice or a strict rule turns "avoid" into a block ("my doctor told me to avoid shrimp").
MEDICAL_RE = re.compile(r"\b(?:doctor|dr|gp|allergist|dietitian|dietician|nutritionist|medical|medically|diagnosed|hospital)\b"
                        r"|\bavoid\b.*\b(?:completely|entirely|strictly|totally|always)\b", re.I)
HEALTH_WORD_RE = re.compile(rf"\b(?:allerg\w*|intoleran\w*|sensitiv\w*|coeliac|celiac)\b|{REACTION_RE.pattern}|{MEDICAL_RE.pattern}", re.I)
# These can be said about a food someone likes ("yogurt is good for my stomach"), so alone they don't overrule a like.
WEAK_REACTION_RE = re.compile(r"^(?:stomach|throat|breath\w*)$", re.I)
PRONOUN_RE = re.compile(r"\b(?:it|them|they|those|these|that)\b", re.I)
NO_ALLERGY_RE = re.compile(r"\b(?:no|not|without|any)\b.{0,20}\ballerg|\ballerg\w*\b.{0,20}\bnone\b", re.I)
NOT_A_PROBLEM_RE = re.compile(r"\bno\s+(?:problems?|issues?)\s+with\b", re.I)
# Conditions that decide the food rule without naming a food.
CONDITIONS = {
    r"\bc(?:o)?eliac\b": {"groups": {"gluten", "wheat"}},
    r"\bfavism\b|\bg6pd\b": {"terms": {"fava", "broadbean", "ful medames", "ta'ameya"}},
}
# Conditions this app has no rule for: it says so instead of staying silent.
UNHANDLED_RE = re.compile(r"\b(?:kidney|gout|pregnan\w*|breastfeeding|cancer|liver\s+disease|eating\s+disorder|anorexia|bulimia)\b", re.I)
SLANG = {"luv": "love", "n": "and"}
JOINED = {r"\begg\s+plant": "eggplant", r"\bpea\s+nut": "peanut", r"\bsea\s+food": "seafood", r"\bshell\s+fish": "shellfish"}

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


# --- spelling mistakes -------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _english_words():
    """Ordinary English words (the NER model's own vocabulary): "break" and "dream" must not become "bread" and "cream"."""
    try:
        from transformers import AutoTokenizer
        return {w.lower() for w in AutoTokenizer.from_pretrained(NER_MODEL).get_vocab() if w.isalpha()}
    except Exception:
        return set()


@lru_cache(maxsize=1)
def _food_words():
    """Every single word in the food table's names, so that a real food ("chickpeas") is never "corrected"."""
    try:
        names = pd.read_csv(FOODS_CLEAN, usecols=["description"])["description"]
    except FileNotFoundError:
        return set()
    return {singular(w) for text in names for w in re.findall(r"[a-z]{3,}", text.lower())}


@lru_cache(maxsize=1)
def _spelling_targets():
    """Words a misspelling can be corrected to: (safety words, all food words), each by first letter."""
    safety = {"allergic", "allergy", "allergies", "intolerant", "intolerance", "vegetarian", "pescatarian", "celiac", "coeliac"}
    safety |= {w for ws in ALLERGEN_TRIGGERS.values() for w in ws if " " not in w} | {w for w in CATEGORY_GROUPS if " " not in w}
    safety |= set(GLUTEN_EXTRA)
    foods = {w for w in food_vocabulary() if " " not in w and "-" not in w} | {w for w in _food_words() if len(w) >= 5}
    index = lambda words: {c: [w for w in words if w[0] == c] for c in {w[0] for w in words}}
    return index(safety), index(foods)


def _distance(a, b, limit):
    """Edit distance counting a swap of two neighbouring letters as one; stops early past `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            cost = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ca != cb))
            if prev2 is not None and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cost = min(cost, prev2[j - 2] + 1)
            row.append(cost)
        prev2, prev = prev, row
    return prev[-1]


def _closest(word, candidates, limit):
    best = [(d, w) for w in candidates if (d := _distance(word, w, limit)) <= limit]
    return min(best)[1] if best else None


def correct_word(word):
    """The food or allergy word a misspelling stands for, or None. Allergy words win over other foods."""
    w = word.lower()
    if (len(w) < 5 or w in _english_words() or w in _LABEL_OF or singular(w) in food_vocabulary()
            or singular(w) in _food_words()):
        return None
    safety, foods = _spelling_targets()
    limit = 2 if len(w) >= 7 else 1
    return (_closest(w, safety.get(w[0], []), limit) or _closest(singular(w), safety.get(w[0], []), limit)
            or _closest(singular(w), foods.get(w[0], []), 2 if len(w) >= 8 else 1))


def fix_spelling(text):
    """Returns (corrected text, {what was typed: what it was read as})."""
    fixes = {}

    def fix(match):
        word = match.group(0)
        new = SLANG.get(word.lower()) or correct_word(word)
        if new and new != word.lower():
            fixes[word] = new
            return new
        return word

    for pattern, joined in JOINED.items():
        text = re.sub(pattern, joined, text, flags=re.I)
    return re.sub(r"(?<![\w'’-])[A-Za-z]+(?![\w'’-])", fix, text), fixes


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
    corrections: dict = field(default_factory=dict)  # spelling mistakes: {typed: read as}
    unclear: list = field(default_factory=list)      # wording the parser could not act on safely: the app asks

    def summary(self):
        return {
            "block": sorted(self.block_terms), "block_allergen_groups": sorted(self.block_groups),
            "block_diet_flags": sorted(self.block_flags), "avoid": sorted(self.avoid_terms),
            "avoid_groups": sorted(self.avoid_groups), "prefer": sorted(self.prefer_terms),
            "corrections": self.corrections, "unclear": self.unclear,
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


def _link_gluten(prefs):
    """Wheat and gluten are blocked together."""
    if prefs.block_groups & {"wheat", "gluten"}:
        prefs.block_groups |= {"wheat", "gluten"}


def _part_around(clause, s, e):
    """The piece of a clause between the commas around a food."""
    end = clause.find(",", e)
    return clause[clause.rfind(",", 0, s) + 1:end if end >= 0 else len(clause)]


def _is_real_food(mention):
    """The model sometimes tags a word that isn't food ("terrible hives"): blocking that protects nobody."""
    return bool(mention.groups) or any(singular(w) in food_vocabulary() or singular(w) in _food_words() for w in mention.term.split())


def health_upgrade(clause, s, e, label, cue):
    """Health wording overrules a softer label. Blocking too much is safe; serving an allergen is not."""
    if label == "block":
        return label, cue
    reaction = REACTION_RE.search(_part_around(clause, s, e)) or (label == "unknown" and REACTION_RE.search(clause))
    if reaction and not (label == "prefer" and WEAK_REACTION_RE.match(reaction.group(0))):
        return "block", reaction.group(0)
    if label == "avoid" and MEDICAL_RE.search(clause):
        return "block", MEDICAL_RE.search(clause).group(0)
    return label, cue


def parse_preferences(text, use_ner=True, use_dict=True, allergies=""):
    """allergies: groups ticked in the checklist, joined by | (e.g. "milk|peanut"). They are always blocked."""
    prefs = Preferences()
    for group in filter(None, (allergies or "").split("|")):
        prefs.mentions.append(Mention(group, "block", "checklist", "checklist", {group}))
        prefs.block_groups.add(group)
    if not text or not text.strip():
        _link_gluten(prefs)
        return prefs
    text, prefs.corrections = fix_spelling(NOT_A_PROBLEM_RE.sub("fine with", text))

    for pattern, rule in CONDITIONS.items():
        found = re.search(pattern, text, re.I)
        if found:
            prefs.mentions.append(Mention(found.group(0).lower(), "block", "condition", "diet", set(rule.get("groups", set()))))
            prefs.block_groups |= set(rule.get("groups", set()))
            prefs.block_terms |= set(rule.get("terms", set()))

    for pattern, rule in DIETS.items():
        if re.search(pattern, text, re.I):
            m = Mention(re.search(pattern, text, re.I).group(0).lower(), "block", "diet", "diet",
                        set(rule.get("groups", set())), set(rule.get("flags", set())))
            prefs.mentions.append(m)
            prefs.block_groups |= m.groups
            prefs.block_flags |= m.flags
            prefs.block_terms |= set(rule.get("terms", set()))

    previous = []  # foods of the clause before, for "I love shrimp but it gives me hives"
    for cs, ce in _clauses(text):
        clause = text[cs:ce]
        spans = find_food_spans(clause, use_ner, use_dict)
        if not spans and previous and REACTION_RE.search(clause) and PRONOUN_RE.search(clause):
            for m in previous:
                groups = m.groups | set(tag_allergens(m.term))
                prefs.prefer_terms.discard(m.term)
                prefs.avoid_terms.discard(m.term)
                prefs.mentions.append(Mention(m.term, "block", REACTION_RE.search(clause).group(0), m.source, groups, m.flags))
                prefs.block_terms.add(m.term)
                prefs.block_groups |= groups
                prefs.block_flags |= m.flags
        previous = []
        for s, e, src in spans:
            term = normalize_term(clause[s:e])
            if not term:
                continue
            if term.endswith("-free"):  # "gluten-free", "dairy-free"
                term, label, cue = term[:-5], "block", "-free"
            else:
                label, cue = health_upgrade(clause, s, e, *classify(clause, s, e))
            groups = set(CATEGORY_GROUPS.get(term, set()))
            flags = set(FLAG_WORDS.get(term, set()))
            if label == "block":
                # Safety cross-check: the deterministic dictionary decides the allergen groups too.
                groups |= set(tag_allergens(term))
            m = Mention(term, label, cue, src, groups, flags)
            prefs.mentions.append(m)
            previous.append(m)
            if label == "unknown":
                prefs.unclear.append(clause.strip())
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
        # The same net for body reactions: an allergen word next to one is blocked, even if no food span was found.
        for part in clause.split(","):
            if REACTION_RE.search(part):
                extra = set(tag_allergens(part)) - prefs.block_groups
                if extra:
                    prefs.mentions.append(Mention(part.strip(), "block", "safety-net", "dict", extra))
                    prefs.block_groups |= extra
        # Health wording, but nothing blocked from this clause: say so, so that the person can reword it.
        lowered = clause.lower()
        blocked_here = any(m.label == "block" and m.term in lowered and _is_real_food(m) for m in prefs.mentions)
        if (HEALTH_WORD_RE.search(clause) and not NO_ALLERGY_RE.search(clause) and not blocked_here) or UNHANDLED_RE.search(clause):
            prefs.unclear.append(clause.strip())
    prefs.unclear = list(dict.fromkeys(prefs.unclear))
    _link_gluten(prefs)
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
