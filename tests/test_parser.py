"""Phase 3 tests: hand-labelled sentences for the preference parser.

The same cases are reused in Phase 8 to report parser accuracy.
Run:  python -m pytest tests -q
"""
import pandas as pd
import pytest

from src.config import FOODS_CLEAN
from src.parser import apply_to_foods, parse_preferences

# (sentence, expected labels {term: label}, expected blocked allergen groups, expected blocked diet flags)
CASES = [
    ("I hate mushrooms and I'm allergic to shrimp", {"mushroom": "avoid", "shrimp": "block"}, {"shellfish"}, set()),
    ("I love grilled chicken and salmon, but I don't like broccoli.",
     {"grilled chicken": "prefer", "salmon": "prefer", "broccoli": "avoid"}, set(), set()),
    ("allergic to peanuts and tree nuts. no dairy please",
     {"peanut": "block", "tree nut": "block", "dairy": "block"}, {"peanut", "tree_nut", "milk"}, set()),
    ("I'm lactose intolerant and I can't eat gluten", {"lactose": "block", "gluten": "block"}, {"milk", "gluten", "wheat"}, set()),
    ("I'm not really into shellfish these days", {"shellfish": "avoid"}, set(), set()),
    ("I enjoy rice, lentils and eggs for breakfast", {"rice": "prefer", "lentil": "prefer", "egg": "prefer"}, set(), set()),
    ("i hate onions, love beef", {"onion": "avoid", "beef": "prefer"}, set(), set()),
    ("I'm vegetarian and I don't eat pork", {"pork": "block"}, set(), {"meat", "pork"}),
    ("I prefer oatmeal and bananas", {"oatmeal": "prefer", "banana": "prefer"}, set(), set()),
    ("Allergic to sesame. Dislike eggplant and spinach.", {"sesame": "block", "eggplant": "avoid", "spinach": "avoid"}, {"sesame"}, set()),
    ("I have a severe peanut allergy", {"peanut": "block"}, {"peanut"}, set()),
    ("gluten-free diet please, I like pasta", {"gluten": "block", "pasta": "prefer"}, {"gluten"}, set()),
    ("I hate shrimp and I'm allergic to peanuts", {"shrimp": "avoid", "peanut": "block"}, {"peanut"}, set()),
    ("mushrooms, I can't stand them", {"mushroom": "avoid"}, set(), set()),
    ("I'm halal and love lamb", {"lamb": "prefer"}, set(), {"pork", "alcohol"}),
    ("I am vegan", {}, {"milk", "egg"}, {"meat"}),
    ("I'm allergic to fish", {"fish": "block"}, {"fish"}, set()),
    ("My favorite food is pizza", {"pizza": "prefer"}, set(), set()),
    ("I don't like tomatoes but I love potatoes", {"tomato": "avoid", "potato": "prefer"}, set(), set()),
    ("Please no seafood", {"seafood": "block"}, {"fish", "shellfish"}, set()),
    ("I can't eat eggs", {"egg": "block"}, {"egg"}, set()),
    ("I really like yogurt and cheese", {"yogurt": "prefer", "cheese": "prefer"}, set(), set()),
    ("I dislike cabbage", {"cabbage": "avoid"}, set(), set()),
    ("allergy to soy and wheat", {"soy": "block", "wheat": "block"}, {"soy", "wheat", "gluten"}, set()),
    ("I'm not a fan of fish, I prefer chicken", {"fish": "avoid", "chicken": "prefer"}, set(), set()),
    ("Chicken is my favorite", {"chicken": "prefer"}, set(), set()),
    ("I'm intolerant to milk", {"milk": "block"}, {"milk"}, set()),
    ("I want more beans and less red meat", {"bean": "prefer", "red meat": "avoid"}, set(), set()),
    ("I eat everything", {}, set(), set()),
    ("", {}, set(), set()),
    ("I have a shrimp allergy and I hate olives", {"shrimp": "block", "olive": "avoid"}, {"shellfish"}, set()),
    ("I'm allergic to almonds", {"almond": "block"}, {"tree_nut"}, set()),
]


@pytest.fixture(scope="module")
def foods():
    return pd.read_csv(FOODS_CLEAN)


@pytest.mark.parametrize("text,labels,groups,flags", CASES)
def test_parser(text, labels, groups, flags):
    p = parse_preferences(text)
    got = {m.term: m.label for m in p.mentions if m.source != "diet" and m.cue != "safety-net"}
    for term, label in labels.items():
        assert got.get(term) == label, f"{term!r}: expected {label}, got {got}"
    assert groups <= p.block_groups, f"missing allergen groups: {groups - p.block_groups}"
    assert flags <= p.block_flags, f"missing diet flags: {flags - p.block_flags}"
    # Nothing that is only disliked/preferred may become a hard block.
    for term, label in labels.items():
        if label != "block":
            assert term not in p.block_terms


def test_safety_net_without_ner():
    """Even with the ML model switched off, allergies are still caught by the dictionary."""
    p = parse_preferences("I'm allergic to peanuts and shrimp", use_ner=False)
    assert {"peanut", "shellfish"} <= p.block_groups


def test_blocked_foods_removed(foods):
    p = parse_preferences("I'm allergic to shrimp and peanuts and I hate mushrooms")
    allowed, blocked = apply_to_foods(foods, p)
    allergens = allowed["allergens"].fillna("")
    assert not allergens.str.contains("shellfish|peanut").any()
    assert not allowed["description"].str.contains(r"\bshrimp", case=False).any()
    assert (allowed.loc[allowed["description"].str.contains(r"\bmushroom", case=False), "pref_score"] < 0).all()
    assert len(allowed) + len(blocked) == len(foods)


def test_vegetarian_halal(foods):
    allowed, _ = apply_to_foods(foods, parse_preferences("I'm vegetarian"))
    assert allowed["is_vegetarian"].all()
    allowed, _ = apply_to_foods(foods, parse_preferences("halal only"))
    assert not allowed["contains_pork"].any() and not allowed["contains_alcohol"].any()


# --- safety layer: wording found by testing on sentences the rules were not built with ------------------------
@pytest.mark.parametrize("text,groups", [
    ("im alergic to peanuts", {"peanut"}),                         # spelling
    ("i am alergic to sesamee", {"sesame"}),
    ("i cant eat glutin", {"gluten"}),
    ("allergik to nuts", {"peanut", "tree_nut"}),
    ("Shellfish gives me hives", {"shellfish"}),                   # reactions
    ("Walnuts make my throat swell.", {"tree_nut"}),
    ("Milk upsets my stomach so I stay away from it", {"milk"}),
    ("Crab makes me throw up.", {"shellfish"}),
    ("I love shrimp but it gives me hives", {"shellfish"}),        # the reaction is in the next clause
    ("I have celiac disease", {"gluten", "wheat"}),                # conditions
    ("I'm coeliac.", {"gluten", "wheat"}),
    ("My doctor told me to avoid shrimp completely.", {"shellfish"}),   # medical advice
    ("My dietitian wants me to cut out milk completely.", {"milk"}),
])
def test_health_wording_blocks(text, groups):
    assert groups <= parse_preferences(text).block_groups


@pytest.mark.parametrize("text,allowed_group", [
    ("I have no problem with nuts", "peanut"),
    ("I'm not a big fan of tuna", "fish"),                         # a dislike stays a dislike
    ("I love chicken, my doctor says to eat more fish", "fish"),   # advice to eat more is not a block
    ("I had a dream about bread and I love it", "wheat"),
])
def test_ordinary_wording_blocks_nothing(text, allowed_group):
    assert allowed_group not in parse_preferences(text).block_groups


def test_dislike_with_big_fan_is_not_a_like():
    assert "tuna" in parse_preferences("Not a big fan of tuna").avoid_terms


@pytest.mark.parametrize("word", ["chickpeas", "break", "dream", "lentils", "koshari", "means", "people"])
def test_spelling_leaves_real_words_alone(word):
    assert parse_preferences(f"I love {word}").corrections == {}


def test_spelling_is_reported():
    assert parse_preferences("im alergic to peanuts").corrections == {"alergic": "allergic"}


def test_unclear_health_wording_is_flagged():
    assert parse_preferences("I get terrible hives sometimes").unclear
    assert parse_preferences("I have kidney disease").unclear
    assert parse_preferences("Mangoes are a bit of a worry").unclear  # a food, but no way to tell what is meant
    assert not parse_preferences("I'm allergic to eggs").unclear
    assert not parse_preferences("No allergies, I love rice").unclear


def test_favism_blocks_fava_dishes(foods):
    allowed, _ = apply_to_foods(foods, parse_preferences("I have favism"))
    assert not allowed["description"].str.contains("fava|broadbean|ful medames", case=False).any()
