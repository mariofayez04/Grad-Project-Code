"""Hand-labelled allergen cases, including known traps.

Run:  python -m pytest tests -q
"""
import pytest

from src.allergens import diet_flags, tag_allergens

CASES = [
    # (description, category, must_have, must_not_have)
    ("Eggplant, raw", "Vegetables and Vegetable Products", [], ["egg"]),
    ("Egg, whole, cooked, hard-boiled", "Dairy and Egg Products", ["egg"], ["milk"]),
    ("Waffles, buttermilk, frozen, ready-to-heat", "Baked Products", ["milk", "wheat", "gluten"], []),
    ("Crackers, snack, Goya Crackers", "Baked Products", ["wheat"], []),
    ("Bread, pound cake type, pan de torta salvadoran", "Baked Products", ["wheat"], []),
    ("Peanut butter, smooth style, without salt", "Legumes and Legume Products", ["peanut"], ["milk", "tree_nut"]),
    ("Nuts, almond butter, plain, without salt added", "Nut and Seed Products", ["tree_nut"], ["milk"]),
    ("Beverages, almond milk, unsweetened", "Beverages", ["tree_nut"], ["milk"]),
    ("Nuts, coconut milk, raw", "Nut and Seed Products", [], ["milk"]),
    ("Water chestnuts, chinese, canned", "Vegetables and Vegetable Products", [], ["tree_nut"]),
    ("Pasta, gluten-free, corn, cooked", "Cereal Grains and Pasta", [], ["wheat", "gluten"]),
    ("Tortillas, ready-to-bake or -fry, corn", "Baked Products", [], ["wheat"]),
    ("Cream of tartar", "Baked Products", [], ["milk"]),
    ("Squash, winter, butternut, cooked, baked", "Vegetables and Vegetable Products", [], ["milk", "tree_nut"]),
    ("Beans, kidney, red, mature seeds, cooked", "Legumes and Legume Products", [], ["milk", "wheat"]),
    ("Crustaceans, shrimp, cooked", "Finfish and Shellfish Products", ["shellfish"], ["fish"]),
    ("Mollusks, oyster, eastern, cooked", "Finfish and Shellfish Products", ["shellfish"], []),
    ("Fish, salmon, Atlantic, farmed, cooked", "Finfish and Shellfish Products", ["fish"], ["shellfish"]),
    ("Tilapia, cooked, dry heat", "Finfish and Shellfish Products", ["fish"], []),
    ("Tofu, firm, prepared with calcium sulfate", "Legumes and Legume Products", ["soy"], ["milk"]),
    ("Hummus, commercial", "Legumes and Legume Products", ["sesame"], []),
    ("Barley, pearled, cooked", "Cereal Grains and Pasta", ["gluten"], ["wheat"]),
    ("Rice, white, long-grain, cooked", "Cereal Grains and Pasta", [], ["wheat", "gluten"]),
    ("Cheese, cheddar", "Dairy and Egg Products", ["milk"], []),
    ("Yogurt, Greek, plain, nonfat", "Dairy and Egg Products", ["milk"], []),
    ("Mayonnaise, regular", "Fats and Oils", ["egg"], []),
    ("Noodles, egg, cooked, enriched", "Cereal Grains and Pasta", ["egg", "wheat"], []),
    ("Chicken, broilers or fryers, breast, meat only, cooked, roasted", "Poultry Products", [], ["milk", "egg", "fish"]),
    ("Cabbage, common (danish, domestic, and pointed types), stored, raw", "Vegetables and Vegetable Products", [], ["wheat"]),
    ("Burrito, beef and bean, frozen", "Meals, Entrees, and Side Dishes", ["wheat"], []),
    ("Pancakes, gluten-free, frozen, ready-to-heat", "Baked Products", ["milk", "egg"], ["wheat"]),
    ("Candies, sweet chocolate", "Sweets", ["milk"], []),
]


@pytest.mark.parametrize("desc,cat,must,must_not", CASES)
def test_allergen_tags(desc, cat, must, must_not):
    tags = tag_allergens(desc, cat)
    for g in must:
        assert g in tags, f"{desc!r} should be tagged {g}, got {tags}"
    for g in must_not:
        assert g not in tags, f"{desc!r} should NOT be tagged {g}, got {tags}"


@pytest.mark.parametrize("desc,cat,veg,pork", [
    ("Nuts, coconut meat, dried", "Nut and Seed Products", True, False),
    ("Chicken, breast, cooked", "Poultry Products", False, False),
    ("Soup, vegetable broth, ready to serve", "Soups, Sauces, and Gravies", True, False),
    ("Soup, pea, split with ham, canned", "Soups, Sauces, and Gravies", False, True),
    ("Lentils, mature seeds, cooked, boiled", "Legumes and Legume Products", True, False),
])
def test_diet_flags(desc, cat, veg, pork):
    is_veg, has_pork, _ = diet_flags(desc, cat)
    assert (is_veg, has_pork) == (veg, pork)
