"""Phase 1, step 1: print the real structure of the USDA SR Legacy tables.

Run:  python -m src.explore_usda
"""
import pandas as pd

from src.config import USDA_DIR, NUTRIENT_IDS

TABLES = ["food", "food_nutrient", "nutrient", "food_category", "food_portion", "measure_unit", "food_attribute", "food_attribute_type"]


def load(name):
    return pd.read_csv(USDA_DIR / f"{name}.csv", low_memory=False)


def main():
    t = {name: load(name) for name in TABLES}
    for name, df in t.items():
        print(f"\n=== {name}.csv  ({len(df):,} rows)")
        print("columns:", list(df.columns))

    nut = t["nutrient"]
    print("\n=== nutrients we need")
    print(nut[nut["id"].isin(NUTRIENT_IDS.values())][["id", "name", "unit_name"]].to_string(index=False))
    print("\nother energy rows:")
    print(nut[nut["name"].str.contains("Energy")][["id", "name", "unit_name"]].to_string(index=False))

    fn = t["food_nutrient"]
    print("\n=== coverage: foods having each nutrient (of", t["food"]["fdc_id"].nunique(), ")")
    for col, nid in NUTRIENT_IDS.items():
        print(f"  {col:10} id {nid}: {fn.loc[fn['nutrient_id'] == nid, 'fdc_id'].nunique():,}")

    print("\n=== food_category")
    counts = t["food"]["food_category_id"].value_counts()
    cat = t["food_category"].assign(n_foods=lambda d: d["id"].map(counts))
    print(cat[["id", "description", "n_foods"]].to_string(index=False))

    p = t["food_portion"].merge(t["measure_unit"], left_on="measure_unit_id", right_on="id", suffixes=("", "_unit"))
    print("\n=== food_portion")
    print("foods with >=1 portion:", p["fdc_id"].nunique())
    print("measure units used:\n", p["name"].value_counts().head(10).to_string())
    print("sample rows:")
    print(p[["fdc_id", "amount", "name", "modifier", "gram_weight"]].head(12).to_string(index=False))

    print("\n=== food_attribute types")
    a = t["food_attribute"].merge(t["food_attribute_type"], left_on="food_attribute_type_id", right_on="id", suffixes=("", "_type"))
    print(a["name_type"].value_counts().to_string())
    print(a[["fdc_id", "name_type", "value"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
