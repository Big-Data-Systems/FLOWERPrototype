import pandas as pd

cats = pd.read_csv("product_categories.csv")
product_types = pd.read_csv("product_types.csv")
products = pd.read_csv("products.csv")

# join into intermediate states
prods_with_type_info = products.merge(product_types, how="left", left_on="type_id_fk", right_on="type_id")
prods_with_cats = prods_with_type_info.merge(cats, how="left", left_on="cat_id_fk", right_on="cat_id")

prods_with_cats.to_csv("processed__product_info.csv", index=False)

# aggregate information
cat_costs = prods_with_cats.groupby("cat_id_fk").agg(
    cat_id=("cat_id_fk", "first"),
    sum_costs=("cost", "sum"),
    inventory=("cat_id_fk", "count")
)

cat_costs.to_csv("processed__category_inventory.csv", index=False)

# create a new frame that will not be considered as a state for FLOWER as there is no output descendant
useless = prods_with_cats.groupby("type_id_fk").sum(numeric_only=True)