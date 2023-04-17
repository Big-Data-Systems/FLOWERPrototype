import pandas as pd

"""
Recommend products based on category sales
"""


cat_sales = pd.read_csv("processed__category_sales.csv")
cat_inv = pd.read_csv("processed__category_inventory.csv")

cat_summary = cat_sales.merge(cat_inv, how='inner', on='cat_id')
cat_summary['profit_potential'] = cat_summary['transaction_price'] * cat_summary['sum_costs']

recommended = cat_summary[
    cat_summary['profit_potential'] == cat_summary.groupby('buyer_id')['profit_potential'].transform(max)
]

recommended[['buyer_id', 'cat_id']].to_csv('processed__category_recs.csv', index=False)