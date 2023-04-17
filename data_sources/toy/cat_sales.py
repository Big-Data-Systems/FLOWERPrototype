import pandas as pd

"""
Calculates amount of money spent by customers on a given category
"""

sales = pd.read_csv("sales.csv")
product_info = pd.read_csv("processed__product_info.csv")


customer_categories = sales.merge(product_info[['item_id', 'cost', 'cat_id']], how='inner', left_on='product_id', right_on='item_id')


customer_categories['transaction_price'] = customer_categories['amount'] * customer_categories['cost']

cat_sums = customer_categories.groupby(['buyer_id', 'cat_id']).sum().reset_index()

cat_sums[['buyer_id', 'cat_id', 'amount', 'transaction_price']].to_csv("processed__category_sales.csv", index=False)