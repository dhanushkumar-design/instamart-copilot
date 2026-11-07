"""Part A: clean synthetic inputs, save audit and SQLite warehouse."""
from pathlib import Path
import json
import sqlite3
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'data/raw'
CLEAN = BASE / 'data/clean'
DB = BASE / 'data/instamart.db'

def build():
    CLEAN.mkdir(parents=True, exist_ok=True)
    customers = pd.read_csv(RAW/'customers.csv')
    products = pd.read_csv(RAW/'products.csv')
    orders = pd.read_csv(RAW/'orders.csv')
    tickets = pd.read_csv(RAW/'tickets.csv')
    audit = {'raw_rows': {'customers':len(customers),'products':len(products),'orders':len(orders),'tickets':len(tickets)}}
    customers['city'] = customers['city'].astype('string').str.strip().str.title()
    customers['gender'] = customers['gender'].astype('string').str.strip().str.lower().replace({'m':'male','f':'female'})
    customers['age'] = pd.to_numeric(customers['age'],errors='coerce')
    customers.loc[~customers['age'].between(18,100), 'age'] = np.nan
    customers['age'] = customers['age'].fillna(customers['age'].median())
    customers['gender'] = customers['gender'].fillna('unknown')
    customers['signup_date'] = pd.to_datetime(customers['signup_date'],errors='coerce')
    orders['order_ts'] = pd.to_datetime(orders['order_ts'],errors='coerce')
    orders['order_amount'] = pd.to_numeric(orders['order_amount'],errors='coerce')
    orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'],errors='coerce')
    orders['quantity'] = pd.to_numeric(orders['quantity'],errors='coerce')
    orders['discount_pct'] = pd.to_numeric(orders['discount_pct'],errors='coerce')
    orders = orders.sort_values(['order_id','order_ts','status'],kind='stable').drop_duplicates('order_id',keep='last').copy()
    audit['deduplicated_orders'] = len(orders)
    orders.loc[~orders['delivery_minutes'].between(0,75),'delivery_minutes'] = np.nan
    orders.loc[orders['status']!='completed','delivery_minutes'] = np.nan
    audit['invalid_amounts'] = int(orders['order_amount'].isna().sum())
    audit['missing_delivery_minutes'] = int(orders['delivery_minutes'].isna().sum())
    # Invalid amounts are not fabricated; retain NA and exclude them from monetary calculations.
    orders = orders[orders['customer_id'].isin(customers['customer_id']) & orders['product_id'].isin(products['product_id'])]
    for name,frame in [('customers',customers),('products',products),('orders',orders),('tickets',tickets)]:
        frame.to_csv(CLEAN/f'{name}.csv',index=False)
    with sqlite3.connect(DB) as con:
        for name,frame in [('customers',customers),('products',products),('orders',orders),('tickets',tickets)]:
            frame.to_sql(name,con,if_exists='replace',index=False)
        con.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_order ON orders(order_id)')
        con.execute('CREATE INDEX IF NOT EXISTS ix_order_customer ON orders(customer_id)')
    audit['clean_rows'] = {'customers':len(customers),'products':len(products),'orders':len(orders),'tickets':len(tickets)}
    (BASE/'reports').mkdir(exist_ok=True)
    (BASE/'reports/audit.json').write_text(json.dumps(audit,indent=2))
    return audit

if __name__=='__main__': print(json.dumps(build(),indent=2))
