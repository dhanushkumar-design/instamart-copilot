
"""Part A: Seven SQL analyses with a Pandas cross-check."""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from .foundation import BASE, DB


QUERIES = {
    "01_completed_orders": """
        SELECT
            COUNT(*) AS completed_orders,
            ROUND(SUM(order_amount), 2) AS total_revenue
        FROM orders
        WHERE status = 'completed'
    """,

    "02_city_revenue": """
        SELECT
            c.city,
            COUNT(o.order_id) AS completed_orders,
            ROUND(SUM(o.order_amount), 2) AS revenue
        FROM orders o
        JOIN customers c
            ON o.customer_id = c.customer_id
        WHERE o.status = 'completed'
        GROUP BY c.city
        ORDER BY revenue DESC
    """,

    "03_top_products": """
        SELECT
            p.product_id,
            p.product_name,
            COUNT(o.order_id) AS completed_orders,
            ROUND(SUM(o.order_amount), 2) AS revenue
        FROM orders o
        JOIN products p
            ON o.product_id = p.product_id
        WHERE o.status = 'completed'
        GROUP BY p.product_id, p.product_name
        ORDER BY revenue DESC
        LIMIT 10
    """,

    "04_customer_order_summary": """
        SELECT
            c.customer_id,
            c.city,
            COUNT(o.order_id) AS completed_orders,
            ROUND(COALESCE(SUM(o.order_amount), 0), 2) AS revenue
        FROM customers c
        LEFT JOIN orders o
            ON c.customer_id = o.customer_id
            AND o.status = 'completed'
        GROUP BY c.customer_id, c.city
        ORDER BY revenue DESC
        LIMIT 20
    """,

    "05_monthly_revenue_cte": """
        WITH monthly_sales AS (
            SELECT
                strftime('%Y-%m', order_ts) AS month,
                COUNT(*) AS completed_orders,
                SUM(order_amount) AS revenue
            FROM orders
            WHERE status = 'completed'
            GROUP BY strftime('%Y-%m', order_ts)
        )
        SELECT
            month,
            completed_orders,
            ROUND(revenue, 2) AS revenue
        FROM monthly_sales
        ORDER BY month
    """,

    "06_product_revenue_rank": """
        WITH product_sales AS (
            SELECT
                p.product_id,
                p.product_name,
                SUM(o.order_amount) AS revenue
            FROM products p
            JOIN orders o
                ON p.product_id = o.product_id
            WHERE o.status = 'completed'
            GROUP BY p.product_id, p.product_name
        )
        SELECT
            product_id,
            product_name,
            ROUND(revenue, 2) AS revenue,
            DENSE_RANK() OVER (
                ORDER BY revenue DESC
            ) AS revenue_rank
        FROM product_sales
        ORDER BY revenue_rank
    """,

    "07_customer_order_sequence": """
        SELECT
            customer_id,
            order_id,
            order_ts,
            ROW_NUMBER() OVER (
                PARTITION BY customer_id
                ORDER BY order_ts, order_id
            ) AS order_number
        FROM orders
        WHERE status = 'completed'
        ORDER BY customer_id, order_number
    """,
}


def run():
    reports = BASE / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    results = {}

    with sqlite3.connect(DB) as con:
        for name, sql in QUERIES.items():
            frame = pd.read_sql_query(sql, con)

            frame.to_csv(
                reports / f"sql_{name}.csv",
                index=False
            )

            results[name] = {
                "rows": len(frame),
                "columns": list(frame.columns),
            }

        # Cross-check SQL revenue against Pandas.
        orders = pd.read_csv(
            BASE / "data" / "clean" / "orders.csv"
        )

        completed = orders[
            orders["status"] == "completed"
        ].copy()

        completed["order_amount"] = pd.to_numeric(
            completed["order_amount"],
            errors="coerce"
        )

        pandas_count = len(completed)
        pandas_revenue = round(
            float(completed["order_amount"].sum()), 2
        )

        sql_result = pd.read_sql_query(
            QUERIES["01_completed_orders"], con
        ).iloc[0]

        sql_count = int(
            sql_result["completed_orders"]
        )

        sql_revenue = round(
            float(sql_result["total_revenue"]), 2
        )

    cross_check = {
        "sql_completed_orders": sql_count,
        "pandas_completed_orders": pandas_count,
        "sql_revenue": sql_revenue,
        "pandas_revenue": pandas_revenue,
        "count_match": sql_count == pandas_count,
        "revenue_match": abs(
            sql_revenue - pandas_revenue
        ) < 0.01,
    }

    results["pandas_cross_check"] = cross_check

    (reports / "sql_analysis_summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
