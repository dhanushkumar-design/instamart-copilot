
import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "reports"
DB = BASE / "data" / "instamart.db"

st.set_page_config(
    page_title="Instamart Copilot",
    page_icon="🛒",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
h1, h2, h3 {letter-spacing: -0.5px;}
[data-testid="stMetric"] {
    background: #172238;
    border: 1px solid #34445e;
    border-radius: 14px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


def load_csv(filename):
    path = REPORTS / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_json(filename):
    path = REPORTS / filename
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def query_db(sql):
    if not DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB) as connection:
        return pd.read_sql_query(sql, connection)


st.sidebar.title("🛒 Instamart Copilot")
st.sidebar.caption("Customer Intelligence & Support")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Sales Analytics",
        "👥 Customer Segments",
        "🤖 Policy Assistant",
        "💳 Refund Agent",
    ],
)

st.sidebar.divider()
st.sidebar.info("Offline Python + SQL + ML + RAG Dashboard")

if page == "🏠 Overview":
    st.title("🛒 Instamart Intelligence Dashboard")
    st.caption("Project performance overview using generated data")

    summary = load_json("sql_analysis_summary.json")
    model = load_json("model_metrics.json")

    cross = summary.get("pandas_cross_check", {})

    customers = query_db("SELECT COUNT(*) AS n FROM customers")
    customer_count = (
        int(customers.iloc[0]["n"])
        if not customers.empty else 0
    )

    a, b, c, d = st.columns(4)

    a.metric("Customers", f"{customer_count:,}")
    b.metric(
        "Completed Orders",
        f"{cross.get('sql_completed_orders', 0):,}",
    )
    c.metric(
        "Total Revenue",
        f"₹{cross.get('sql_revenue', 0):,.2f}",
    )
    d.metric("ML F1 Score", model.get("f1", "N/A"))

    st.subheader("📈 Monthly Revenue")

    monthly = load_csv("sql_05_monthly_revenue_cte.csv")

    if not monthly.empty:
        fig = px.line(
            monthly,
            x="month",
            y="revenue",
            markers=True,
            title="Monthly Completed-Order Revenue",
        )
        st.plotly_chart(fig, use_container_width="stretch")
    else:
        st.warning("Monthly revenue report not found.")

    st.subheader("Project Modules")

    st.markdown("""
    - **Part A:** Data Cleaning and SQL Analytics
    - **Part B:** Customer Segmentation and Machine Learning
    - **Part C:** Support Ticket Classification
    - **Part D:** Policy RAG and Refund Agent
    """)

elif page == "📊 Sales Analytics":
    st.title("📊 Sales Analytics")

    city = load_csv("sql_02_city_revenue.csv")
    products = load_csv("sql_03_top_products.csv")

    if not city.empty:
        selected_cities = st.multiselect(
            "Select Cities",
            options=city["city"].tolist(),
            default=city["city"].tolist(),
        )

        filtered = city[city["city"].isin(selected_cities)]

        fig = px.bar(
            filtered,
            x="city",
            y="revenue",
            color="city",
            title="Revenue by City",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(filtered, use_container_width=True)

    if not products.empty:
        st.subheader("🏆 Top 10 Products")

        fig = px.bar(
            products,
            x="product_name",
            y="revenue",
            color="revenue",
            title="Top Products by Revenue",
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation & ML")

    segments = load_csv("customer_segments.csv")
    model = load_json("model_metrics.json")

    if not segments.empty:
        counts = (
            segments["segment"]
            .value_counts()
            .reset_index()
        )
        counts.columns = ["segment", "customers"]

        left, right = st.columns(2)

        with left:
            fig = px.pie(
                counts,
                names="segment",
                values="customers",
                hole=0.5,
                title="Customer Segment Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.scatter(
                segments,
                x="pca_1",
                y="pca_2",
                color="segment",
                hover_data=["customer_id"],
                title="PCA Customer Segmentation",
            )
            st.plotly_chart(fig, use_container_width=True)

        selected = st.selectbox(
            "Filter Customer Segment",
            ["All"] + sorted(segments["segment"].unique().tolist()),
        )

        visible = (
            segments if selected == "All"
            else segments[segments["segment"] == selected]
        )

        st.dataframe(visible.head(100), use_container_width=True)

    st.subheader("Machine Learning Results")

    cols = st.columns(4)
    for col, key, label in zip(
        cols,
        ["accuracy", "precision", "recall", "roc_auc"],
        ["Accuracy", "Precision", "Recall", "ROC-AUC"],
    ):
        col.metric(label, model.get(key, "N/A"))

    st.write("Selected Model:", model.get("selected_model", "N/A"))

elif page == "🤖 Policy Assistant":
    st.title("🤖 RAG Policy Assistant")
    st.caption("Ask questions about the synthetic Instamart policies")

    question = st.text_input(
        "Your Question",
        placeholder="What is the refund policy for late delivery?",
    )

    if st.button("Ask Policy Assistant", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                from src.policy import PolicyAssistant

                assistant = PolicyAssistant()
                result = assistant.answer(question.strip())

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Policy Sources")
                st.write(
                    ", ".join(result["sources"])
                    if result["sources"]
                    else "No matching policy source"
                )

                st.caption(
                    f"Mock LLM tokens: {result.get('tokens', 0)}"
                )

            except Exception as error:
                st.error(f"Policy assistant error: {error}")

    st.info(
        "This uses the supplied offline mock LLM, "
        "not an external AI API."
    )

elif page == "💳 Refund Agent":
    st.title("💳 Refund Agent Evaluation")
    st.caption("Read-only results from simulated refund scenarios")

    results = load_csv("agent_eval_results.csv")

    if not results.empty:
        passed = int(results["correct"].astype(str).str.lower().eq("true").sum())

        a, b, c = st.columns(3)
        a.metric("Scenarios", len(results))
        b.metric("Passed", passed)
        c.metric("Total LLM Calls", int(results["llm_calls"].sum()))

        st.subheader("Scenario Results")
        st.dataframe(results, use_container_width=True)

        fig = px.bar(
            results,
            x="scenario_id",
            y="tokens_used",
            color="outcome",
            title="Tokens Used by Refund Scenario",
        )
        st.plotly_chart(fig, use_container_width=True)

        scenario = st.selectbox(
            "Inspect Scenario",
            results["scenario_id"].tolist(),
        )

        selected = results[
            results["scenario_id"] == scenario
        ].iloc[0]

        st.write("**Outcome:**", selected["outcome"])
        st.write("**LLM Calls:**", int(selected["llm_calls"]))
        st.write("**Tokens Used:**", int(selected["tokens_used"]))

    st.warning(
        "Refund processing is simulated. "
        "This page does not execute refund tools."
    )
