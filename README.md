# Instamart Customer Intelligence & Support Copilot

An offline Python capstone project covering data cleaning, SQL analytics, customer segmentation, machine learning, support ticket classification, policy retrieval, and a simulated refund agent.

All customer records and transactions are synthetic. No real payment is processed.

## Project Features

### Part A: Data Cleaning and SQL Analytics
- Clean and normalize customer and order data.
- Remove duplicate orders and identify invalid values.
- Store cleaned data in SQLite.
- Execute seven SQL analysis queries using JOINs, CTEs, and window functions.
- Validate completed-order counts and revenue using Pandas.

### Part B: Customer Segmentation and Lapse Prediction
- Calculate Recency, Frequency, and Monetary (RFM) features.
- Segment 1,200 customers into four groups using KMeans.
- Apply PCA for two-dimensional customer visualization.
- Train Logistic Regression and Random Forest models.
- Tune hyperparameters using GridSearchCV.
- Select a classification threshold using validation data.
- Predict customers with no completed orders during September 2025.

**Model evaluation results:**

| Metric | Result |
|---|---|
| Selected model | Logistic Regression |
| F1 Score | 0.7250 |
| Accuracy | 0.7250 |
| Precision | 0.5959 |
| Recall | 0.9255 |
| ROC-AUC | 0.7617 |
| Selected threshold | 0.40 |

These metrics are specific to the generated dataset and held-out evaluation split.

### Part C: Support Ticket Classification
- Classify synthetic customer support tickets using the supplied offline mock LLM.
- Redact personally identifiable information.
- Handle transient errors and malformed JSON.
- Apply fallback classification rules.
- Record classification metrics and token usage.

### Part D: Policy Retrieval and Refund Agent
- Retrieve relevant policy information using TF-IDF.
- Evaluate policy answers against 14 supplied questions.
- Simulate refund decisions using bounded agent tool calls.
- Evaluate seven refund scenarios.
- Record agent outcomes without processing real payments.

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, SQLite, and the Python standard library.

## Project Structure

```text
instamart_copilot/
├── data/
├── reports/
├── src/
│   ├── foundation.py
│   ├── sql_analysis.py
│   ├── segments.py
│   ├── triage.py
│   ├── policy.py
│   └── agent.py
├── tests/
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── run_project.py
└── README.md
```

## Installation

Requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
```

## Run the Complete Project

```bash
python run_project.py
```

The pipeline generates synthetic data, builds the SQLite database, performs SQL analytics, trains and evaluates ML models, classifies tickets, evaluates policy retrieval, and tests refund-agent scenarios.

## Reports

Generated outputs are saved in the `reports/` directory, including:

- `audit.json`
- `sql_analysis_summary.json`
- `sql_*.csv`
- `customer_segments.csv`
- `model_metrics.json`
- `ticket_predictions.csv`
- `triage_metrics.json`
- `policy_eval_results.csv`
- `agent_eval_results.csv`

## Evaluation Summary

| Component | Observed Result |
|---|---|
| SQL queries | 7 executed |
| SQL and Pandas cross-check | Matched |
| Customer clusters | 4 |
| Lapse prediction F1 | 0.725 |
| Ticket classification accuracy | 1.00 |
| Policy fact-match rate | 1.00 on 14 questions |
| Refund agent pass rate | 7/7 scenarios |

## Safety and Limitations

- `generate_data.py` and `mock_llm.py` are supplied reference components and should remain unchanged.
- All data is fictional.
- Invalid monetary values are not fabricated.
- A lapse prediction is a proxy for customer inactivity, not a guarantee of future churn.
- The refund agent is a simulation, not a real payment system.
- The seven-day refund deadline cannot be fully verified from the available timestamps.
- Evaluation scores are dataset-specific and do not establish production readiness.