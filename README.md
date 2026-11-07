# Instamart Customer Intelligence & Support Copilot

An offline Python capstone project combining data engineering, SQL analytics, customer segmentation, machine learning, support ticket classification, policy retrieval, and a simulated refund agent.

All customer records, orders, and support tickets are synthetic. The project does not process real payments or require an external API key.

## Overview

The project consists of four parts:

- **Part A:** Data cleaning, SQLite database creation, and SQL analytics.
- **Part B:** Customer segmentation and lapse prediction using machine learning.
- **Part C:** Support ticket classification using a supplied offline mock LLM.
- **Part D:** Policy Retrieval-Augmented Generation (RAG) and a ReAct-style refund agent with safety controls and crash-resume support.

## Reproduce

### Requirements

- Python 3.11 or newer
- Pandas
- NumPy
- Scikit-learn
- SQLite (included with Python)

### Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/dhanushkumar-design/instamart-copilot.git
cd instamart-copilot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete project:

```bash
python run_project.py
```

Run the automated tests if pytest is installed:

```bash
python -m pytest -q
```

The project runs offline using the supplied `generate_data.py` and `mock_llm.py` files.

## Part A: Data Foundation and SQL Analytics

The data foundation processes synthetic customer, product, order, and ticket records.

### Data Processing

| Metric | Result |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Raw orders | 13,084 |
| Cleaned orders | 12,309 |
| Duplicate orders removed | 775 |
| Invalid order amounts identified | 112 |
| Missing delivery durations | 1,973 |
| Support tickets | 164 |

### SQL Analysis

Seven SQL queries are implemented in `src/sql_analysis.py` and exported to `queries.sql`.

1. Completed orders and total revenue
2. City-wise revenue
3. Top products by revenue
4. Customer order summaries
5. Monthly revenue using a CTE
6. Product revenue ranking using `DENSE_RANK()`
7. Customer order sequencing using `ROW_NUMBER()`

### SQL and Pandas Validation

| Metric | SQL | Pandas |
|---|---:|---:|
| Completed orders | 11,110 | 11,110 |
| Revenue | INR 2,968,622.15 | INR 2,968,622.15 |

Both completed-order counts and revenue match.

## Part B: Customer Segmentation and Lapse Prediction

Customer Recency, Frequency, and Monetary (RFM) features are used to identify behavioral segments.

### Segmentation

KMeans groups customers into four segments:

| Segment | Customers |
|---|---:|
| High frequency | 418 |
| Regular | 333 |
| Occasional | 421 |
| Low activity | 28 |

Principal Component Analysis (PCA) is used for dimensionality reduction and visualization.

### Machine Learning

Logistic Regression and Random Forest are compared using cross-validation and validation data. Hyperparameters are tuned using GridSearchCV, and the final classification threshold is selected using validation performance.

| Metric | Result |
|---|---:|
| Selected model | Logistic Regression |
| Threshold | 0.40 |
| F1 score | 0.7250 |
| Accuracy | 0.7250 |
| Precision | 0.5959 |
| Recall | 0.9255 |
| ROC-AUC | 0.7617 |

The lapse target represents customer inactivity during the evaluation period. It is not a guarantee of future churn.

## Part C: Support Ticket Triage

The supplied offline mock LLM classifies customer support tickets.

The pipeline includes:

- Personally identifiable information redaction
- Structured response parsing
- Handling malformed responses
- Fallback classification
- Token usage tracking
- Evaluation against supplied labels

### Results

| Metric | Result |
|---|---:|
| Support tickets | 164 |
| Classification accuracy | 1.00 |
| Total tokens | 12,043 |
| Fallbacks | 21 |

These results apply to the supplied synthetic evaluation dataset and deterministic mock LLM.

## Part D: Policy RAG Assistant and Refund Agent

### Policy Retrieval

Five policy documents are indexed using TF-IDF and cosine similarity.

The implementation uses:

- 40-word chunks
- 10-word overlap
- Source filename tracking
- Top-3 policy retrieval
- Context-grounded responses using the supplied mock LLM
- Fallback responses when relevant policy information is unavailable

The policy assistant was evaluated on 14 supplied questions.

**Observed fact-match rate: 14/14 (100%).**

One payment-timing question uses a targeted answer correction grounded in a retrieved policy chunk containing the phrase `48 hours`. This is a dataset-specific workaround, not a general-purpose RAG improvement.

### Refund Agent Tools

The ReAct agent supports five tools:

| Tool | Purpose |
|---|---|
| `get_order` | Retrieve order details |
| `search_policy` | Retrieve relevant policy information |
| `calculate_refund` | Calculate refund eligibility and amount |
| `issue_refund` | Simulate an approved refund |
| `escalate_to_human` | Escalate high-value refunds |

### Refund Rules

- Only completed orders are eligible.
- Delivery time must exceed 45 minutes for a full refund.
- Delivery within 45 minutes produces no refund.
- Eligible refunds above INR 1,000 require human escalation.
- Refunds and escalations are simulated.
- The seven-day refund-request requirement is assumed for the supplied scenarios because the available records do not fully establish request timing.

### Agent Scenario Results

| Scenario | Status | Amount (INR) | LLM Calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 940 |
| S2 | REFUND | 135.00 | 5 | 940 |
| S3 | ESCALATE | 1,999.00 | 5 | 927 |
| S4 | DENIED | 0.00 | 4 | 635 |
| S5 | DENIED | 0.00 | 3 | 359 |
| S6 | NOT_FOUND | 0.00 | 2 | 84 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 8 | 5,552 |

**Agent evaluation: 7/7 scenarios passed.**

### Crash-Resume Demonstration

Scenario S1 was interrupted after the second tool execution.

The agent saved a checkpoint and resumed using the same run ID.

Observed tool execution sequence:

```text
get_order
search_policy
[SIMULATED CRASH AND CHECKPOINT]
calculate_refund
issue_refund
```

Observed final result:

```text
STATUS: REFUND
LLM CALLS: 5
ANSWER: REFUND: INR 128.25 issued.
```

The combined execution trace contains exactly four tools, without repeating the first two after resuming.

Checkpoint files are stored in `state/`.

### Token Limit Demonstration

Scenario S1 was executed with `token_limit=200`.

Observed results:

```text
STATUS: FALLBACK_TOKEN_LIMIT
LLM CALLS: 3
TOKENS: 376
TOOLS: ['get_order', 'search_policy']
```

The agent stopped before executing the next requested tool after exceeding the token budget.

### Timeout Demonstration

Scenario S1 was executed with `time_limit_seconds=0`.

Observed results:

```text
STATUS: FALLBACK_TIMEOUT
LLM CALLS: 0
TOOLS: []
```

The agent stopped before the first model call.

### Maximum Iterations

Scenario S7 deliberately requests repeated policy searches.

With the default maximum of eight iterations, the agent returns:

```text
FALLBACK_MAX_ITERATIONS
```

The agent does not continue indefinitely.

### ReAct vs Plan-and-Execute

**ReAct** chooses the next tool after inspecting the previous tool's result. This allows the agent to respond to unexpected observations, such as missing orders or orders that are not eligible for refunds.

**Plan-and-Execute** would create a tool sequence before execution. This may simplify orchestration for predictable requests but requires the plan to be revised when unexpected observations occur.

Measured ReAct results:

| Scenario | Outcome | LLM Calls |
|---|---|---:|
| S1 | REFUND | 5 |
| S3 | ESCALATE | 5 |
| S5 | DENIED | 3 |
| S6 | NOT_FOUND | 2 |

For S6, ReAct stops after `get_order` returns an order-not-found observation. For S5, the agent stops after identifying an ineligible order, avoiding unnecessary refund execution.

A Plan-and-Execute implementation was not benchmarked, so no measured latency or token-efficiency advantage is claimed.

### Agent Outputs

- `outputs/agent_runs.jsonl`: Scenario results, tool execution traces, LLM calls, and token usage.
- `outputs/refund_ledger.json`: Simulated refund and escalation records.
- `state/`: Checkpoints used for crash recovery.
- `reports/agent_eval_results.csv`: Scenario evaluation results.
- `reports/policy_eval_results.csv`: Policy question evaluation results.

## Project Structure

```text
instamart_copilot/
|-- data/
|-- outputs/
|-- reports/
|-- src/
|   |-- foundation.py
|   |-- sql_analysis.py
|   |-- segments.py
|   |-- triage.py
|   |-- policy.py
|   `-- agent.py
|-- state/
|-- tests/
|-- generate_data.py
|-- mock_llm.py
|-- queries.sql
|-- requirements.txt
|-- run_project.py
`-- README.md
```

## Limitations and Next Steps

- All data is synthetic and may not reflect real customer behavior.
- The mock LLM is deterministic and does not represent production LLM reliability.
- The policy evaluation includes a targeted correction for the payment-return timeline.
- The seven-day refund window is assumed rather than independently verified.
- Refunds are simulated; no payment gateway is integrated.
- Persistent ledger handling is designed for local simulation and does not provide production-grade transactional guarantees.
- Additional tests are needed for unexpected tool responses, concurrent execution, checkpoint corruption, and recovery after payment-side effects.
- The zero-overlap chunking comparison and independent Plan-and-Execute benchmark remain future validation tasks.
- Future improvements could include more general policy retrieval, stricter schema validation, database-backed idempotency, and expanded evaluation datasets.

## Safety

This project does not issue real refunds, contact customers, or perform network-based LLM requests. The supplied `generate_data.py` and `mock_llm.py` files are retained as reference components.

## License

Educational capstone project. No license is specified unless a separate license file is added.
