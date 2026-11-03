# Instamart Customer Intelligence

## Overview

This project analyzes Instamart customer, order, product and support-ticket data. It covers data cleaning, SQLite analysis, customer segmentation, lapse prediction, ticket classification, PII redaction and a simulated refund agent.

All refund actions are simulated. No real payments or refunds are processed.

## Reproduce

Run all commands from the project root using the project virtual environment.

### 1. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Audit and clean raw data

```powershell
.\.venv\Scripts\python.exe data\clean\audit_data.py
.\.venv\Scripts\python.exe data\clean\clean_customers.py
.\.venv\Scripts\python.exe data\clean\clean_orders.py
.\.venv\Scripts\python.exe data\clean\clean_products.py
```

### 3. Build and validate the database

```powershell
.\.venv\Scripts\python.exe build_database.py
.\.venv\Scripts\python.exe validate_project.py
```

### 4. Run customer segmentation and lapse prediction

```powershell
.\.venv\Scripts\python.exe customer_segments.py
.\.venv\Scripts\python.exe lapse_model.py
```

### 5. Run ticket classification

The Part C script uses the pre-generated assignment input file: `outputs/customer_segments_assignment.csv`.

```powershell
.\.venv\Scripts\python.exe src\part_c_triage.py
```

### 6. Refund agent

Refund agent scenarios are not part of the routine reproduction sequence. They can append entries to the simulated refund ledger and run log. Use an isolated copy of the project when testing scenarios or crash/resume behaviour.

The refund agent uses a simulated ledger and does not process real payments.

## Part A — Data Cleaning and Database Preparation

### Raw Data Quality Audit

The raw datasets were inspected before cleaning.

| Data Quality Issue | Observed Count |
|---|---:|
| Raw orders | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Customers with blank age | 61 |
| Customers with age outside 10–100 | 6 |

### Data Cleaning Results

- Duplicate order rows were removed.
- Orders were reduced to one row per unique order ID.
- Pending orders were excluded from the cleaned orders dataset.
- Missing order amounts were handled during cleaning.
- Missing delivery times were retained for cancelled orders where applicable.
- Customer age and gender values were cleaned and standardized.

### Clean Dataset Summary

| Dataset | Clean Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |

The cleaned customers, products and orders datasets were loaded into SQLite database `data/instamart.db`.

### Customer Data Validation

- Customer age values range from 18 to 60.
- No missing customer ages remained.
- Gender categories: female 610, male 560, unknown 30.
- City distribution: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.

### Order Data Validation

- Completed orders: 11,110
- Cancelled orders: 1,199
- Pending orders: 0
- Missing order amounts: 0
- Completed-order revenue: INR 2,998,668.80

 ### Part A — SQL Analysis

Seven SQL queries were executed successfully against the SQLite database (`data/instamart.db`) using Python's `pandas.read_sql_query()`.

| Query | SQL Concept | Result |
|---|---|---:|
| Q1 | WHERE and HAVING | 30 customers |
| Q2 | Three-table JOIN | 30 city-category combinations |
| Q3 | Scalar subquery | 11 products |
| Q4 | LEFT JOIN | 25 customers without completed orders |
| Q5 | Common Table Expressions (CTEs) | 550 customers above average spend |
| Q6 | RANK, DENSE_RANK and ROW_NUMBER | 1,175 customer records |
| Q7 | Running total using a window function | 183 daily records |

### Key Findings

- Q1 identified 30 customers whose completed orders above INR 500 totalled more than INR 8,000.
- Q2 analysed completed-order revenue across six cities and product categories.
- Q3 identified 11 products priced above the average catalogue price.
- Q4 identified 25 customers with no completed orders.
- Q5 found 550 customers whose completed spending exceeded the average customer spend of INR 2,552.06.
- Q6 ranked customers by completed spending within each city using three ranking window functions.
- Q7 calculated daily completed revenue and its cumulative running total.

**Validation:** All seven SQL queries executed successfully using the submission database.

### Additional SQL Validation Results

- Q6: 8 customer records had different values for `RANK()` and `ROW_NUMBER()`, demonstrating how ranking functions handle tied spending values differently.
- Q7: The final cumulative completed-order revenue was INR 2,998,668.80 across 183 days.
- All seven SQL queries passed execution against the submission SQLite database.


### Statistical Analysis — Completed Orders

| Metric | Result |
|---|---:|
| Mean order amount | INR 269.91 |
| Median order amount | INR 135.00 |
| Skewness | 4.61 |
| First quartile (Q1) | INR 64.80 |
| Third quartile (Q3) | INR 297.00 |
| Interquartile range (IQR) | INR 232.20 |
| Upper outlier fence | INR 645.30 |
| Upper outliers | 897 |
| Lower outliers | 0 |
| Amount–discount correlation | -0.095 |
| Amount–quantity correlation | +0.087 |

The positive skewness indicates that most completed orders have relatively lower amounts, while a smaller number of high-value orders create a long right tail. Both correlations are weak; they do not establish a causal relationship.

## Part B — Customer Lapse Prediction

### Objective

The objective was to identify customers likely to lapse using customer transaction behaviour and machine learning models.

### Dataset and Target

- Cohort size: 1,156 customers
- Lapsed customers: 443 (38.32%)
- Non-lapsed customers: 713
- Label period ended: 2025-10-01
- Prediction cutoff: 2025-09-01

### RFM Analysis and Customer Segmentation

RFM analysis used 1,175 customers with valid completed-order history, using 2025-10-01 as the snapshot date.

- Recency: median 19 days, minimum 0, maximum 180.
- Frequency: mean 9.46 orders, maximum 33.
- Monetary value: mean INR 2,552.06, maximum INR 14,766.55.

K-Means was evaluated for K=2 through K=8 using StandardScaler on the original recency, frequency, and monetary features without log transformation, with `random_state=42` and `n_init=10`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.7606 | 0.4650 |
| 3 | 1268.2346 | 0.4452 |
| 4 | 743.0711 | 0.5255 |
| 5 | 617.3666 | 0.4805 |
| 6 | 503.1706 | 0.4819 |
| 7 | 408.4605 | 0.4163 |
| 8 | 366.5035 | 0.4123 |

K=4 was selected from the permitted K=3, 4, and 5 choices because it had the highest silhouette score among those candidates.

| Segment | Customers | Average Monetary Value |
|---|---:|---:|
| Low Activity Customers | 182 | INR 685.99 |
| Occasional Customers | 436 | INR 1,057.62 |
| Regular Customers | 419 | INR 3,410.02 |
| Premium Customers | 138 | INR 7,129.70 |

Premium status was assigned to the segment with the highest average monetary value. Segment IDs are arbitrary cluster labels.

PCA explained variance:
- Component 1: 63.43%
- Component 2: 21.80%
- Combined: 85.23%

### Model Comparison

| Model | Cross-validation F1 | Test Accuracy | Lapsed Precision | Lapsed Recall | Lapsed F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6506 | 73% | 0.64 | 0.65 | 0.65 |
| Random Forest | 0.6811 | 74% | 0.66 | 0.65 | 0.66 |

The baseline model achieved 61.68% accuracy but had an F1 score of 0 for the lapsed class.

### Train/Test Split and Hyperparameter Tuning

The lapse dataset was split into 924 training rows and 232 test rows using stratification and `random_state=42`.

- Logistic Regression: GridSearchCV selected `C=100`; best cross-validation F1 was 0.6506.
- Random Forest: GridSearchCV selected `max_depth=8` and `min_samples_leaf=10`; best cross-validation F1 was 0.6811.
- Random Forest tuning evaluated 9 parameter combinations using 5-fold cross-validation (45 fits).

### Test Set Confusion Matrices

Logistic Regression:

| | Predicted not lapsed | Predicted lapsed |
|---|---:|---:|
| Actual not lapsed | 111 | 32 |
| Actual lapsed | 31 | 58 |

Random Forest:

| | Predicted not lapsed | Predicted lapsed |
|---|---:|---:|
| Actual not lapsed | 113 | 30 |
| Actual lapsed | 31 | 58 |

### Training-Only Oversampling

Oversampling was applied only to the training data. The test set remained unchanged.

| Model | Lapsed Precision | Lapsed Recall | Lapsed F1 |
|---|---:|---:|---:|
| Unbalanced Logistic Regression | 0.6444 | 0.6517 | 0.6480 |
| Oversampled Logistic Regression | 0.5891 | 0.8539 | 0.6972 |

Oversampling increased recall but reduced precision, showing the trade-off between identifying more lapsed customers and generating additional false positives.

### Preprocessing and Feature Interpretation

- Numeric features were imputed and standardized using training data only.
- Categorical features were imputed and one-hot encoded with one category dropped.
- Full cohort frequency mean: 8.0285.
- Training-only fitted scaler frequency mean: 8.1147.

Logistic Regression coefficient examples:
- Frequency: -1.6982
- Favorite category — Electronics: -1.1367
- Monetary value: +0.2846

Top Random Forest feature importances:
- Frequency: 0.2796
- Monetary value: 0.2712
- Recency: 0.1151
- Average delivery time: 0.0613
- Cancellation rate: 0.0523

Feature importance and coefficients describe model associations; they do not establish causation.

### Selected Model

Random Forest was selected based on its higher training-only cross-validation F1 score (0.6811 vs. 0.6506). The test set was used only to evaluate the selected model, not to choose between models.The most important features included:

- Purchase frequency
- Monetary value
- Recency
- Average delivery time
- Cancellation rate

### Model Validation and Outputs

- The trained model was saved as `outputs/lapse_model.joblib`.
- Reloaded model predictions matched the original predictions.
- `outputs/retention_list.csv` was generated with 10 customers.

### Limitations

The premium flag was based on RFM information from the full observation window, including September data. Therefore, the retention list is illustrative and should not be treated as a fully leakage-free production prediction.

## Part C — Ticket Classification and PII Redaction

### Objective

The objective was to classify customer support tickets, redact personally identifiable information (PII), and evaluate zero-shot and few-shot classification performance.

### Dataset Audit

- Total tickets: 164
- Empty tickets: 4
- Long tickets: 4
- Estimated tokens in long tickets: 734

### PII Redaction

The redaction process detected and replaced:

| PII Type | Replacements |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

No matching PII patterns remained after redaction.

### Retry and Error Handling

The mock LLM call uses a retry mechanism for transient errors.

- Maximum retries: 3 after the initial attempt.
- Backoff: exponential, starting at 0.05 seconds.
- Jitter: random multiplier between 0.9 and 1.15.
- Random seed: 42 for reproducible jitter.
- If all retries fail, the transient error is raised rather than silently ignored.

At the start of each pipeline run, the mock LLM state, transient error counter, and retry log are reset. This keeps retry statistics separate between runs.

### Zero-shot vs. Few-shot Evaluation

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.7% | 100% |
| Total tokens | 8,932 | 21,256 |
| Tokens per classified ticket | 60.8 | 144.6 |

The zero-shot run also recorded 12 keyword-based classifications and 9 invalid model outputs. The few-shot run achieved higher accuracy on this project dataset but used more tokens.

### Budget Evaluation

A budget demonstration using a 6,000-token limit produced:

- 90 JSON classifications
- 52 budget-exhausted results
- 8 keyword-based classifications
- 6 invalid model outputs
- 4 skipped empty tickets
- 4 flagged long tickets

The final token accounting was 5,975 out of the 6,000-token budget. The pipeline reserves an estimated 68 tokens before starting each new model call to avoid exceeding the configured budget.

All 164 tickets were processed in the final run.

| Route | Tickets |
|---|---:|
| Support queue | 59 |
| Premium desk | 54 |
| Security team | 34 |
| Human review | 17 |

Priority distribution:
- HIGH: 88
- NORMAL: 76

### Ticket Routing Rules

The final routing logic applies the following rules in order:

| Condition | Assigned Route | Priority |
|---|---|---|
| Empty ticket | Human review | NORMAL |
| Long ticket | Human review | NORMAL |
| Invalid model output | Human review | NORMAL |
| Budget exhausted | Human review | NORMAL |
| Fraud category | Security team | HIGH |
| Premium customer | Premium desk | HIGH |
| All other valid tickets | Support queue | NORMAL |

Fraud tickets are routed to the security team before checking premium status. Tickets requiring human review are not automatically assigned to another team.

### Output Files

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

### Limitations

The reported accuracy is based on the supplied project dataset and mock LLM evaluation. It should not be interpreted as guaranteed production performance.

## Part D — Refund Agent

### 1. Project Overview


This project implements a tool-using refund support agent for Instamart. The agent retrieves order information from SQLite, searches refund policies, calculates eligible refunds, issues eligible refunds, and escalates high-value refunds to a human.



The implementation uses Python, SQLite, a policy retriever, and the supplied `mock_llm.py`.



**Safety design:** The agent checks order status before processing a refund. Refunds above ₹1,000 are escalated to a human instead of being automatically issued. The system includes iteration, token, and time limits.



### 2. Scenario Evaluation


| Scenario | Expected outcome | Observed outcome | Result |
|---|---|---|---|
| S1 | REFUND ₹128.25 | REFUND ₹128.25 | PASS |
| S2 | REFUND ₹135.00 | REFUND ₹135.00 | PASS |
| S3 | ESCALATE ₹1,999.00 | ESCALATE | PASS |
| S4 | DENIED ₹0.00 | DENIED | PASS |
| S5 | DENIED ₹0.00 | DENIED | PASS |
| S6 | NOT_FOUND ₹0.00 | NOT_FOUND | PASS |
| S7 | FALLBACK_LOOP | FALLBACK_LOOP | PASS |


All seven scenarios produced the expected outcome categories.



### Scenario observations



- S1 and S2 received refunds because their orders qualified under the late-delivery rule.

- S3 was escalated because the refund amount exceeded ₹1,000.

- S4 was denied because the delivery was within the 45-minute threshold. The denial reason should be reviewed against the customer's stated complaint.

- S5 was denied because the order was not eligible for a refund.

- S6 returned NOT_FOUND because the order did not exist.

- S7 triggered the repeated-tool-call protection. The agent detected the repeated policy-search request, prevented its re-execution, and returned FALLBACK_LOOP after 2 LLM calls and 1 tool execution.



### 3. Safety and Fallback Tests


| Test | Configuration | Observed result |
|---|---|---|
| Token limit | `token_limit=1` | `FALLBACK_TOKEN_LIMIT` |
| Timeout | `time_limit_seconds=0` | `FALLBACK_TIMEOUT` |
| Maximum iterations | `max_iterations=1` | `FALLBACK_MAX_ITERATIONS` |


The timeout test returned `FALLBACK_TIMEOUT` with zero LLM calls and no tool executions.

The token-limit test returned `FALLBACK_TOKEN_LIMIT` with a limit of 200 tokens. It used 430 tokens across three LLM calls and executed two tools: `get_order` and `search_policy`. No refund was issued.

The maximum-iteration test returned `FALLBACK_MAX_ITERATIONS` after three LLM calls and three `search_policy` tool executions.

All three safety tests passed their expected results.

### 4. Crash, Resume and Loop Protection Tests

Crash recovery was tested using a dedicated checkpoint ID and an isolated temporary refund ledger. The main refund ledger was not modified.

**S7 loop-protection test:**
- The mock LLM returned the same `search_policy` call repeatedly.
- The first tool execution was recorded.
- The repeated call was detected before a second execution.
- The agent returned `FALLBACK_LOOP` after 2 LLM calls and 1 tool execution.
- The isolated S7 checkpoint records one completed tool execution. The final status was verified from the test output.

**S1 normal workflow crash-and-resume test:**
- Simulated a crash immediately after the first tool execution (`get_order`).
- Verified that a checkpoint was created.
- Resumed the agent from the saved checkpoint.
- The remaining tools executed: `search_policy`, `calculate_refund`, and `issue_refund`.
- The resumed workflow returned `REFUND: INR 128.25 issued.`
- Used a separate test ledger, leaving the main refund ledger untouched.
- Verified that exactly one refund entry was created for ORD-100048, amount INR 128.25.

**Duplicate-action protection:**
- Repeated refund and escalation attempts for already-processed orders returned `already_processed`.
- The isolated ledger contained 3 entries before and after the duplicate attempts.
- No additional ledger entries were created.

**Checkpoint integrity:** The inspected checkpoints contained run IDs, scenario IDs, conversation messages, token counts, completed tool steps, arguments and tool observations. The checkpoint format does not store the final status; statuses above were verified from test output.

**Results:** The tested loop-protection, checkpoint creation, resume operation and duplicate-action checks passed.

These tests verify the tested recovery paths; they do not establish crash safety under concurrent execution or every possible failure condition. The ledger duplicate check and append are separate operations, so concurrent requests could still require additional protection.
### 5. Run Logging

The file `outputs/agent_runs.jsonl` contains historical scenario logs for S1 through S7. These entries reflect the runs recorded at that time and should not be treated as the latest regression results.

The latest regression tests were run with isolated artifacts:
- Refund ledger: `outputs/regression_fix_20261103/`
- Checkpoints: `state/regression_fix_20261103/`
- S7 loop-fix checkpoint: `state/S7_LOOP_FIX_TEST_20261103.json`

The isolated regression tests left the main refund ledger untouched.

**Important:** `evaluate_agent.py` can append entries to the refund ledger and overwrite evaluation checkpoints. Do not run it in this preserved working project. To reproduce the full evaluation, first create a separate isolated copy of the project and use separate ledger and checkpoint paths.

Checkpoint files are stored in the `state/` directory.

### 6. ReAct vs. Plan-and-Execute

The implemented refund agent follows the ReAct pattern. It alternates between reasoning, tool execution, and observations. Each tool result can influence the next action.

Plan-and-Execute creates a sequence of steps before executing them. This can reduce repeated planning calls, but unexpected tool results may require replanning.

| Aspect | ReAct | Plan-and-Execute |
|---|---|---|
| Workflow | Reasons and acts iteratively | Plans before execution |
| Adaptability | Responds to each observation | May require replanning |
| Limitation | Repeated reasoning can increase calls | Initial plans can become outdated |

In the recorded offline S1 comparison, ReAct used 738 estimated mock tokens across 4 tool calls. Its exact LLM call count was not recorded. Plan-and-Execute used 2 mock LLM calls and 41 estimated tokens across 4 tool calls. Both produced the same INR 510 refund outcome.

These measurements use the mock LLM's word-based token estimation. They are prototype results, not production performance benchmarks.

### Zero-Overlap RAG Evaluation

The policy assistant was evaluated with chunk overlap temporarily set to 0.

- Answerable questions: 12
- Expected document retrieved: 12/12
- Expected fact found in answer: 10/12
- Unanswerable questions with correct fallback: 2/2

Two answerable questions missed the expected fact:
1. Full-refund eligibility for delivery time: expected "more than 45 minutes".
2. Failed payment deducted amount return time: expected "48 hours".

The original overlap setting was restored to 10 after evaluation.

## Limitations and Next Steps

- The datasets are supplied project data; results should not be treated as live Instamart business metrics.
- The Part B premium-customer retention list is illustrative because premium membership uses the full observation window, including September data.
- Part C uses a mock LLM implementation. Its classification and token results are not production language-model measurements.
- customer_segments_assignment.csv is a pre-generated Part C input and is not recreated by customer_segments.py.
- The refund agent uses a simulated ledger. It does not issue real refunds or connect to a payment provider.
- Future improvements include leakage-free time-based evaluation, documenting the Part C input-generation pipeline, and testing refund-ledger concurrency and persistence safeguards.
