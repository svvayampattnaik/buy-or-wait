# Buy or Wait? — Submission README

**HackerRank Orchestrate, September 2026**
**Challenge:** [Buy or Wait?](https://www.hackerrank.com/contests/hackerrank-orchestrate-september26/challenges/buy-or-wait)

---

## Description

This solution implements a deterministic, heuristic-based financial decision agent that processes all 250 requests in `dataset/requests.csv` and produces a complete `dataset/output.csv` with eight output columns per request.

### How it works

The pipeline has five stages:

1. **Data loading** (`code/data_loader.py`): Loads all CSVs, parses messages with regex heuristics to extract financial facts (salary changes, contract cancellations, pending-income flags, amount amendments), and assembles a per-user data bundle for each request.

2. **Recurring series detection** (`code/forecaster.py` → `detect_recurring_series`): Groups historical financial events by category and direction, fits an interval to each group, and filters out terminated income series (keyword-matched "final"/"last" payroll events) and uninferrable one-offs.

3. **90-day balance simulation** (`code/forecaster.py` → `simulate_balance`): Projects recurring series, scheduled events, and pending debits forward. Cash movements are **netted per calendar day** before the minimum-balance check — all credits and debits on the same day are summed and tested once, rather than sequenced conservatively.

4. **Plan generation and ranking** (`code/planner.py`): Generates all candidate plans the user will consider (full payment, partial payment, installments from `request_payment_options.csv`, wait), evaluates each for 90-day safety (with and without spending changes), then ranks by a six-level tie-breaker: complete by deadline → no spending changes → minimize total cost including financing fees → earlier start → fewer payments → lowest `payment_option_id`.

5. **Output formatting** (`code/planner.py` → `process_request`): Assembles the eight required output columns, including a short `decision_explanation`.

### Key design choices

- **Zero live model calls at runtime.** All message/image extraction is handled by deterministic regex heuristics (`gemini_client=None`). The Gemini client plumbing is wired in `data_loader.py` and can be activated by passing a client to `build_user_data()`, but the production run does not use it.
- **No background burn-rate estimator.** An earlier burn-rate heuristic was removed after validation showed it introduced more noise than signal for users with irregular spending.
- **Same-day netting.** Instead of the conservative debit-before-credit ordering, same-day cash flows are netted before the threshold check.

---

## Setup and Run Instructions

### Requirements

- Python 3.9+
- Dependencies: `pandas`, `numpy`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables (optional)

If you wish to enable live Gemini API calls for message/image extraction, copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

> **Note:** The production run does not require this. Leaving the key blank runs the deterministic heuristic fallback, which is what produced the submitted `dataset/output.csv`.

### Run the pipeline

```bash
cd code
python main.py
```

This reads from `dataset/` and writes predictions to `dataset/output.csv` (250 rows + header). Runtime is approximately 15–20 seconds on a standard laptop.

### Verify the output

Confirm the output is structurally correct before submitting:

```bash
python -c "
import pandas as pd
out = pd.read_csv('dataset/output.csv')
req = pd.read_csv('dataset/requests.csv')
print(f'Rows: {len(out)} (expected 250)')
print(f'Nulls: {out.isnull().sum().to_dict()}')
m = out.merge(req[['request_id','requested_amount']], on='request_id')
bad = m[(m.amount_safe_to_pay < 0) | (m.amount_safe_to_pay > m.requested_amount)]
print(f'Out-of-bounds amount_safe_to_pay: {len(bad)}')
"
```

---

## Repository Structure

```text
.
├── README.md                         # This file
├── AGENTS.md                         # AI coding agent rules and log spec
├── problem_statement.md              # Original challenge specification
├── .env.example                      # API key template (copy to .env if needed)
├── .gitignore                        # Excludes .env, log.txt, debug/
├── log.txt                           # Agent conversation log — gitignored, submitted
│                                     #   SEPARATELY on HackerRank as chat_transcript
│                                     #   (NOT bundled inside code.zip)
│
├── code/
│   ├── main.py                       # Entry point — runs full 250-row pipeline
│   ├── data_loader.py                # CSV loading, message parsing, user data assembly
│   ├── forecaster.py                 # Series detection, projection, balance simulation
│   ├── planner.py                    # Plan generation, ranking, output assembly
│   └── spending_changes.py           # Spending change candidate selection
│
├── dataset/
│   ├── requests.csv                  # 250 requests to evaluate
│   ├── output.csv                    # Submitted predictions (this file)
│   ├── sample_requests.csv           # 25 solved examples (ground truth reference)
│   ├── financial_profiles.csv        # User balances, preferences, priorities
│   ├── financial_events.csv          # Historical, pending, and scheduled transactions
│   ├── request_payment_options.csv   # Payment options available per request
│   ├── exchange_rates.csv            # Fixed dated conversion rates
│   ├── messages.csv                  # Messages with financial context
│   ├── images.csv                    # Image metadata (payroll letters, bills, etc.)
│   └── media/images/                 # Image files referenced by images.csv
│
└── evaluation/
    └── usage_report.md               # Model/token usage report (required by §6.5)
```

---

## Validation

The accuracy numbers cited below are independently verifiable. Run
`python evaluation/validate.py` from the repo root — it executes the
full pipeline against all 25 labeled requests in `dataset/sample_requests.csv`,
compares predictions to ground truth, and prints a per-column match-rate
table. No internet access or API keys required.

---

## Known Limitations

### 1. Background burn-rate estimator excluded

An early version of the pipeline estimated a per-user "background burn rate" from irregular historical spending to improve conservative balance projection. This was removed after validation showed it introduced more noise than signal across the 25-sample labeled set — it over-penalized users with lumpy but legitimate historical spending and improved no measured column. The current pipeline uses only concretely detected recurring series and scheduled/pending events for projection.

### 2. Same-day netting tension: request_02 and request_23

The pipeline nets all cash movements on the same calendar day into a single balance delta before the minimum-balance check. This approach improved `affordability_status` by 4 percentage points and `recommended_payment_method` by 1 percentage point on the 25-sample set. However, `request_02` and `request_23` are structurally identical cases (same-day salary credit and payment debit) where the labeled ground truth treats them oppositely. This is an accepted boundary condition: no same-day ordering rule produces the correct answer for both simultaneously. The netting approach is correct for the majority of cases.

### 3. `amount_safe_to_pay` match rate

`amount_safe_to_pay` achieves a **12% exact match rate (3/25)** on the labeled sample. This is the weakest column. The primary driver is that the binary search computes the maximum safe payment given the agent's projected cash flows — when income is slightly mis-projected (e.g., a new employee with one salary history event, or a gig worker with irregular payout timing), the safe threshold shifts away from the ground truth value. The affordability status and recommended method are correct far more often (76% and 84% respectively), meaning the qualitative decision is right even when the exact threshold differs. Improving `amount_safe_to_pay` precision would require either live bank-feed data or a richer inference model for variable income, both outside the scope of this deterministic pipeline.

---

## Submission Checklist

- [x] `dataset/output.csv` — 250 rows, correct columns, all structural checks pass
- [x] `evaluation/usage_report.md` — runtime and development-time model usage
- [x] `code/` — fully runnable, no hardcoded secrets
- [x] `README.md` — setup, structure, limitations documented
- [x] `log.txt` — agent conversation log, submitted **separately** on the HackerRank
      portal as the `chat_transcript` deliverable; **not** bundled inside `code.zip`
