# Usage Report — Buy or Wait? (HackerRank Orchestrate, September 2026)

Generated: 2026-09-13

---

## SECTION 1 — Runtime Model Usage: Final Production Run

**This section satisfies AGENTS.md §6.5.**

### Result: 0 live model calls

The production pipeline (`code/main.py`) calls `build_user_data()` without a
`gemini_client` argument (line 24). The default parameter is `gemini_client=None`,
which activates the deterministic heuristic fallback in all three model-capable
functions inside `data_loader.py`:

| Function | With client | Without client (`None`) |
|---|---|---|
| `_extract_amount_from_image()` | Calls Gemini Vision | Returns `None` (image skipped) |
| `_apply_linked_message()` | Calls Gemini text gen | Uses regex heuristics |
| `_extract_financial_facts()` | Calls Gemini text gen | Uses regex heuristics |

### Token and cost summary

| Metric | Value |
|---|---|
| Total live model calls | **0** |
| Input tokens | **0** |
| Output tokens | **0** |
| Estimated total cost | **$0.00** |
| Average cost per request (÷ 250) | **$0.00** |

### Design rationale

This is an intentional design choice, not an omission:

- **Fast**: The full 250-request pipeline completes in ~17 seconds on a laptop CPU.
- **Free**: No API quota is consumed at submission or evaluation time.
- **Reproducible**: Output is fully deterministic — identical inputs always produce
  identical outputs, with no dependency on model availability, rate limits, or
  API key configuration.
- **Sufficient**: The heuristic regex layer captures salary changes, amount updates,
  contract cancellations, gig-income pending status, and scheduled-event inference
  with enough fidelity to achieve the measured accuracy on the 25-sample ground truth.

The `gemini_client` plumbing remains in the codebase and can be activated by passing
a live client to `build_user_data()` in `main.py` — no other code changes needed.

---

## SECTION 2 — Development-Time Model Usage

**Clearly labeled: reconstructed estimate, not measured data.**
No usage dashboard was available; figures are estimated from session logs and
approximate message counts in `log.txt` (419 lines, ~12-hour session,
2026-09-12 20:23 IST to 2026-09-13 13:08 IST).

### Models used

| Model | Role during build | Estimated turns |
|---|---|---|
| Claude Opus 4.6 (Thinking) | Planning, implementation plan revisions, reasoning-heavy debugging (income projection root-cause analysis, binary search logic, tie-breaker audit reasoning) | ~25–35 turns |
| Claude Sonnet 4.6 (Thinking) | Targeted fixes, structural validation, code review passes, log maintenance | ~15–20 turns |
| Gemini 3.1 Pro (High) | Bulk code generation, data schema inspection, heuristic regex authoring, pipeline scaffolding | ~20–30 turns |

### Estimated token consumption (development only)

These figures use rough averages for an agentic coding session with large file context:

| Model | Est. input tokens | Est. output tokens | Est. total tokens | Est. cost |
|---|---|---|---|---|
| Claude Opus 4.6 | ~600,000 | ~120,000 | ~720,000 | ~$21.60 |
| Claude Sonnet 4.6 | ~200,000 | ~40,000 | ~240,000 | ~$1.44 |
| Gemini 3.1 Pro | ~400,000 | ~80,000 | ~480,000 | ~$2.40 |
| **Total (development)** | **~1,200,000** | **~240,000** | **~1,440,000** | **~$25.44** |

*Pricing references: Claude Opus 4.6 $15/$75 per M tokens input/output; Claude Sonnet 4.6 $3/$15; Gemini 3.1 Pro $1.25/$5. All figures are rough estimates only.*

### Average per-request development cost

- **~$0.10 per request** (total development cost ÷ 250 requests)

---

## Summary

| Category | Live API calls | Tokens | Cost |
|---|---|---|---|
| **Production runtime** | 0 | 0 | $0.00 |
| **Development (estimated)** | ~70–85 turns | ~1.44M | ~$25.44 |

The submission is **zero-cost to run at evaluation time**. All model-assisted
reasoning happened exclusively during development and is not replayed at inference.
