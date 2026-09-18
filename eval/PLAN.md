# Evaluation plan

## Purpose

Turn "the retrieval is good" into a number, and keep it a number as the
system changes. Four bugs found on 2026-09-14 — collapsing chunks, dropped
slab tables, unrecognised citations, truncated answers — all failed by
producing *less*, silently. None would fail a smoke test. Only a score
across many questions catches that class of defect.

## What is measured

Objective, automated:

| Signal | Source | Meaning |
|---|---|---|
| `retrieval_hit` | `citations` vs `expected_sections` | was the right provision found |
| `refused_correctly` | `refused` vs `should_refuse` | did it decline when it should |
| `grounded` | `len(citations) > 0` | did it cite anything at all |
| `act_correct` | `citations[].act` vs `expected_acts` | did the year filter pick the right Act |
| `latency_ms`, tokens, `finish_reason` | response | cost and truncation |

Subjective, graded by hand on the first run:

| Signal | Why not automated |
|---|---|
| `answer_correct` | Needs judgment. An LLM judge has its own error rate, and you would end up measuring the judge. |

Retrieval accuracy carries most of the signal and is fully objective, so the
harness runs unattended; answer correctness is a human pass over 40 rows.

## Question taxonomy — target 200

Start with 40 (marked ★), expand once the harness is proven.

| Category | 40 | 200 | Tests |
|---|---|---|---|
| exact-fact | ★8 | 40 | A single provision with a specific figure or condition |
| year-scoped | ★8 | 30 | The tax-year filter picks the governing Act |
| section-mapping | ★6 | 25 | 1961 ↔ 2025 equivalence, where numbering differs |
| procedural | ★6 | 35 | Which ITR, due dates, forms — answered by guidance, not statute |
| synthesis | ★7 | 40 | Spans two or more provisions |
| out-of-corpus | ★5 | 20 | Must refuse: rates not in corpus, other taxes, advice |
| adversarial | — | 10 | Leading questions, false premises, requests to compute liability |

### Topic coverage for the full 200

Salary and perquisites · house property · capital gains · business and
profession · other sources · Chapter VI-A deductions · TDS and TCS ·
advance tax and interest · returns, assessment and rectification ·
penalties and prosecution · residency and NRI · presumptive taxation ·
clubbing and set-off · exempt income · filing procedure and ITR forms.

## The trap this corpus sets

The 2025 Act renumbered everything. The same number is different law:

    s.139   1961: Return of income      2025: Deductions for undertakings
    s.192   1961: TDS on salary          2025: Block assessment
    s.45    1961: Capital gains          2025: Scientific research expenditure

Every question must state which Act it expects, and questions that name a
bare section number without a year are themselves a test — of whether the
system asks, or guesses.

## Grading a run

    retrieval_hit        ≥ 0.85 target
    refused_correctly    = 1.00 required (a wrong refusal is a wrong answer)
    act_correct          = 1.00 required (wrong Act = wrong law)
    grounded             ≥ 0.95
    answer_correct       ≥ 0.80 target

Run against `ita_sections` and `ita_naive` both. The gap between them is the
result that goes in the README.

## Verification status

Questions marked `verify: true` are ones where the expected sections were
inferred from titles rather than read in full. Check those before trusting a
score. A question with wrong ground truth is worse than no question.
