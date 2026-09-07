# B1 Second-Judge Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-judge the 608 LLM-generated/ground-truth equation pairs with claude-fable-5-1 using the identical prompt, measure inter-judge agreement (Cohen's κ), and record the result in the thesis.

**Architecture:** `evaluate_llm_equiv.py` exposes `build_prompt()`; `judge_second_opinion.py` iterates the first judge's `per_case` list, calls the second model, and writes the same result format with `_fable51` suffix; `analyze_judge_agreement.py` aligns both files per pair and computes agreement/κ and metric differences; the thesis §5.2 / §6.9 (and §6.1 if needed) cite `judge_agreement_stats.json`.

**Tech Stack:** anthropic Python SDK 0.103 (`client.messages.create`, adaptive thinking, `output_config.effort`), numpy, scikit-learn (κ cross-check in tests), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-07-b1-second-judge-design.md`.
- Model `claude-fable-5-1`; thinking adaptive; no `fallbacks`; check `stop_reason == "refusal"`; `max_tokens` 16000.
- Same prompt as the first judge; same case order; same `[:12]` truncation of generated equations.
- No commit.

---

### Task 1: Shared prompt + second-judge script + agreement analysis with tests
- [ ] Patch `evaluate_llm_equiv.py`: add `build_prompt(inp, out, correct, gen, k)`; `judge()` calls it (no behavior change).
- [ ] Write `tests/test_judge_agreement.py` (κ toy check vs sklearn, alignment assert, pooled stats) → run → fail.
- [ ] Write `analyze_judge_agreement.py` with `cohen_kappa(a, b)`, `pair_labels(per_case, K)`, `agreement(ref_cases, new_cases)`; run tests → pass.
- [ ] Write `judge_second_opinion.py`; smoke: `--label set_A --limit 2 --tag smoke` → prints 2 judgments, writes `_smoke` file.

### Task 2: Full run
- [ ] `caffeinate -i bash run_second_judge.sh` in the background (3 labels in parallel); wait; run the analysis; read the summary.

### Task 3: Thesis and records
- [ ] Verify citation pages (Claude Fable 5.1 / Opus 4.8) with WebFetch; add bibitems only if the pages exist.
- [ ] Edit `Experiment.tex` §5.2 (judge check sentences), `ResultsAndDiscussion.tex` §6.9 item 4 (and §6.1 if the decision rule requires a range); build (exit 0); render the page.
- [ ] Devlog entry (uplatex), gap analysis B1 ✅, memory; full pytest.
