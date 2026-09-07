# W3 Conclusion + Abstract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Chapter 7 outline with three paragraphs and rewrite the Abstract from the finished chapters.

**Architecture:** Text-only changes to `thesis/master_thesis/Conclusion.tex` and the Abstract block of `main.tex`, plus a one-sentence consistency fix in `Introduction.tex` (closure rates named by setting). Verification is the LaTeX build, a grep for leftover outline markers, and a number cross-check against the JSON sources.

**Tech Stack:** uplatex/dvipdfmx via latexmk, pytest (existing suite).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-07-w3-conclusion-abstract-design.md`.
- Numbers (verified 2026-09-07): strat_A 0.5206/0.7429, strat_B 0.4661/0.689; external best mixtures A 0.6008/0.6020, B 0.5548/0.5439; LLM 0.2545/0.2258/0.3003; greedy X≥8 +0.056; stage-1 redesign 0.7799; DoF-stop closed 0.9363 (A) / 0.8826 (B) vs oracle 0.8651 / 0.7806, exact-match identical.
- English: active voice, concrete words, abbreviations expanded at first use in each of Abstract and Chapter 7. No commit.

---

### Task 1: Chapter 7 prose
- [ ] Write `Conclusion.tex` (3 paragraphs, `% source:` comment on the results paragraph).
- [ ] `grep -c "\\item \\textbf{P" Conclusion.tex` → 0.

### Task 2: Abstract
- [ ] Replace the Abstract block in `main.tex` (between the `\\` spacer lines and `\newpage`) with the three-paragraph version; update the block comment to record the rewrite date.
- [ ] Fix the Introduction sentence on DoF-stop closure rates to name the settings.

### Task 3: Build and records
- [ ] `cd thesis/master_thesis && latexmk -gg main.tex` → exit 0, no undefined refs; render the Abstract page and the Conclusion page and read them.
- [ ] Devlog entry (uplatex build), gap analysis W3 ✅ (next: B1/B2 → W4–W6), memory update, full pytest.
