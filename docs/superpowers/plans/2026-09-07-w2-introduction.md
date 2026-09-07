# W2 Introduction + Figure 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Chapter 1 outline (P1–P8) into prose and add Figure 1 (AutoPMoB overview with this study's stage and the closed-set task) after P2.

**Architecture:** `generate_figures_thesis.py` gains `overview()` that draws the two-panel schematic with matplotlib patches and mathtext and saves `figures/fig_autopmob_overview.{pdf,png}`. `Introduction.tex` is rewritten as paragraphs with `% source:` comments; the figure environment sits after the AutoPMoB paragraph.

**Tech Stack:** matplotlib 3.10 (Agg, mathtext), pytest, uplatex/dvipdfmx via latexmk.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-07-w2-introduction-design.md`.
- Numbers only from `experiments/*.json` (values verified 2026-09-07: strat_A 0.5206/0.7429, strat_B 0.4661/0.689, p_wilcoxon 0.00195; LLM 0.2545/0.2258/0.3003; external best mixtures A 0.6008/0.6020, B 0.5548/0.5439; domain agreement −0.0001 p=0.98; greedy X≥8 +0.056; DoF closed 0.8826/0.9363 vs 0.7806/0.8651, exact 0.3678/0.5262; stage-1 redesign 0.7799).
- CSTR equations: dC_A/dt = (F/V)(C_A0 − C_A) − r_A; r_A = k C_A; k = A exp(−E/RT). Edges: ODE–rate law share r_A; rate law–Arrhenius share k.
- English: active voice, no clichés, abbreviations expanded at first use. No commit.

---

### Task 1: Figure 1 generator with test

**Files:**
- Modify: `generate_figures_thesis.py` (add `overview()`, register in `__main__`)
- Test: `tests/test_fig_overview.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import generate_figures_thesis as G

def test_overview_generates_files_and_labels():
    fig = G.overview()
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    for needle in ["this study", "Equation-set retrieval", "Extraction", "Equivalence",
                   "11,146 equations", "361 documents", "r_A = k C_A", "degrees of freedom"]:
        assert needle in texts, needle
    for ext in ("pdf", "png"):
        assert (G.FIG / f"fig_autopmob_overview.{ext}").exists()
```

- [ ] **Step 2: Run** `python3 -m pytest tests/test_fig_overview.py -q` → AttributeError (no `overview`).

- [ ] **Step 3: Implement `overview()`** — two axes (`gridspec` 1.3:2.6), `FancyBboxPatch` boxes, `FancyArrowPatch` arrows, mathtext equations; return the figure after `save(fig, "fig_autopmob_overview")` (change `save` to not close when a `keep` flag is set, or re-create; simplest: `overview()` builds the figure, saves via `fig.savefig` for both extensions, and returns `fig`).

- [ ] **Step 4: Run the test** → PASS; render PNG and inspect visually; iterate on overlaps.

---

### Task 2: Chapter 1 prose

**Files:**
- Modify: `thesis/master_thesis/Introduction.tex` (replace the whole outline)

- [ ] **Step 1:** Write eight paragraphs per spec §2, with the figure environment after P2 and `% source:` comments on P5–P6.
- [ ] **Step 2:** `grep -c "\\\\item \\\\textbf{P" Introduction.tex` → 0.
- [ ] **Step 3:** Build: `cd thesis/master_thesis && latexmk -gg main.tex`; exit 0, no undefined refs; render pages 1–3 and check the figure placement and the text.
- [ ] **Step 4:** Self-check against the 11 English principles and G1/G2/G4/G5; fix inline.

---

### Task 3: Records

**Files:** `docs/development_log.tex`, `docs/論文化ギャップ分析_2026-08-30.md`, memory.

- [ ] Append devlog entry (uplatex build), mark W2 ✅ in the gap analysis (W3 next), update project memory; run the full pytest suite.
