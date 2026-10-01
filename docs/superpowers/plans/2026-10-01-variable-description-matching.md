# Variable-Description Matching (Symbol Replacement and Robustness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Represent every variable by its natural-language description instead of its symbol, compute the variable-overlap features by E5 similarity with a threshold τ and one-to-one matching, and compare symbol matching (S1) with description matching (D1) on Settings A/B × 10 seeds. The comparison also covers test-time symbol renaming (S1-r50/r100) and a leak reference that copies the source descriptions (D0).

**Architecture:** One numeric module (`var_matching.py`) computes the matching, the stage-1 approximation, τ, and the ten reranker-10S features. The same code serves both representations: symbol mode is a 0/1 equality matrix with τ = 1.0, so it must reproduce the existing features exactly. `var_conditions.py` builds the per-condition view of each case (context, query variables, SVD query text) and the two matchers. `run_var_description.py` re-implements the reranker-10S training loop with identical RNG consumption, so S0 reproduces the canonical w30-70 result. `make_var_descriptions.py` generates case-side paraphrases, symbol-free contexts, and alternative symbols through the Anthropic API (Opus 5.5, Message Batches). `var_desc_qc.py` checks the generated outputs. `analyze_var_description.py` runs the pre-check, the paired tests, and the figure.

**Tech Stack:** Python 3.12, numpy 2.2, scipy 1.16 (`scipy.sparse.csgraph.maximum_bipartite_matching`), scikit-learn 1.8, torch 2.7 (CPU for the reranker; MPS for E5), transformers (E5 `intfloat/e5-base-v2` via `external_baselines.embed_texts`), anthropic 0.103.1, python-dotenv, matplotlib, pytest 9.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-01-variable-description-matching-design.md` (approved, commit d04e41c).
- Do not modify existing scripts (`set_aware_reranker.py`, `two_stage_query_conditioned.py`, `external_baselines.py`, `evaluate_multi_eq.py`, …). Import from them only.
- **No git commit / push** unless the user explicitly asks (a post-commit hook auto-pushes to GitHub). Leave all changes in the working tree.
- Run tests from the repo root with `python3 -m pytest …` (this puts the root on `sys.path`; tests import root modules directly).
- Seeds: `42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999`. Split: `set_aware_reranker.stratified_src_split` with the features from `external_baselines.load_setting`. Setting A = `original, multisource_, dae_` (1,823 cases); Setting B = `dae_` (1,000 cases).
- Reranker: reranker-10S (7 base + gComp/gCoh/gDom) with top_k 50, epochs 15, lr 1e-3, hidden 64, margin 0.1, batch 16, n_neg 8, weight_decay 1e-4, pairwise loss. Stage-1 weights: text 0.3, variables 0.7 (w30-70).
- Regression targets: S0, Setting A, seed 42, w30-70 must give `Recall@K_correct` 0.8402538556793877, `MAP` 0.9241361544566646, `Recall@20` 0.9647575894029795, `n_cases` 423 (`experiments/xs1/reranker-10S_w30-70__42.json`; tolerance 1e-4 per spec). S0 stage-1 coverage@50 must be 0.9670508667903073 (w30-70) and 0.9202866372367196 (w70-30) (`experiments/stage1_weight_stats.json`).
- E5: `intfloat/e5-base-v2`, prefix `query: ` on both sides, mean pooling, L2 normalization, cache in `experiments/embeddings/` (via `external_baselines.embed_texts`). DB descriptions are used as extracted, with no preprocessing beyond `norm()` (strip).
- LLM generation only through the Anthropic API, never by the Claude Code session itself. Use model `claude-opus-5-5` (confirmed by the user 2026-10-01) with `output_config={"effort": "medium", "format": {"type": "json_schema", ...}}`. Send no `thinking`, `temperature`, or `tool_choice` parameters. Credentials come from `ANTHROPIC_API_KEY` loaded from `.env` via python-dotenv. Bulk requests go through Message Batches. The standard API path (pilot and retries) adds the server-side fallback (`betas=["server-side-fallback-2026-07-01"]`, `extra_body={"fallbacks": "default"}`; the SDK 0.103.1 has no `fallbacks` keyword) and records the serving model (`response.model`).
- **Spending money:** the 20-case pilot is approved by the spec. **Submitting the batch requires the user's explicit OK after the pilot cost report.**
- Significance means paired t-test **and** Wilcoxon signed-rank test both give p < 0.05 (`analyze_significance.paired_stats`, 10 seeds). The CI is the t-based 95% interval of the 10 per-seed differences. "Equivalent" means the CI lies within ±0.02.
- Pre-check gate: if the E5 AUC for D1 is < 0.80, stop before the main experiment and consult the user.
- `experiments/var_descriptions_raw/` is untracked (add it to `.gitignore`). `experiments/var_descriptions.json` is a tracked deliverable.
- Figure: English labels; colors `#2a78d6` (D1), `#eb6834` (S1), `#1baf7a` (D0). These are validated with the dataviz validator (all-pairs CVD ΔE ≥ 9.2; aqua contrast 2.74 → direct labels required). Error bars are mean ± SD over 10 seeds.

## File Structure

| File | Responsibility |
|---|---|
| `var_matching.py` (new) | Pure numeric: max matching, occurrence incidence, stage-1 approximate counts, Jaccard from counts, Youden τ, f6 domain flags, the 10 reranker-10S features |
| `var_desc_qc.py` (new) | Pure QC of generated outputs: symbol leaks, verbatim copies, word overlap, duplicate/alt-symbol checks |
| `var_conditions.py` (new) | Data loading with variable occurrences; per-condition case views (S0/S1/S1-r/D1/D0); renaming; D0 descriptions; symbol and description matchers; calibration pairs |
| `run_var_description.py` (new) | Per-condition precompute → train → evaluate (S1 model also on S1-r), stage-1 coverage sweep, CLI |
| `make_var_descriptions.py` (new) | Anthropic API generation: pilot, batch submit/collect, retry, finalize (QC → JSON, QC report, 50-row sample CSV) |
| `analyze_var_description.py` (new) | `precheck` (AUC gate, hardest pairs), `stats` (paired tests, CI, verdicts, family breakdown), `figure` |
| `run_var_description.sh` (new) | Main runs A/B in parallel, τ ± 0.02, coverage sweep |
| `tests/test_var_matching.py`, `tests/test_var_desc_qc.py`, `tests/test_var_conditions.py`, `tests/test_run_var_description.py`, `tests/test_make_var_descriptions.py`, `tests/test_analyze_var_description.py` (new) | pytest |
| `.gitignore` (modify) | ignore `experiments/var_descriptions_raw/` |
| `docs/development_log.tex` (modify) | dated entry with the results |

The spec listed five scripts. This plan splits out `var_desc_qc.py` and `var_conditions.py` so that every file has one responsibility and the pure parts can be tested without the API or the full data.

---

### Task 1: Matching, stage-1 approximation, τ, and features (`var_matching.py`)

**Files:**
- Create: `var_matching.py`
- Test: `tests/test_var_matching.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (`two_stage_query_conditioned.jaccard` only in tests).
- Produces:
  - `SYMBOL_TAU: float = 1.0`, `K_REF: int = 5`
  - `max_matching(adj: np.ndarray[bool] (nq×nv)) -> np.ndarray[int64] (nq,)` — matched column per row, −1 if unmatched
  - `occurrence_incidence(eq_ptr: np.ndarray (n_eq+1,)) -> scipy.sparse.csr_matrix (n_occ×n_eq)`
  - `stage1_match_counts(hit: np.ndarray[bool] (nq×n_occ), inc) -> np.ndarray[int64] (n_eq,)`
  - `jaccard_from_counts(m, nq: int, n_vars) -> np.ndarray[float32] (n_eq,)`
  - `youden_tau(sims, labels) -> float`
  - `domain_flags(cands: list[int], eq_domains: list[str], ctx: str) -> np.ndarray[float32]`
  - `case_features(ts_c, svd_c, f6_c, q_in, q_out, QC, CR, cptr, cand_doms, tau, k_ref=K_REF) -> (feats float32 (n_cand×10), m_exact int64 (n_cand,))`. Columns: [f0 text, f1 Jaccard, f2 SVD, f3 input coverage, f4 output coverage, f5 specificity, f6 domain, gComp, gCoh, gDom].

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_var_matching.py
"""var_matching.py（照合・第 1 段の近似・τ・特徴量）の単体テスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.4, §4
"""
import numpy as np

from var_matching import (
    SYMBOL_TAU, max_matching, occurrence_incidence, stage1_match_counts,
    jaccard_from_counts, youden_tau, domain_flags, case_features,
)
from two_stage_query_conditioned import jaccard


def test_max_matching_crossing_case_finds_two():
    # 行 0 は列 0・1、行 1 は列 0 だけ。貪欲に (0,0) を取ると 1 本だが、最大は 2 本
    adj = np.array([[True, True], [True, False]])
    m = max_matching(adj)
    assert (m >= 0).sum() == 2
    assert m[0] == 1 and m[1] == 0


def test_max_matching_two_queries_one_target():
    assert (max_matching(np.array([[True], [True]])) >= 0).sum() == 1


def test_max_matching_empty_inputs():
    assert list(max_matching(np.zeros((3, 0), dtype=bool))) == [-1, -1, -1]
    assert list(max_matching(np.zeros((2, 2), dtype=bool))) == [-1, -1]


def test_stage1_exact_counts_equal_jaccard():
    # 式 0: {a, b}、式 1: 変数なし、式 2: {b, c, d}
    occ_sym = ["a", "b", "b", "c", "d"]
    eq_ptr = np.array([0, 2, 2, 5])
    inc = occurrence_incidence(eq_ptr)
    eq_sets = [{"a", "b"}, set(), {"b", "c", "d"}]
    for q in (["a"], ["b", "c"], ["a", "b", "c", "d", "e"], ["z"]):
        hit = np.array([[s == o for o in occ_sym] for s in q])
        got = jaccard_from_counts(stage1_match_counts(hit, inc), len(q), np.diff(eq_ptr))
        want = np.array([jaccard(set(q), s) for s in eq_sets], dtype=np.float32)
        assert got.dtype == np.float32
        assert np.array_equal(got, want)


def test_stage1_approx_is_upper_bound_and_exact_with_single_edges():
    rng = np.random.default_rng(0)
    eq_ptr = np.array([0, 3, 7, 7, 12])
    inc = occurrence_incidence(eq_ptr)
    for _ in range(300):
        hit = rng.random((4, 12)) < 0.3
        m = stage1_match_counts(hit, inc)
        for j in range(4):
            a, b = eq_ptr[j], eq_ptr[j + 1]
            exact = int((max_matching(hit[:, a:b]) >= 0).sum())
            assert m[j] >= exact
            if (hit[:, a:b].sum(axis=1) <= 1).all():   # 各 q の辺が 1 本以下なら一致する
                assert m[j] == exact


def test_youden_tau_separates_two_groups():
    sims = np.array([0.95, 0.92, 0.90, 0.70, 0.60, 0.50])
    labels = np.array([1, 1, 1, 0, 0, 0])
    tau = youden_tau(sims, labels)
    assert 0.70 < tau <= 0.90


def test_domain_flags_substring_rule():
    flags = domain_flags([0, 1, 2], ["CSTR", "", "Distillation"], "A CSTR with a jacket")
    assert list(flags) == [1.0, 0.0, 0.0]


def test_case_features_symbol_toy():
    # 照会 Q = (C 出力, F 入力, T 入力)。候補 0 = {F, T, V}、候補 1 = {C, T}。参照集合 = 上位 5 件 = 両方
    q = np.array(["C", "F", "T"], dtype=object)
    q_in = np.array([False, True, True])
    q_out = np.array([True, False, False])
    cv = np.array(["F", "T", "V", "C", "T"], dtype=object)
    cptr = np.array([0, 3, 5])
    QC = (q[:, None] == cv[None, :]).astype(np.float32)
    CR = (cv[:, None] == cv[None, :5]).astype(np.float32)
    feats, m = case_features(np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([1.0, 0.0], dtype=np.float32),
                             q_in, q_out, QC, CR, cptr, ["cstr", "cstr"], SYMBOL_TAU)
    assert list(m) == [2, 2]
    # 候補 0: |Q∩V|=2 (F,T)、|Q∪V|=4 → f1=0.5、f3=2/2、f4=0/1、f5=2/3
    assert feats[0, 1] == np.float32(0.5) and feats[0, 3] == 1.0 and feats[0, 4] == 0.0
    assert feats[0, 5] == np.float32(2 / 3)
    # 参照（自身を除く）= 候補 1 {C, T}: gComp は F だけ → 1/3、gCoh は V={F,T,V} のうち T → 1/3、gDom = 1
    assert feats[0, 7] == np.float32(1 / 3) and feats[0, 8] == np.float32(1 / 3) and feats[0, 9] == 1.0
    assert feats[0, 0] == np.float32(0.1) and feats[0, 2] == np.float32(0.3) and feats[0, 6] == 1.0
    # 候補 1: f1 = 2/3、f3 = 1/2（T）、f4 = 1、f5 = 1。参照 = 候補 0 {F,T,V}: gComp は C → 1/3、gCoh は T → 1/2
    assert feats[1, 1] == np.float32(2 / 3) and feats[1, 3] == np.float32(0.5) and feats[1, 4] == 1.0
    assert feats[1, 5] == 1.0 and feats[1, 7] == np.float32(1 / 3) and feats[1, 8] == np.float32(0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_var_matching.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'var_matching'`

- [ ] **Step 3: Write the implementation**

```python
# var_matching.py
"""変数の照合と特徴量（説明版と記号版を同じコードで計算する）.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.4
- 「同じ変数」= 類似度 ≥ τ。記号版は完全一致の 0/1 行列と τ = SYMBOL_TAU で同じ関数を通す。
- 1 対 1 の対応は二部グラフの最大マッチング（Hopcroft–Karp）。
- 第 1 段（全式）は |M| の上限 min(対応先のある Q の数, 対応先のある V の数) で近似する。
- case_features は set_aware_reranker.compute_features_with_set（reranker-10S）と同じ 10 列を返す。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

SYMBOL_TAU = 1.0  # 記号版：一致 = 1.0 / 不一致 = 0.0 の行列と組で使う
K_REF = 5         # set_aware_reranker.K_REF と同じ（参照集合 = 第 1 段の上位 5 件）


def max_matching(adj: np.ndarray) -> np.ndarray:
    """bool 行列 adj（照会の変数 × 式の変数）の最大マッチング。各行の対応列（なければ -1）."""
    nq, nv = adj.shape
    if nq == 0 or nv == 0 or not adj.any():
        return np.full(nq, -1, dtype=np.int64)
    m = maximum_bipartite_matching(csr_matrix(adj.astype(np.int8)), perm_type="column")
    return np.asarray(m, dtype=np.int64)


def occurrence_incidence(eq_ptr: np.ndarray) -> csr_matrix:
    """変数の出現（式ごとに連続して並ぶ）→ 式 の 0/1 行列 (n_occ × n_eq)。変数のない式は空の列."""
    eq_ptr = np.asarray(eq_ptr, dtype=np.int64)
    n_eq, n_occ = len(eq_ptr) - 1, int(eq_ptr[-1])
    cols = np.repeat(np.arange(n_eq), np.diff(eq_ptr))
    return csr_matrix((np.ones(n_occ, dtype=np.float32), (np.arange(n_occ), cols)),
                      shape=(n_occ, n_eq))


def stage1_match_counts(hit: np.ndarray, inc: csr_matrix) -> np.ndarray:
    """第 1 段の近似 |M|（全式）。hit: bool (|Q| × n_occ)、inc: occurrence_incidence.

    min(対応先のある Q の数, 対応先のある V の数) は最大マッチングの大きさの上限で、
    完全一致の照合なら |Q ∩ V| と等しい。
    """
    n_eq = inc.shape[1]
    if hit.shape[0] == 0:
        return np.zeros(n_eq, dtype=np.int64)
    per_q = np.asarray(inc.T @ hit.T.astype(np.float32)).T          # (|Q| × n_eq)
    n_q = (per_q > 0).sum(axis=0)
    n_v = np.asarray(inc.T @ hit.any(axis=0).astype(np.float32)).ravel()
    return np.minimum(n_q, n_v).astype(np.int64)


def jaccard_from_counts(m, nq: int, n_vars) -> np.ndarray:
    """m / (|Q| + |V| − m)（分母 0 は 0）を float32 で。two_stage_query_conditioned.jaccard と同じ値."""
    m = np.asarray(m, dtype=np.float64)
    union = nq + np.asarray(n_vars, dtype=np.float64) - m
    out = np.zeros(len(m), dtype=np.float64)
    np.divide(m, union, out=out, where=union > 0)
    return out.astype(np.float32)


def youden_tau(sims, labels) -> float:
    """正例 (1) と負例 (0) を最もよく分ける τ（Youden の J = 真陽性率 − 偽陽性率 が最大）。sim ≥ τ を正と判定."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(np.asarray(labels), np.asarray(sims, dtype=np.float64))
    return float(thr[int(np.argmax(tpr - fpr))])


def domain_flags(cands: list, eq_domains: list, ctx: str) -> np.ndarray:
    """f6（分野の一致）。compute_features_with_set と同じ規則（小文字の部分文字列）."""
    dl = ctx.lower()
    out = np.zeros(len(cands), dtype=np.float32)
    for k, j in enumerate(cands):
        d = eq_domains[j].lower()
        out[k] = 1.0 if (d and d in dl) or (dl and dl in d) else 0.0
    return out


def case_features(ts_c, svd_c, f6_c, q_in, q_out, QC, CR, cptr, cand_doms, tau, k_ref=K_REF):
    """reranker-10S の 10 列 [f0, f1, f2, f3, f4, f5, f6, gComp, gCoh, gDom] と各候補の |M|.

    ts_c, svd_c, f6_c: 候補ごとの f0, f2, f6
    q_in, q_out: 照会の各変数が入力か・出力か（bool, |Q|）
    QC: 照会の変数 × 候補の変数（候補の順に連結）の類似度 (|Q| × cptr[-1])
    CR: 候補の変数 × 参照集合（上位 k_ref 件）の変数の類似度 (cptr[-1] × cptr[k_ref])
    cptr: 候補 k の変数は列 cptr[k]:cptr[k+1]
    cand_doms: 候補の分野（gDom 用、正規化済み）
    """
    n = len(cptr) - 1
    nq = QC.shape[0]
    q_in = np.asarray(q_in, dtype=bool)
    q_out = np.asarray(q_out, dtype=bool)
    n_in, n_out = int(q_in.sum()), int(q_out.sum())
    kr = min(k_ref, n)
    ref_end = int(cptr[kr])
    QR = QC[:, :ref_end] >= tau
    CRb = CR >= tau
    feats = np.zeros((n, 10), dtype=np.float32)
    m_exact = np.zeros(n, dtype=np.int64)
    for k in range(n):
        a, b = int(cptr[k]), int(cptr[k + 1])
        nv = b - a
        matched = max_matching(QC[:, a:b] >= tau) >= 0
        m = int(matched.sum())
        m_exact[k] = m
        union = nq + nv - m
        feats[k, 0] = ts_c[k]
        feats[k, 1] = m / union if union else 0.0
        feats[k, 2] = svd_c[k]
        feats[k, 3] = int((matched & q_in).sum()) / n_in if n_in else 0.0
        feats[k, 4] = int((matched & q_out).sum()) / n_out if n_out else 0.0
        feats[k, 5] = m / nv if nv else 0.0
        feats[k, 6] = f6_c[k]
        keep = np.ones(ref_end, dtype=bool)       # 参照集合の変数の列（自分自身を除く）
        if k < kr:
            keep[a:b] = False
            ref = [r for r in range(kr) if r != k]
        else:
            ref = list(range(kr))
        q_cov = QR[:, keep].any(axis=1)
        feats[k, 7] = int((matched & ~q_cov).sum()) / nq if nq else 0.0
        if nv:
            feats[k, 8] = int(CRb[a:b][:, keep].any(axis=1).sum()) / nv
        if ref:
            same = sum(1 for r in ref if cand_doms[r] == cand_doms[k] and cand_doms[r])
            feats[k, 9] = same / len(ref)
    return feats, m_exact
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_var_matching.py -q`
Expected: `8 passed`

- [ ] **Step 5: Checkpoint (no commit)**

Run: `git status --short`
Expected: `?? var_matching.py` and `?? tests/test_var_matching.py` only (plus the plan file). Do not commit.

---

### Task 2: QC of generated outputs (`var_desc_qc.py`)

**Files:**
- Create: `var_desc_qc.py`
- Test: `tests/test_var_desc_qc.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `MATH_CHARS: set[str]` = `{"$", "\\", "_", "^", "{", "}"}`
  - `bare(symbol: str) -> str` — LaTeX decorations removed (`\alpha` → `alpha`, `T_{in}` → `Tin`)
  - `symbol_leaks(text: str, symbols: list[str]) -> tuple[list[str], list[str]]` — (hard, review)
  - `normalize_desc(s: str) -> str` — lowercase, bracketed units removed, punctuation → space
  - `word_jaccard(a: str, b: str) -> float`
  - `check_output(payload: dict, output: dict | None) -> tuple[list[str], list[str]]` — codes `"category|detail"`. Hard categories: `no-output, symbol-set, context-empty, context-leak, para-empty, para-leak, verbatim, dup-para, alt-same, alt-collides, alt-dup`. Review categories: `context-single, para-single`.
  - `payload` shape (from Task 5): `{"context": str, "variables": [{"symbol", "role", "descriptions": [str]}]}`; `output` shape: `{"context_nosym": str, "variables": [{"symbol", "paraphrase", "alt_symbol", "meaning_conflict"}]}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_var_desc_qc.py
"""var_desc_qc.py（生成結果の品質の確認）の単体テスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.2, §4 の 6
"""
from var_desc_qc import bare, symbol_leaks, normalize_desc, word_jaccard, check_output


def test_bare_removes_latex_decorations():
    assert bare("\\alpha") == "alpha" and bare("T_{in}") == "Tin" and bare("$C_A$") == "CA"


def test_symbol_leaks_hard_and_review():
    syms = ["T", "T_{in}", "\\alpha", "UA", "a"]
    assert symbol_leaks("temperature $T$ of the reactor", syms)[0]     # 数式記号の文字
    assert symbol_leaks("feed T_{in}", syms)[0]
    assert symbol_leaks("overall UA value", syms)[0]                   # 2 文字以上の記号
    assert symbol_leaks("the alpha coefficient", syms)[0]              # 装飾を除いた形
    hard, review = symbol_leaks("temperature of the feed entering a reactor", syms)
    assert hard == [] and review == []                                 # 冠詞 a は見ない
    hard, review = symbol_leaks("the T of the jacket", syms)
    assert hard == [] and review == ["single|T"]                       # 1 文字は目視へ
    hard, review = symbol_leaks("A reactor with a jacket", ["A"])
    assert hard == [] and review == []                                 # 文頭の A は冠詞として扱う


def test_normalize_and_overlap():
    assert normalize_desc("Inlet temperature for reactor vessel [K]") == "inlet temperature for reactor vessel"
    assert word_jaccard("temperature of the feed", "inlet temperature [K]") == 1 / 5


PAYLOAD = {"context": "Reactor with jacket; share variables $T$, $V$.",
           "variables": [
               {"symbol": "T", "role": "input", "descriptions": ["temperature inside the reactor vessel [K]"]},
               {"symbol": "T_j", "role": "input", "descriptions": ["temperature inside the jacket [K]"]},
               {"symbol": "V", "role": "output", "descriptions": ["reactor vessel volume [L]"]}]}


def good():
    return {"context_nosym": "Reactor with jacket; the reactor temperature and volume are shared.",
            "variables": [
                {"symbol": "T", "paraphrase": "temperature of the reactor contents",
                 "alt_symbol": "\\theta", "meaning_conflict": False},
                {"symbol": "T_j", "paraphrase": "temperature of the cooling jacket",
                 "alt_symbol": "T_c", "meaning_conflict": False},
                {"symbol": "V", "paraphrase": "volume of liquid held in the reactor",
                 "alt_symbol": "V_R", "meaning_conflict": False}]}


def _has(codes, cat):
    return any(c.startswith(cat + "|") for c in codes)


def test_check_output_good_has_no_hard_issue():
    hard, review = check_output(PAYLOAD, good())
    assert hard == [] and review == []


def test_check_output_detects_each_problem():
    o = good(); o["variables"][0]["paraphrase"] = "Temperature inside the reactor vessel"
    assert _has(check_output(PAYLOAD, o)[0], "verbatim")
    o = good(); o["variables"][1]["paraphrase"] = o["variables"][0]["paraphrase"]
    assert _has(check_output(PAYLOAD, o)[0], "dup-para")
    o = good(); o["variables"][2]["alt_symbol"] = "V"
    assert _has(check_output(PAYLOAD, o)[0], "alt-same")
    o = good(); o["variables"][2]["alt_symbol"] = "T"
    assert _has(check_output(PAYLOAD, o)[0], "alt-collides")
    o = good(); o["variables"][1]["alt_symbol"] = "\\theta"
    assert _has(check_output(PAYLOAD, o)[0], "alt-dup")
    o = good(); o["variables"].pop()
    assert _has(check_output(PAYLOAD, o)[0], "symbol-set")
    o = good(); o["context_nosym"] = "share $T$"
    assert _has(check_output(PAYLOAD, o)[0], "context-leak")
    o = good(); o["variables"][0]["paraphrase"] = "temperature T_{j} nearby"
    assert _has(check_output(PAYLOAD, o)[0], "para-leak")
    assert check_output(PAYLOAD, None) == (["no-output|"], [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_var_desc_qc.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'var_desc_qc'`

- [ ] **Step 3: Write the implementation**

```python
# var_desc_qc.py
"""LLM が生成したケース側の説明（記号なしの説明文・言い換え・別の記号）の品質の確認.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.2
問題は "分類|詳細" の文字列で返す。hard は作り直しの対象、review は目視の対象。
"""
from __future__ import annotations

import re

MATH_CHARS = set("$\\_^{}")
_START_ARTICLES = {"A", "I"}  # 文頭では冠詞・代名詞として現れる 1 文字


def bare(symbol: str) -> str:
    """LaTeX の装飾を除いた形（\\alpha → alpha、T_{in} → Tin）."""
    return re.sub(r"[\\{}$_^\s]", "", symbol or "")


def _word(token: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])")


def symbol_leaks(text: str, symbols: list) -> tuple:
    """(hard, review)。hard: 数式記号の文字、2 文字以上の記号（そのまま・装飾を除いた形）の単語としての出現。
    review: 1 文字の記号の単独の語としての出現（冠詞 a は見ない、A・I は文頭以外だけ）."""
    text = text or ""
    hard, review = [], []
    if any(ch in text for ch in MATH_CHARS):
        hard.append("math-char")
    for s in symbols:
        toks = [t for t in dict.fromkeys([(s or "").strip(), bare(s)]) if t and not set(t) & MATH_CHARS]
        for tok in toks:
            if len(tok) >= 2 and _word(tok).search(text):
                hard.append(f"symbol:{s}")
                break
            if len(tok) == 1:
                if tok == "a":
                    continue
                if tok in _START_ARTICLES:
                    pat = re.compile(r"(?<=[A-Za-z0-9,;:)] )" + tok + r"(?![A-Za-z0-9])")
                else:
                    pat = _word(tok)
                if pat.search(text):
                    review.append(f"single|{s}")
                    break
    return hard, review


def normalize_desc(s: str) -> str:
    """小文字化し、角括弧の単位と句読点を除く（写しの判定用）."""
    s = re.sub(r"\[[^\]]*\]", " ", (s or "").lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def word_jaccard(a: str, b: str) -> float:
    A, B = set(normalize_desc(a).split()), set(normalize_desc(b).split())
    return len(A & B) / len(A | B) if (A | B) else 0.0


def check_output(payload: dict, output) -> tuple:
    """生成結果 1 件を確かめる。(hard, review) の問題の一覧."""
    if not output:
        return ["no-output|"], []
    syms = [v["symbol"] for v in payload["variables"]]
    descs = {v["symbol"]: v["descriptions"] for v in payload["variables"]}
    got_list = output.get("variables") or []
    got = {v.get("symbol"): v for v in got_list}
    hard, review = [], []
    if sorted(got) != sorted(syms) or len(got_list) != len(syms):
        hard.append("symbol-set|" + ",".join(sorted(set(syms) ^ set(got))))
    ctx = output.get("context_nosym") or ""
    if not ctx.strip():
        hard.append("context-empty|")
    h, r = symbol_leaks(ctx, syms)
    hard += [f"context-leak|{x}" for x in h]
    review += [f"context-{x}" for x in r]
    seen, alts = {}, []
    for s in syms:
        v = got.get(s)
        if v is None:
            continue
        p = v.get("paraphrase") or ""
        if not p.strip():
            hard.append(f"para-empty|{s}")
        else:
            h, r = symbol_leaks(p, syms)
            hard += [f"para-leak|{s}:{x}" for x in h]
            review += [f"para-{x}" for x in r]
            key = normalize_desc(p)
            if any(key == normalize_desc(d) for d in descs.get(s, [])):
                hard.append(f"verbatim|{s}")
            if key in seen:
                hard.append(f"dup-para|{s}={seen[key]}")
            else:
                seen[key] = s
        a = (v.get("alt_symbol") or "").replace(" ", "")
        if not a or a == s.replace(" ", ""):
            hard.append(f"alt-same|{s}")
        alts.append((s, a))
    orig = {s.replace(" ", "") for s in syms}
    values = [a for _, a in alts]
    for s, a in alts:
        if a and a in orig and a != s.replace(" ", ""):
            hard.append(f"alt-collides|{s}->{a}")
        if a and values.count(a) > 1:
            hard.append(f"alt-dup|{s}->{a}")
    return hard, review
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_var_desc_qc.py -q`
Expected: `5 passed`

- [ ] **Step 5: Checkpoint (no commit)**

Run: `python3 -m pytest tests/test_var_matching.py tests/test_var_desc_qc.py -q`
Expected: `13 passed`

---

### Task 3: Condition views, matchers, calibration pairs (`var_conditions.py`)

**Files:**
- Create: `var_conditions.py`
- Test: `tests/test_var_conditions.py`

**Interfaces:**
- Consumes: Task 1 (`SYMBOL_TAU`, `K_REF`, `occurrence_incidence`, `stage1_match_counts`, `jaccard_from_counts`, `case_features`, `domain_flags` in the equivalence test). From existing code: `external_baselines.load_setting(setting) -> dict(setting, cases, eq_keys, eq_texts, eq_vars, correct_lists, case_srcs, feats)`, `external_baselines.embed_texts(texts, model_name, kind)`, `external_baselines.E5_MODEL`.
- Produces:
  - `CONDITIONS = ("S0", "S1", "D1", "D0")`, `VAR_DESC_JSON` (Path)
  - `@dataclass CaseView(ctx: str, orig_syms: list[str], syms: list[str], texts: list[str] | None, q_in: np.ndarray[bool], q_out: np.ndarray[bool], io_text: str)`
  - `load_data(setting: str) -> dict`. It returns the `load_setting` keys plus `eqs`, `eq_by_key`, `eq_domains`, `occ_sym: list[str]`, `occ_desc: list[str]`, `eq_ptr: int64 (n_eq+1,)`, `n_vars: int64 (n_eq,)`, and `occ_eq: int64 (n_occ,)`. Occurrences are each equation's unique normalized symbols in sorted order, with the first description.
  - `load_var_descriptions(path=VAR_DESC_JSON) -> dict[case_id, entry]`
  - `rename_subset(case_id: str, syms: list[str], p: float) -> set[str]`
  - `d0_description(sym: str, case: dict, eq_by_key: dict) -> str`
  - `io_text(ctx: str, case: dict, rep: Callable[[str], str]) -> str`
  - `build_view(cond, case, vd_case, eq_by_key, p=0.0) -> CaseView`; `build_views(cond, data, vd, p=0.0) -> list[CaseView]`
  - `SymbolMatcher(data, views)` and `DescriptionMatcher(occ_emb, q_emb, q_idx)`, both with `.kind` (`"symbol"`/`"description"`), `.q_occ(ci) -> float32 (|Q|×n_occ)`, `.occ_occ(a_idx, b_idx) -> float32`
  - `embed_unique(texts, kind) -> (np.ndarray, dict[str,int])`, `occ_embeddings(data) -> np.ndarray (n_occ×768)`
  - `make_matcher(cond, data, views)`
  - `calibration_pairs(idx: list[int], views, data) -> dict(case, item, occ, label)` (int64 arrays); `pair_sims(pairs, matcher) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_var_conditions.py
"""var_conditions.py（条件ごとのケースの見え方・照合器・τ の組）のテスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.1, §3.4, §4 の 1・5
実データを読むテスト（load_data）は 10 秒前後かかる。
"""
import numpy as np
import pytest

from var_conditions import (
    rename_subset, build_view, build_views, load_data, SymbolMatcher, calibration_pairs,
)
from two_stage_query_conditioned import case_text, io_vars, in_vars, out_vars


def test_rename_subset_ceil_nested_deterministic():
    syms = ["A", "B", "C"]
    assert rename_subset("x", syms, 0.0) == set()
    assert rename_subset("x", syms, 1.0) == set(syms)
    half = rename_subset("x", syms, 0.5)
    assert len(half) == 2 and half <= rename_subset("x", syms, 1.0)
    assert half == rename_subset("x", list(reversed(syms)), 0.5)
    assert len(rename_subset("y", ["T"], 0.5)) == 1


CASE = {"case_id": "toy_1", "context": "Reactor sharing $T$.",
        "input_variables": ["T", "F"], "output_variables": ["C_A"],
        "correct_model_ids": ["src__eq_2", "src__eq_1"], "variant_type": "original"}
EQ_BY_KEY = {"src__eq_1": {"variables": {"T": "temperature in the tank [K]", "F": "feed flow"}},
             "src__eq_2": {"variables": {"C_A": "concentration of A", "T": "reactor temperature"}}}
VD = {"context_nosym": "Reactor sharing the temperature.",
      "variables": {
          "T": {"paraphrase": "temperature of the reactor", "alt_symbol": "\\theta", "meaning_conflict": False},
          "F": {"paraphrase": "volumetric feed rate", "alt_symbol": "q", "meaning_conflict": False},
          "C_A": {"paraphrase": "concentration of species A", "alt_symbol": "c_A", "meaning_conflict": False}}}


def test_view_S0_matches_case_text():
    v = build_view("S0", CASE, None, EQ_BY_KEY)
    assert v.ctx == case_text(CASE) and v.io_text == case_text(CASE, io=True)
    assert v.orig_syms == sorted(io_vars(CASE)) == v.syms == ["C_A", "F", "T"]
    assert list(v.q_in) == [False, True, True] and list(v.q_out) == [True, False, False]
    assert v.texts is None


def test_view_S1_rename_all():
    v = build_view("S1", CASE, VD, EQ_BY_KEY, p=1.0)
    assert v.ctx == "Reactor sharing the temperature."
    assert v.syms == ["c_A", "q", "\\theta"] and v.orig_syms == ["C_A", "F", "T"]
    assert v.io_text == "Reactor sharing the temperature. INPUT \\theta q OUTPUT c_A"
    v0 = build_view("S1", CASE, VD, EQ_BY_KEY, p=0.0)
    assert v0.syms == ["C_A", "F", "T"] and v0.io_text == "Reactor sharing the temperature. INPUT T F OUTPUT C_A"


def test_view_D1_and_D0():
    d1 = build_view("D1", CASE, VD, EQ_BY_KEY)
    assert d1.texts == ["concentration of species A", "volumetric feed rate", "temperature of the reactor"]
    assert d1.io_text == ("Reactor sharing the temperature. INPUT temperature of the reactor "
                          "volumetric feed rate OUTPUT concentration of species A")
    d0 = build_view("D0", CASE, VD, EQ_BY_KEY)
    # 正解式の順（src__eq_2 → src__eq_1）で記号を最初に含む式の説明
    assert d0.texts == ["concentration of A", "feed flow", "reactor temperature"]


def test_calibration_pairs_labels():
    d = {"eq_ptr": np.array([0, 2, 4]), "occ_sym": ["F", "T", "C_A", "T"], "correct_lists": [[1, 0]]}
    v = build_view("D1", CASE, VD, EQ_BY_KEY)
    pr = calibration_pairs([0], [v], d)
    got = sorted(zip(pr["item"].tolist(), pr["occ"].tolist(), pr["label"].tolist()))
    # 照会 (C_A, F, T)。式 0 = {F, T}、式 1 = {C_A, T}
    assert got == sorted([(0, 2, 1), (0, 3, 0), (1, 0, 1), (1, 1, 0),
                          (2, 2, 0), (2, 3, 1), (2, 0, 0), (2, 1, 1)])


@pytest.fixture(scope="module")
def data_a():
    return load_data("A")


def test_load_data_occurrences(data_a):
    d = data_a
    assert len(d["cases"]) == 1823 and len(d["eq_keys"]) == 11146
    assert int(d["eq_ptr"][-1]) == len(d["occ_sym"]) == len(d["occ_desc"]) == 58008
    assert int((d["n_vars"] == 0).sum()) == 54
    for j in range(0, 11146, 97):
        a, b = d["eq_ptr"][j], d["eq_ptr"][j + 1]
        assert set(d["occ_sym"][a:b]) == d["eq_vars"][j]


def test_symbol_features_reproduce_existing(data_a):
    """完全一致の照合で、新しい特徴量が set_aware_reranker.compute_features_with_set と一致する（§4 の 1）."""
    from set_aware_reranker import compute_features_with_set
    from var_matching import (SYMBOL_TAU, K_REF, occurrence_incidence, stage1_match_counts,
                              jaccard_from_counts, case_features, domain_flags)
    d = data_a
    views = build_views("S0", d, None)
    sm = SymbolMatcher(d, views)
    inc = occurrence_incidence(d["eq_ptr"])
    n_eq = len(d["eq_keys"])
    ts = np.linspace(0.0, 1.0, n_eq)
    svd = np.linspace(1.0, 0.0, n_eq)
    for ci in list(range(0, len(d["cases"]), 91))[:20]:
        v = views[ci]
        full = sm.q_occ(ci)
        vs = jaccard_from_counts(stage1_match_counts(full >= SYMBOL_TAU, inc), len(v.syms), d["n_vars"])
        cands = np.argsort(-vs)[:50].tolist()
        occ = np.concatenate([np.arange(d["eq_ptr"][j], d["eq_ptr"][j + 1]) for j in cands]).astype(np.int64)
        cptr = np.concatenate([[0], np.cumsum([d["n_vars"][j] for j in cands])]).astype(np.int64)
        ref_end = int(cptr[min(K_REF, len(cands))])
        new, _ = case_features(ts[cands], svd[cands], domain_flags(cands, d["eq_domains"], v.ctx),
                               v.q_in, v.q_out, full[:, occ], sm.occ_occ(occ, occ[:ref_end]),
                               cptr, [d["eq_domains"][j] for j in cands], SYMBOL_TAU)
        c = d["cases"][ci]
        old = compute_features_with_set(cands, ts, io_vars(c), in_vars(c), out_vars(c), svd,
                                        d["eq_vars"], d["eq_domains"], case_text(c), None, io_vars(c),
                                        False, ("Comp", "Coh", "Dom"))
        assert np.array_equal(new, old), c["case_id"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_var_conditions.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'var_conditions'`

- [ ] **Step 3: Write the implementation**

```python
# var_conditions.py
"""条件ごとのケースの見え方（説明文・照会の変数）と、変数の照合器.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.1, §3.4
条件:
  S0 元の説明文 ＋ 文献の記号
  S1 記号なしの説明文 ＋ 文献の記号（p > 0 でテスト用に一部を別の記号へ = S1-r）
  D1 記号なしの説明文 ＋ LLM の言い換え
  D0 記号なしの説明文 ＋ 正解式の説明をそのまま（漏れの大きさを測る参考）
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from two_stage_query_conditioned import (  # noqa: E402
    load_equations, norm, eq_key, case_text, io_vars, in_vars, out_vars,
)
from external_baselines import load_setting, embed_texts, E5_MODEL  # noqa: E402

EXP = ROOT / "experiments"
VAR_DESC_JSON = EXP / "var_descriptions.json"
CONDITIONS = ("S0", "S1", "D1", "D0")


@dataclass
class CaseView:
    ctx: str               # 説明文（f0・第 1 段の文章類似度・f6 に使う）
    orig_syms: list        # 照会の変数の元の記号（sorted(io_vars)）
    syms: list             # 照合に使う記号（S1-r では一部が別の記号）
    texts: list | None     # 説明版の照会テキスト（D1/D0）。記号版は None
    q_in: np.ndarray       # bool: 入力変数か
    q_out: np.ndarray      # bool: 出力変数か
    io_text: str           # f2（SVD）の照会文。case_text(c, io=True) と同じ形


def load_data(setting: str) -> dict:
    """設定 A/B のデータ（external_baselines.load_setting ＋ 分野・変数の出現）."""
    data = load_setting(setting)
    eqs = [e for e in load_equations() if eq_key(e)]
    assert [eq_key(e) for e in eqs] == data["eq_keys"]
    occ_sym, occ_desc, eq_ptr = [], [], [0]
    for e in eqs:
        vd = e.get("variables") or {}
        first = {}
        if isinstance(vd, dict):
            for k, v in vd.items():
                s = norm(k)
                if s and s not in first:
                    first[s] = norm(v) if isinstance(v, str) else ""
        for s in sorted(first):
            occ_sym.append(s)
            occ_desc.append(first[s])
        eq_ptr.append(len(occ_sym))
    eq_ptr = np.asarray(eq_ptr, dtype=np.int64)
    n_vars = np.diff(eq_ptr)
    data.update(
        eqs=eqs,
        eq_by_key=dict(zip(data["eq_keys"], eqs)),
        eq_domains=[norm(e.get("domain") or "") for e in eqs],
        occ_sym=occ_sym, occ_desc=occ_desc, eq_ptr=eq_ptr, n_vars=n_vars,
        occ_eq=np.repeat(np.arange(len(eqs)), n_vars),
    )
    return data


def load_var_descriptions(path: Path = VAR_DESC_JSON) -> dict:
    """make_var_descriptions.py finalize の出力（case_id → 生成結果）."""
    return json.load(open(path, encoding="utf-8"))["cases"]


def rename_subset(case_id: str, syms: list, p: float) -> set:
    """付け替える変数（⌈p × 変数の数⌉ 個）。case_id から決まる乱数の並びの先頭なので、p を変えても入れ子."""
    n = math.ceil(round(p * len(syms), 9))
    if n <= 0:
        return set()
    order = sorted(syms)
    random.Random(int(hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:8], 16)).shuffle(order)
    return set(order[:n])


def d0_description(sym: str, case: dict, eq_by_key: dict) -> str:
    """正解式（correct_model_ids の順）のうち、記号 sym を最初に含む式の説明."""
    for m in case.get("correct_model_ids") or []:
        e = eq_by_key.get(norm(m))
        if not e:
            continue
        for k, v in (e.get("variables") or {}).items():
            if norm(k) == sym:
                return norm(v) if isinstance(v, str) else ""
    raise KeyError(f"{case.get('case_id')}: symbol {sym!r} not in correct equations")


def io_text(ctx: str, case: dict, rep) -> str:
    """case_text(case, io=True) と同じ形で、入出力の各記号を rep(記号) に置き換えた文字列."""
    ins = " ".join(rep(norm(v)) for v in (case.get("input_variables") or []) if norm(v))
    outs = " ".join(rep(norm(v)) for v in (case.get("output_variables") or []) if norm(v))
    return " ".join(filter(None, [ctx, f"INPUT {ins}" if ins else "", f"OUTPUT {outs}" if outs else ""]))


def build_view(cond: str, case: dict, vd_case, eq_by_key: dict, p: float = 0.0) -> CaseView:
    orig = sorted(io_vars(case))
    iv, ov = in_vars(case), out_vars(case)
    q_in = np.array([s in iv for s in orig], dtype=bool)
    q_out = np.array([s in ov for s in orig], dtype=bool)
    if cond == "S0":
        return CaseView(case_text(case), orig, list(orig), None, q_in, q_out, case_text(case, io=True))
    if vd_case is None:
        raise KeyError(f"{case.get('case_id')}: no generated descriptions in var_descriptions.json")
    ctx = norm(vd_case["context_nosym"])
    var = vd_case["variables"]
    if cond == "S1":
        alt = {s: norm(var[s]["alt_symbol"]) for s in rename_subset(case["case_id"], orig, p)}
        rep = lambda s: alt.get(s, s)  # noqa: E731
        return CaseView(ctx, orig, [rep(s) for s in orig], None, q_in, q_out, io_text(ctx, case, rep))
    if cond == "D1":
        para = {s: norm(var[s]["paraphrase"]) for s in orig}
    elif cond == "D0":
        para = {s: d0_description(s, case, eq_by_key) for s in orig}
    else:
        raise ValueError(cond)
    rep = lambda s: para[s]  # noqa: E731
    return CaseView(ctx, orig, list(orig), [para[s] for s in orig], q_in, q_out, io_text(ctx, case, rep))


def build_views(cond: str, data: dict, vd, p: float = 0.0) -> list:
    return [build_view(cond, c, (vd or {}).get(c.get("case_id")), data["eq_by_key"], p)
            for c in data["cases"]]


class SymbolMatcher:
    """記号の完全一致（一致 = 1.0）。τ = SYMBOL_TAU と組で使う."""
    kind = "symbol"

    def __init__(self, data: dict, views: list):
        self.occ = np.asarray(data["occ_sym"], dtype=object)
        self.pos = {}
        for o, s in enumerate(data["occ_sym"]):
            self.pos.setdefault(s, []).append(o)
        self.syms = [list(v.syms) for v in views]

    def q_occ(self, ci: int) -> np.ndarray:
        syms = self.syms[ci]
        out = np.zeros((len(syms), len(self.occ)), dtype=np.float32)
        for i, s in enumerate(syms):
            out[i, self.pos.get(s, [])] = 1.0
        return out

    def occ_occ(self, a, b) -> np.ndarray:
        return (self.occ[a][:, None] == self.occ[b][None, :]).astype(np.float32)


class DescriptionMatcher:
    """説明の E5 埋め込みのコサイン類似度."""
    kind = "description"

    def __init__(self, occ_emb: np.ndarray, q_emb: np.ndarray, q_idx: list):
        self.occ_emb, self.q_emb, self.q_idx = occ_emb, q_emb, q_idx

    def q_occ(self, ci: int) -> np.ndarray:
        return self.q_emb[self.q_idx[ci]] @ self.occ_emb.T

    def occ_occ(self, a, b) -> np.ndarray:
        return self.occ_emb[a] @ self.occ_emb[b].T


def embed_unique(texts: list, kind: str) -> tuple:
    """重複を除いて E5 で埋め込む（接頭辞 'query: '、キャッシュつき）。(行列, テキスト→行)."""
    uniq = sorted(set(texts))
    E = embed_texts(["query: " + t for t in uniq], E5_MODEL, kind)
    return E, {t: i for i, t in enumerate(uniq)}


def occ_embeddings(data: dict) -> np.ndarray:
    """DB の変数の出現ごとの埋め込み (n_occ × 768)。data にキャッシュする."""
    if "_occ_emb" not in data:
        E, idx = embed_unique(data["occ_desc"], "vardesc_db")
        data["_occ_emb"] = E[[idx[d] for d in data["occ_desc"]]]
    return data["_occ_emb"]


def make_matcher(cond: str, data: dict, views: list):
    if cond in ("S0", "S1"):
        return SymbolMatcher(data, views)
    q_emb, idx = embed_unique([t for v in views for t in v.texts], f"vardesc_{cond}")
    return DescriptionMatcher(occ_embeddings(data), q_emb,
                              [np.array([idx[t] for t in v.texts], dtype=np.int64) for v in views])


def calibration_pairs(idx: list, views: list, data: dict) -> dict:
    """τ の決定と事前確認に使う組。正例: 照会の変数 s と、正解式の中の同じ記号 s の説明。
    負例: 同じ正解式の中の別の記号の説明."""
    eq_ptr, occ_sym, cl = data["eq_ptr"], data["occ_sym"], data["correct_lists"]
    rows = []
    for ci in idx:
        for k, s in enumerate(views[ci].orig_syms):
            for j in cl[ci]:
                a, b = int(eq_ptr[j]), int(eq_ptr[j + 1])
                if s not in occ_sym[a:b]:
                    continue
                for o in range(a, b):
                    rows.append((ci, k, o, 1 if occ_sym[o] == s else 0))
    arr = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    return {"case": arr[:, 0], "item": arr[:, 1], "occ": arr[:, 2], "label": arr[:, 3]}


def pair_sims(pairs: dict, matcher) -> np.ndarray:
    """組ごとの類似度（説明版の照合器だけ）."""
    rows = [matcher.q_idx[c][k] for c, k in zip(pairs["case"].tolist(), pairs["item"].tolist())]
    return np.einsum("ij,ij->i", matcher.q_emb[rows], matcher.occ_emb[pairs["occ"]])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_var_conditions.py -q`
Expected: `7 passed` (about 10–30 s because `load_data("A")` reads the full DB).
If `test_symbol_features_reproduce_existing` fails, print `np.argwhere(new != old)` for the failing case. Fix `case_features` in Task 1's file; do not change the reference function.

- [ ] **Step 5: Checkpoint (no commit)**

Run: `python3 -m pytest tests/test_var_matching.py tests/test_var_desc_qc.py tests/test_var_conditions.py -q`
Expected: `20 passed`

---

### Task 4: Runner with exact S0 reproduction (`run_var_description.py`)

**Files:**
- Create: `run_var_description.py`
- Test: `tests/test_run_var_description.py`

**Interfaces:**
- Consumes: Task 1 (`SYMBOL_TAU, K_REF, occurrence_incidence, stage1_match_counts, jaccard_from_counts, youden_tau, domain_flags, case_features`) and Task 3 (`CONDITIONS, load_data, load_var_descriptions, build_views, make_matcher, calibration_pairs, pair_sims`). From existing code: `two_stage_query_conditioned.Reranker, norm, in_vars, out_vars`, `set_aware_reranker.stratified_src_split`, `evaluate_multi_eq.compute_all_ranks, case_metrics, aggregate_metrics`.
- Produces: CLI `python3 run_var_description.py --setting {A,B} [--conds S0,S1,D1,D0] [--seed-list ...] [--renames 0.5,1.0] [--tau-delta X] [--w-text 0.3 --w-var 0.7] [--coverage] [--save-per-case] --output PATH`.
  - Training/evaluation output JSON: `{"config": {...}, "tau": {label: {seed_str: τ}}, "approx_mismatch": {label: {seed_str: rate}}, "results": {label: summary}}`. Labels: `S0, S1, S1-r50, S1-r100, D1, D0`, or `D1-tau-0.02` / `D1-tau+0.02` when `--tau-delta` ≠ 0. Each summary has the same shape as `set_aware_reranker.run_mode`: metric `{mean, std}`, `per_seed` (list of `aggregate_metrics` dicts with `seed`), and optionally `per_case` (records with `seed, mode, case_id, variant, Recall@K_correct, AP, …`).
  - Coverage output JSON (`--coverage`): `{"config": {...}, "coverage": {cond: {"tau": τ, "w70-30": c, "w50-50": c, "w30-70": c, "w00-100": c}}}`.

- [ ] **Step 1: Write the failing (slow) regression tests**

```python
# tests/test_run_var_description.py
"""run_var_description.py の回帰の確認: S0 が既報（w30-70・seed 42）と第 1 段の被覆率を再現する.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §4 の 7
実データで学習するので数分かかる。RUN_SLOW=1 のときだけ実行する。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "experiments/xs1/reranker-10S_w30-70__42.json"
slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="RUN_SLOW=1 のときだけ（数分かかる）")


@slow
def test_s0_reproduces_reference(tmp_path):
    out = tmp_path / "s0.json"
    subprocess.run([sys.executable, "run_var_description.py", "--setting", "A", "--conds", "S0",
                    "--seed-list", "42", "--output", str(out)], cwd=ROOT, check=True)
    got = json.load(open(out))["results"]["S0"]["per_seed"][0]
    ref = json.load(open(REF))["results"]["reranker-10S"]["per_seed"][0]
    assert got["n_cases"] == ref["n_cases"] == 423
    for k in ("Recall@K_correct", "MAP", "Recall@20"):
        assert abs(got[k] - ref[k]) <= 1e-4, (k, got[k], ref[k])


@slow
def test_s0_coverage_reproduces_a1(tmp_path):
    out = tmp_path / "cov.json"
    subprocess.run([sys.executable, "run_var_description.py", "--setting", "A", "--conds", "S0",
                    "--coverage", "--output", str(out)], cwd=ROOT, check=True)
    cov = json.load(open(out))["coverage"]["S0"]
    assert abs(cov["w30-70"] - 0.9670508667903073) < 1e-9
    assert abs(cov["w70-30"] - 0.9202866372367196) < 1e-9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `RUN_SLOW=1 python3 -m pytest tests/test_run_var_description.py -q -x`
Expected: FAIL, because `subprocess.CalledProcessError` reports that `run_var_description.py` does not exist (`can't open file`).

- [ ] **Step 3: Write the implementation**

```python
# run_var_description.py
#!/usr/bin/env python3
"""変数を記号でなく説明で照合する実験の学習と評価.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.5
条件: S0（元の説明文・記号）, S1（記号なしの説明文・記号）, D1（言い換え）, D0（正解式の説明をそのまま）。
S1 の模型は、テストケースの入出力の一部を別の記号に替えた S1-r{50,100} でも評価する。
学習ループは set_aware_reranker.run_mode（reranker-10S, 静的）と同じ順序で乱数を消費するので、
S0 は既報（experiments/xs1/reranker-10S_w30-70__42.json）を再現する。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from two_stage_query_conditioned import Reranker, norm, in_vars, out_vars  # noqa: E402
from set_aware_reranker import stratified_src_split  # noqa: E402
from evaluate_multi_eq import compute_all_ranks, case_metrics, aggregate_metrics  # noqa: E402
from var_matching import (  # noqa: E402
    SYMBOL_TAU, K_REF, occurrence_incidence, stage1_match_counts, jaccard_from_counts,
    youden_tau, domain_flags, case_features,
)
from var_conditions import (  # noqa: E402
    CONDITIONS, load_data, load_var_descriptions, build_views, make_matcher,
    calibration_pairs, pair_sims,
)

SPEC = "docs/superpowers/specs/2026-10-01-variable-description-matching-design.md"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
HP = dict(top_k=50, epochs=15, lr=1e-3, hidden_dim=64, margin=0.1, batch_size=16,
          n_neg_samples=8, weight_decay=1e-4)
SUMMARY_KEYS = ["MRR_first", "MRR_worst", "MRR_avg", "MAP",
                "Precision@C", "Recall@K_correct",
                "FullRecall@3", "FullRecall@10",
                "Recall@3", "Recall@5", "Recall@10", "Recall@20",
                "multi_only__MRR_first", "multi_only__MRR_worst",
                "multi_only__MAP", "multi_only__FullRecall@3", "multi_only__FullRecall@10",
                "multi_only__Recall@3", "multi_only__Recall@10",
                "multi_only__Recall@K_correct"]
COV_WEIGHTS = [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]


def wlabel(w) -> str:
    return f"w{int(round(w[0] * 100)):02d}-{int(round(w[1] * 100)):02d}"


def tau_for(matcher, views, data, idx, tau_delta=0.0) -> float:
    """記号版は SYMBOL_TAU。説明版は idx のケースの組で Youden の J が最大の τ（＋感度用の差分）."""
    if matcher.kind == "symbol":
        return SYMBOL_TAU
    pr = calibration_pairs(idx, views, data)
    return youden_tau(pair_sims(pr, matcher), pr["label"]) + tau_delta


def stage1(ci, view, matcher, tau, X_ctx, X_eq, data, inc, w_text, w_var, top_k):
    """第 1 段: (候補, 文章類似度 ts, 近似 |M|)."""
    from sklearn.metrics.pairwise import cosine_similarity
    ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
    full = matcher.q_occ(ci)
    m_approx = stage1_match_counts(full >= tau, inc)
    vs = jaccard_from_counts(m_approx, len(view.syms), data["n_vars"])
    cands = np.argsort(-(w_text * ts + w_var * vs))[:top_k].tolist()
    return cands, ts, full, m_approx


def prepare(idx, views, matcher, tau, X_ctx, X_eq, svd_sim, data, inc, w_text, w_var,
            top_k=HP["top_k"]):
    """各ケースの (候補, 特徴量) と、第 1 段の近似と厳密な |M| が食い違う割合."""
    eq_ptr, n_vars, doms = data["eq_ptr"], data["n_vars"], data["eq_domains"]
    prep, mismatch, total = {}, 0, 0
    for ci in idx:
        v = views[ci]
        cands, ts, full, m_approx = stage1(ci, v, matcher, tau, X_ctx, X_eq, data, inc,
                                           w_text, w_var, top_k)
        occ = np.concatenate([np.arange(eq_ptr[j], eq_ptr[j + 1]) for j in cands]).astype(np.int64)
        cptr = np.concatenate([[0], np.cumsum([n_vars[j] for j in cands])]).astype(np.int64)
        ref_end = int(cptr[min(K_REF, len(cands))])
        feats, m_exact = case_features(
            ts[cands], svd_sim[ci][cands], domain_flags(cands, doms, v.ctx), v.q_in, v.q_out,
            full[:, occ], matcher.occ_occ(occ, occ[:ref_end]), cptr, [doms[j] for j in cands], tau)
        mismatch += int((m_approx[cands] != m_exact).sum())
        total += len(cands)
        prep[ci] = (cands, feats)
    return prep, (mismatch / total if total else 0.0)


def train(seed, tr, prep, correct_lists, n_feat=10):
    """set_aware_reranker.run_mode の静的 reranker と同じ学習（同じ順序で乱数を消費する）."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Reranker(n_feat, HP["hidden_dim"])
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    rng = random.Random(seed)
    tr = list(tr)
    for _ in range(HP["epochs"]):
        rng.shuffle(tr)
        for s in range(0, len(tr), HP["batch_size"]):
            losses = []
            for ci in tr[s:s + HP["batch_size"]]:
                corr = set(correct_lists[ci])
                if not corr:
                    continue
                cands, feats = prep[ci]
                pos = [j for j in cands if j in corr]
                neg = [j for j in cands if j not in corr]
                if not pos or not neg:
                    continue
                c2k = {j: k for k, j in enumerate(cands)}
                scores = model(torch.tensor(feats, dtype=torch.float32))
                chosen = rng.sample(neg, min(HP["n_neg_samples"], len(neg)))
                for p in pos:
                    for ng in chosen:
                        losses.append(F.relu(HP["margin"] - scores[c2k[p]] + scores[c2k[ng]]))
            if losses:
                loss = torch.stack(losses).mean()
                opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def evaluate(model, te, prep, data) -> list:
    cases, cl, ek = data["cases"], data["correct_lists"], data["eq_keys"]
    out = []
    with torch.no_grad():
        for ci in te:
            corr = set(cl[ci])
            if not corr:
                continue
            cands, feats = prep[ci]
            scores = model(torch.tensor(feats, dtype=torch.float32)).numpy().ravel()
            order = sorted(range(len(cands)), key=lambda k: -scores[k])
            cm = case_metrics(compute_all_ranks([cands[k] for k in order], corr, miss_rank=10_000))
            c = cases[ci]
            cm["variant"] = norm(c.get("variant_type") or "?")
            cm["case_id"] = c.get("case_id", f"idx_{ci}")
            cm["n_input"] = len(in_vars(c))
            cm["n_output"] = len(out_vars(c))
            cm["n_sources"] = len({ek[j].split("__")[0] for j in corr})
            out.append(cm)
    return out


def per_case_record(seed, label, cm) -> dict:
    rec = {"seed": seed, "mode": label, **cm}
    rec.pop("FullRecall", None); rec.pop("Recall", None)
    return rec


def summarize(label, aggs, per_case=None) -> dict:
    out = {"mode": label, "n_features": 10}
    for k in SUMMARY_KEYS:
        vals = [a.get(k) for a in aggs if a.get(k) is not None]
        if vals:
            out[k] = {"mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0, 4)}
    out["per_seed"] = aggs
    if per_case is not None:
        out["per_case"] = per_case
    return out


def run_condition(cond, data, vd, seeds, renames, tau_delta, w_text, w_var, save_per_case,
                  tfidf, X_eq):
    from sklearn.decomposition import TruncatedSVD
    cs, cl = data["case_srcs"], data["correct_lists"]
    inc = occurrence_incidence(data["eq_ptr"])
    views = build_views(cond, data, vd)
    matcher = make_matcher(cond, data, views)
    X_ctx = tfidf.transform([v.ctx for v in views])
    main_label = cond if tau_delta == 0 else f"{cond}-tau{tau_delta:+.2f}"
    rviews = ({f"S1-r{int(round(p * 100))}": build_views("S1", data, vd, p) for p in renames}
              if cond == "S1" else {})
    labels = [main_label] + list(rviews)
    aggs = {lab: [] for lab in labels}
    pcs = {lab: [] for lab in labels} if save_per_case else None
    taus, mism = {}, {}
    for seed in seeds:
        t0 = time.time()
        split = stratified_src_split(cs, data["feats"], seed)
        tr = [i for i, s in enumerate(cs) if s in split["train"] and cl[i]]
        te = [i for i, s in enumerate(cs) if s in split["test"] and cl[i]]
        svd = TruncatedSVD(n_components=256, random_state=seed)
        E = svd.fit_transform(X_eq)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

        def svd_sim_of(vs_):
            Q = svd.transform(tfidf.transform([v.io_text for v in vs_]))
            Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            return Q @ E.T

        tau = tau_for(matcher, views, data, tr, tau_delta)
        taus[str(seed)] = tau
        prep, mism[str(seed)] = prepare(tr + te, views, matcher, tau, X_ctx, X_eq,
                                        svd_sim_of(views), data, inc, w_text, w_var)
        model = train(seed, tr, prep, cl)
        evals = {main_label: evaluate(model, te, prep, data)}
        for lab, rv in rviews.items():
            rprep, _ = prepare(te, rv, make_matcher("S1", data, rv), SYMBOL_TAU, X_ctx, X_eq,
                               svd_sim_of(rv), data, inc, w_text, w_var)
            evals[lab] = evaluate(model, te, rprep, data)
        for lab, res in evals.items():
            agg = aggregate_metrics(res)
            agg["seed"] = seed
            aggs[lab].append(agg)
            if pcs is not None:
                pcs[lab] += [per_case_record(seed, lab, cm) for cm in res]
            print(f"  [seed={seed}] {lab:12s} R@K={agg['Recall@K_correct']:.4f} MAP={agg['MAP']:.4f} "
                  f"R@20={agg['Recall@20']:.4f} n={agg['n_cases']} tau={tau:.4f} "
                  f"mismatch={mism[str(seed)]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    results = {lab: summarize(lab, aggs[lab], pcs[lab] if pcs is not None else None) for lab in labels}
    return results, {main_label: taus}, {main_label: mism}


def coverage_sweep(conds, data, vd, tfidf, X_eq, k=50, seed=42) -> dict:
    """第 1 段の重みを変えたときの被覆率（上位 k 件に入る正解式の割合）。説明版の τ は seed の学習用ケースで決める."""
    from sklearn.metrics.pairwise import cosine_similarity
    cs, cl = data["case_srcs"], data["correct_lists"]
    inc = occurrence_incidence(data["eq_ptr"])
    split = stratified_src_split(cs, data["feats"], seed)
    tr = [i for i, s in enumerate(cs) if s in split["train"] and cl[i]]
    out = {}
    for cond in conds:
        views = build_views(cond, data, vd)
        matcher = make_matcher(cond, data, views)
        X_ctx = tfidf.transform([v.ctx for v in views])
        tau = tau_for(matcher, views, data, tr)
        acc = {wlabel(w): [] for w in COV_WEIGHTS}
        for ci in [i for i in range(len(cl)) if cl[i]]:
            ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
            vs = jaccard_from_counts(stage1_match_counts(matcher.q_occ(ci) >= tau, inc),
                                     len(views[ci].syms), data["n_vars"])
            corr = set(cl[ci])
            for w in COV_WEIGHTS:
                top = set(np.argsort(-(w[0] * ts + w[1] * vs))[:k].tolist())
                acc[wlabel(w)].append(len(corr & top) / len(corr))
        out[cond] = {"tau": tau, **{lab: float(np.mean(v)) for lab, v in acc.items()}}
        print(f"  coverage {cond}: " + "  ".join(f"{lab}={out[cond][lab]:.4f}" for lab in acc), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["A", "B"], required=True)
    ap.add_argument("--conds", default="S0,S1,D1,D0")
    ap.add_argument("--seed-list", default=None, help="カンマ区切り（既定: 正典の 10 seed）")
    ap.add_argument("--renames", default="0.5,1.0", help="S1 の模型で評価する付け替えの割合")
    ap.add_argument("--tau-delta", type=float, default=0.0, help="説明版の τ に足す値（感度の確認用）")
    ap.add_argument("--w-text", type=float, default=0.3)
    ap.add_argument("--w-var", type=float, default=0.7)
    ap.add_argument("--coverage", action="store_true", help="学習せず、第 1 段の被覆率の掃引だけ出す")
    ap.add_argument("--save-per-case", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer

    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    assert all(c in CONDITIONS for c in conds), conds
    seeds = [int(s) for s in args.seed_list.split(",")] if args.seed_list else SEEDS
    renames = [float(p) for p in args.renames.split(",") if p.strip()]
    data = load_data(args.setting)
    vd = load_var_descriptions() if any(c != "S0" for c in conds) else None
    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(data["eq_texts"])
    config = {**vars(args), **HP, "spec": SPEC, "n_cases": len(data["cases"])}
    if args.coverage:
        out = {"config": config, "coverage": coverage_sweep(conds, data, vd, tfidf, X_eq)}
    else:
        results, taus, mism = {}, {}, {}
        for cond in conds:
            print(f"\n=== setting {args.setting} / {cond} ===", flush=True)
            r, t, m = run_condition(cond, data, vd, seeds, renames, args.tau_delta, args.w_text,
                                    args.w_var, args.save_per_case, tfidf, X_eq)
            results.update(r); taus.update(t); mism.update(m)
        out = {"config": config, "tau": taus, "approx_mismatch": mism, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the regression tests to verify they pass**

Run: `RUN_SLOW=1 python3 -m pytest tests/test_run_var_description.py -q`
Expected: `2 passed` (several minutes). The S0 per-seed line prints `R@K=0.8403 MAP=0.9241 R@20=0.9648 n=423`.
If R@K differs, the RNG order or the features differ from `set_aware_reranker.run_mode`. Compare the first training case's `cands` and `feats` against a direct call of `set_aware_reranker.stage1` and `compute_features_with_set` before changing anything else.

- [ ] **Step 5: Checkpoint (no commit)**

Run: `python3 -m pytest tests/test_var_matching.py tests/test_var_desc_qc.py tests/test_var_conditions.py tests/test_run_var_description.py -q`
Expected: `20 passed, 2 skipped` (the slow tests skip without `RUN_SLOW=1`).

---

### Task 5: API generation script and 20-case pilot (`make_var_descriptions.py`)

**Files:**
- Create: `make_var_descriptions.py`
- Test: `tests/test_make_var_descriptions.py`
- Modify: `.gitignore` (append one line)

**Interfaces:**
- Consumes: Task 2 (`check_output`, `word_jaccard`). From existing code: `two_stage_query_conditioned.load_equations, load_cases, norm, eq_key`, `external_baselines.SETTINGS, variant_matches`.
- Produces:
  - `case_payload(case: dict, eqk: dict) -> {"context", "variables": [{"symbol", "role", "descriptions"}]}` (variables ordered inputs then outputs, unique; descriptions in `correct_model_ids` order, unique)
  - `request_params(payload) -> dict` (model, max_tokens, system, messages, output_config)
  - `parse_message(msg) -> (dict | None, str | None)`
  - `pick(records: list[dict], payload) -> (record | None, qc_failed: bool)`
  - Raw records appended to `experiments/var_descriptions_raw/{pilot,batch,retry}.jsonl`: `{case_id, source, prompt_version, model_requested, model_served, stop_reason, usage, output, error, time}`
  - Final `experiments/var_descriptions.json` = `{"meta": {...}, "cases": {case_id: {"context_nosym", "variables": {symbol: {"paraphrase", "alt_symbol", "meaning_conflict"}}, "model_served", "source", "qc_failed", "qc_hard", "qc_review"}}}` (consumed by `var_conditions.load_var_descriptions`); `experiments/var_descriptions_qc.json` (key `meaning_conflict_ratio` consumed by Task 9); `experiments/var_descriptions_sample50.csv`.

- [ ] **Step 1: Write the failing offline tests**

```python
# tests/test_make_var_descriptions.py
"""make_var_descriptions.py のオフラインの単体テスト（API は呼ばない）.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.2
"""
import json
from types import SimpleNamespace

from make_var_descriptions import (
    MODEL, OUTPUT_SCHEMA, case_payload, request_params, parse_message, pick,
)

CASE = {"case_id": "toy_1", "context": "Reactor sharing $T$.",
        "input_variables": ["T", "F"], "output_variables": ["C_A"],
        "correct_model_ids": ["src__eq_2", "src__eq_1"], "variant_type": "original"}
EQK = {"src__eq_1": {"variables": {"T": "temperature in the tank [K]", "F": "feed flow"}},
       "src__eq_2": {"variables": {"C_A": "concentration of A", "T": "reactor temperature"}}}


def test_case_payload_order_roles_descriptions():
    p = case_payload(CASE, EQK)
    assert p["context"] == "Reactor sharing $T$."
    assert [v["symbol"] for v in p["variables"]] == ["T", "F", "C_A"]
    assert [v["role"] for v in p["variables"]] == ["input", "input", "output"]
    assert p["variables"][0]["descriptions"] == ["reactor temperature", "temperature in the tank [K]"]


def test_request_params_uses_opus_effort_and_schema():
    prm = request_params(case_payload(CASE, EQK))
    assert prm["model"] == MODEL == "claude-opus-5-5"
    assert prm["output_config"]["effort"] == "medium"
    assert prm["output_config"]["format"] == {"type": "json_schema", "schema": OUTPUT_SCHEMA}
    assert not {"thinking", "temperature", "tool_choice"} & set(prm)
    assert json.loads(prm["messages"][0]["content"])["variables"][2]["symbol"] == "C_A"


def _msg(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, model=MODEL,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=20))


def test_parse_message():
    assert parse_message(_msg('{"a": 1}')) == ({"a": 1}, None)
    assert parse_message(_msg("", "refusal")) == (None, "refusal")
    out, err = parse_message(_msg("not json"))
    assert out is None and err.startswith("json")


GOOD = {"context_nosym": "Reactor sharing the temperature.",
        "variables": [
            {"symbol": "T", "paraphrase": "temperature of the liquid in the reactor",
             "alt_symbol": "\\theta", "meaning_conflict": False},
            {"symbol": "F", "paraphrase": "volumetric feed rate", "alt_symbol": "q", "meaning_conflict": False},
            {"symbol": "C_A", "paraphrase": "amount of species A per volume",
             "alt_symbol": "c_A", "meaning_conflict": False}]}


def test_pick_prefers_latest_passing_record():
    payload = case_payload(CASE, EQK)
    bad = {**GOOD, "context_nosym": "has $T$"}
    rec, failed = pick([{"output": GOOD, "time": "1"}, {"output": bad, "time": "2"}], payload)
    assert rec["time"] == "1" and failed is False
    rec, failed = pick([{"output": bad, "time": "2"}], payload)
    assert rec["time"] == "2" and failed is True
    rec, failed = pick([{"output": None, "time": "3"}], payload)
    assert rec is None and failed is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_make_var_descriptions.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'make_var_descriptions'`

- [ ] **Step 3: Write the implementation**

```python
# make_var_descriptions.py
#!/usr/bin/env python3
"""ケース側の変数の言い換え・記号なしの説明文・別の記号を Anthropic API で生成する.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.2
このセッション（Claude Code）の出力では作らない。生成はすべて API（モデルは本人が確定した Opus 5.5）。

  python3 make_var_descriptions.py pilot --n 20   # 通常の API で 20 件。品質・トークン数・料金の見積もり
  python3 make_var_descriptions.py submit         # 合格の記録がないケースを Message Batches API で送る
  python3 make_var_descriptions.py collect        # バッチが終わるのを待って結果を取り込む
  python3 make_var_descriptions.py retry          # 不合格・失敗のケースを通常の API で作り直す（1 ケース最大 3 回）
  python3 make_var_descriptions.py finalize       # 品質の確認 → var_descriptions.json・_qc.json・_sample50.csv

生の応答は experiments/var_descriptions_raw/*.jsonl に追記する（追跡しない）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)
import anthropic  # noqa: E402

from two_stage_query_conditioned import load_equations, load_cases, norm, eq_key  # noqa: E402
from external_baselines import SETTINGS, variant_matches  # noqa: E402
from var_desc_qc import check_output, word_jaccard  # noqa: E402

EXP = ROOT / "experiments"
RAW_DIR = EXP / "var_descriptions_raw"
OUT_JSON = EXP / "var_descriptions.json"
QC_JSON = EXP / "var_descriptions_qc.json"
SAMPLE_CSV = EXP / "var_descriptions_sample50.csv"
SPEC = "docs/superpowers/specs/2026-10-01-variable-description-matching-design.md"
MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
PROMPT_VERSION = "v1"
MAX_ATTEMPTS = 3   # 最初の 1 回 ＋ 作り直し 2 回
PRICE = {"standard": (4.0, 20.0), "batch": (2.0, 10.0)}  # USD / 100 万トークン（入力, 出力）。2026-09-25 時点
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You help build an evaluation set for retrieving equations from the literature.
Each request describes a modelling task: a short description ("context") and its input and output variables. Each variable comes with its symbol, its role, and one or more descriptions of that symbol taken from the source equations.

Rewrite the task as an engineer would state it without knowing the notation of the source papers. Return JSON with:

1. "context_nosym": the context with every mention of a variable symbol (for example "$T$", "C_A" or "\\alpha") replaced by plain words for that quantity. Keep all other content. Do not add information.

2. "variables": one entry for each given variable, with
   - "symbol": the symbol exactly as given.
   - "paraphrase": a description of the quantity in your own words, about 3 to 15 words. Do not use the symbol, any other symbol, or mathematical notation. Do not copy the given descriptions; reword them. Omit units. Keep every qualifier that distinguishes this quantity from similar ones, such as inlet, outlet, jacket, coolant, wall, initial, steady-state, reference, or the species or phase it refers to.
   - "alt_symbol": another symbol that a different textbook or engineer might use for the same quantity, in the same LaTeX style as the given symbols. It must differ from the given symbol, from every other given symbol, and from every other alt_symbol in this request.
   - "meaning_conflict": true only if the given descriptions of this symbol name different quantities. In that case, describe the quantity of the first description.

No two variables may share the same paraphrase."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "context_nosym": {"type": "string"},
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "paraphrase": {"type": "string"},
                    "alt_symbol": {"type": "string"},
                    "meaning_conflict": {"type": "boolean"},
                },
                "required": ["symbol", "paraphrase", "alt_symbol", "meaning_conflict"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["context_nosym", "variables"],
    "additionalProperties": False,
}


def setting_a_cases() -> list:
    return [c for c in load_cases() if variant_matches(c.get("variant_type", ""), SETTINGS["A"])]


def eq_index() -> dict:
    return {eq_key(e): e for e in load_equations() if eq_key(e)}


def case_payload(case: dict, eqk: dict) -> dict:
    """LLM に渡す内容（式の LaTeX と ID は渡さない）."""
    ins = [norm(v) for v in case.get("input_variables") or [] if norm(v)]
    outs = [norm(v) for v in case.get("output_variables") or [] if norm(v)]
    variables = []
    for s in dict.fromkeys(ins + outs):
        descs = []
        for m in case.get("correct_model_ids") or []:
            e = eqk.get(norm(m))
            if not e:
                continue
            for k, v in (e.get("variables") or {}).items():
                if norm(k) == s and isinstance(v, str) and norm(v) and norm(v) not in descs:
                    descs.append(norm(v))
        role = "input and output" if (s in ins and s in outs) else ("input" if s in ins else "output")
        variables.append({"symbol": s, "role": role, "descriptions": descs})
    return {"context": norm(case.get("context") or ""), "variables": variables}


def request_params(payload: dict) -> dict:
    return dict(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )


def parse_message(msg) -> tuple:
    """(出力 dict or None, エラー文字列 or None)."""
    if msg.stop_reason == "refusal":
        return None, "refusal"
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if not text:
        return None, f"empty ({msg.stop_reason})"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"json: {e}"


def record(case_id: str, source: str, msg=None, error: str | None = None) -> dict:
    out, err = parse_message(msg) if msg is not None else (None, error)
    return {
        "case_id": case_id, "source": source, "prompt_version": PROMPT_VERSION,
        "model_requested": MODEL, "model_served": getattr(msg, "model", None),
        "stop_reason": getattr(msg, "stop_reason", None),
        "usage": ({"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
                  if msg is not None else None),
        "output": out, "error": err,
        "time": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }


def append_raw(name: str, rec: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_raw() -> list:
    recs = []
    for p in sorted(RAW_DIR.glob("*.jsonl")):
        recs += [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    return sorted(recs, key=lambda r: r["time"])


def call_standard(client, params: dict, use_fallbacks: bool) -> tuple:
    """通常の API で 1 件（429・5xx・接続エラーは待って最大 4 回）。(メッセージ or None, エラー or None)."""
    last = None
    for attempt in range(4):
        try:
            if use_fallbacks:
                return client.beta.messages.create(**params, betas=[FALLBACK_BETA],
                                                   extra_body={"fallbacks": "default"}), None
            return client.messages.create(**params), None
        except anthropic.RateLimitError as e:
            last = e
            time.sleep(20 * (attempt + 1))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last = e
                time.sleep(5 * (attempt + 1))
                continue
            return None, f"{e.status_code}: {e.message}"
        except anthropic.APIConnectionError as e:
            last = e
            time.sleep(5 * (attempt + 1))
    return None, f"gave up: {last}"


def judge(rec: dict, payload: dict) -> tuple:
    """(合格か, hard の問題, 目視の印)."""
    hard, review = check_output(payload, rec.get("output"))
    return (rec.get("output") is not None and not hard), hard, review


def by_case(recs: list, payloads: dict) -> dict:
    out = {}
    for r in recs:
        if r["case_id"] in payloads:
            out.setdefault(r["case_id"], []).append(r)
    return out


def pick(rs: list, payload: dict) -> tuple:
    """合格の最新記録。なければ記号の集合がそろった最新記録（不合格）。どちらもなければ None."""
    ok = [r for r in rs if judge(r, payload)[0]]
    if ok:
        return ok[-1], False
    full = [r for r in rs if r.get("output")
            and not any(h.startswith("symbol-set|") for h in judge(r, payload)[1])]
    return (full[-1], True) if full else (None, True)


def client_() -> "anthropic.Anthropic":
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def payloads_all() -> dict:
    eqk = eq_index()
    return {c["case_id"]: case_payload(c, eqk) for c in setting_a_cases()}


def cmd_pilot(args) -> None:
    payloads = payloads_all()
    ids = random.Random(20261001).sample(sorted(payloads), args.n)
    client = client_()
    tin = tout = n_ok = 0
    for cid in ids:
        msg, err = call_standard(client, request_params(payloads[cid]), not args.no_fallbacks)
        rec = record(cid, "pilot", msg, err)
        append_raw("pilot", rec)
        ok, hard, review = judge(rec, payloads[cid])
        n_ok += ok
        if rec["usage"]:
            tin += rec["usage"]["input_tokens"]
            tout += rec["usage"]["output_tokens"]
        print(f"  {cid}: {'OK' if ok else 'NG'} served={rec['model_served']} usage={rec['usage']} "
              f"err={rec['error']} hard={hard[:3]} review={review[:3]}", flush=True)
    n, N = len(ids), len(payloads)
    pin, pout = PRICE["batch"]
    est_in, est_out = tin / n * N, tout / n * N
    print(f"pilot: {n_ok}/{n} 合格。1 件あたり平均トークン 入力 {tin / n:.0f}・出力 {tout / n:.0f}")
    print(f"全 {N} 件をバッチで送る見積もり: 入力 {est_in / 1e6:.2f}M・出力 {est_out / 1e6:.2f}M → "
          f"約 {est_in / 1e6 * pin + est_out / 1e6 * pout:.1f} USD（{MODEL}, バッチ単価 入力 {pin}・出力 {pout} USD/100万）")


def cmd_submit(args) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    payloads = payloads_all()
    groups = by_case(load_raw(), payloads)
    todo = [cid for cid in sorted(payloads) if not any(judge(r, payloads[cid])[0] for r in groups.get(cid, []))]
    if not todo:
        print("送るケースはない")
        return
    batch = client_().messages.batches.create(requests=[
        Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**request_params(payloads[cid])))
        for cid in todo])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = RAW_DIR / "batches.json"
    batches = json.load(open(log)) if log.exists() else []
    batches.append({"id": batch.id, "n": len(todo), "collected": False,
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    json.dump(batches, open(log, "w"), indent=1)
    print(f"submitted batch {batch.id}: {len(todo)} requests")


def cmd_collect(args) -> None:
    log = RAW_DIR / "batches.json"
    batches = json.load(open(log))
    client = client_()
    for b in batches:
        if b["collected"]:
            continue
        while True:
            cur = client.messages.batches.retrieve(b["id"])
            if cur.processing_status == "ended":
                break
            print(f"  {b['id']}: {cur.processing_status} (processing={cur.request_counts.processing})", flush=True)
            time.sleep(60)
        n = Counter()
        for r in client.messages.batches.results(b["id"]):
            if r.result.type == "succeeded":
                rec = record(r.custom_id, "batch", r.result.message)
            else:
                rec = record(r.custom_id, "batch", None, f"batch {r.result.type}")
            append_raw("batch", rec)
            n[r.result.type] += 1
        b["collected"] = True
        json.dump(batches, open(log, "w"), indent=1)
        print(f"collected {b['id']}: {dict(n)}")


def cmd_retry(args) -> None:
    payloads = payloads_all()
    groups = by_case(load_raw(), payloads)
    todo = [cid for cid in sorted(payloads)
            if not any(judge(r, payloads[cid])[0] for r in groups.get(cid, []))
            and len(groups.get(cid, [])) < MAX_ATTEMPTS]
    print(f"retry: {len(todo)} cases")
    client = client_()
    for cid in todo:
        for _ in range(MAX_ATTEMPTS - len(groups.get(cid, []))):
            msg, err = call_standard(client, request_params(payloads[cid]), not args.no_fallbacks)
            rec = record(cid, "retry", msg, err)
            append_raw("retry", rec)
            ok, hard, _ = judge(rec, payloads[cid])
            print(f"  {cid}: {'OK' if ok else 'NG'} err={rec['error']} hard={hard[:3]}", flush=True)
            if ok:
                break


def quantiles(a: list) -> dict:
    import numpy as np
    x = np.asarray(a, dtype=float)
    return {"mean": round(float(x.mean()), 4),
            **{f"p{q}": round(float(np.percentile(x, q)), 4) for q in (10, 50, 90)}}


def cmd_finalize(args) -> None:
    payloads = payloads_all()
    raw = load_raw()
    groups = by_case(raw, payloads)
    out_cases, missing, failed = {}, [], []
    hard_counts, review_counts, served = Counter(), Counter(), Counter()
    overlaps, n_vars, n_conflict = [], 0, 0
    for cid in sorted(payloads):
        rec, qc_failed = pick(groups.get(cid, []), payloads[cid])
        if rec is None:
            missing.append(cid)
            continue
        hard, review = check_output(payloads[cid], rec["output"])
        if qc_failed:
            failed.append(cid)
        hard_counts.update(h.split("|")[0] for h in hard)
        review_counts.update(r.split("|")[0] for r in review)
        served[rec["model_served"]] += 1
        vars_out = {v["symbol"]: {"paraphrase": v["paraphrase"], "alt_symbol": v["alt_symbol"],
                                  "meaning_conflict": bool(v["meaning_conflict"])}
                    for v in rec["output"]["variables"]}
        for pv in payloads[cid]["variables"]:
            v = vars_out.get(pv["symbol"])
            if v is None:
                continue
            n_vars += 1
            n_conflict += v["meaning_conflict"]
            overlaps.append(max((word_jaccard(v["paraphrase"], d) for d in pv["descriptions"]), default=0.0))
        out_cases[cid] = {"context_nosym": rec["output"]["context_nosym"], "variables": vars_out,
                          "model_served": rec["model_served"], "source": rec["source"],
                          "qc_failed": qc_failed, "qc_hard": hard, "qc_review": review}
    usage = {}
    for r in raw:
        if not r.get("usage"):
            continue
        u = usage.setdefault(r["source"], {"input_tokens": 0, "output_tokens": 0, "n": 0})
        u["input_tokens"] += r["usage"]["input_tokens"]
        u["output_tokens"] += r["usage"]["output_tokens"]
        u["n"] += 1
    for src, u in usage.items():
        pin, pout = PRICE["batch" if src == "batch" else "standard"]
        u["usd"] = round(u["input_tokens"] / 1e6 * pin + u["output_tokens"] / 1e6 * pout, 2)
    meta = {"model": MODEL, "effort": EFFORT, "prompt_version": PROMPT_VERSION, "spec": SPEC,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_cases": len(out_cases), "n_missing": len(missing), "n_qc_failed": len(failed),
            "served_models": dict(served)}
    json.dump({"meta": meta, "cases": out_cases}, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    qc = {"n_cases_setting_A": len(payloads), "missing": missing, "qc_failed": failed,
          "hard_issue_counts": dict(hard_counts), "review_flag_counts": dict(review_counts),
          "n_variables": n_vars, "meaning_conflict": n_conflict,
          "meaning_conflict_ratio": n_conflict / n_vars if n_vars else 0.0,
          "word_overlap_with_source": quantiles(overlaps) if overlaps else None,
          "served_models": dict(served), "usage": usage,
          "usd_total": round(sum(u["usd"] for u in usage.values()), 2)}
    json.dump(qc, open(QC_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pool = [(cid, pv) for cid in sorted(out_cases) for pv in payloads[cid]["variables"]
            if pv["symbol"] in out_cases[cid]["variables"]]
    with open(SAMPLE_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "symbol", "role", "source_descriptions", "paraphrase", "alt_symbol",
                    "meaning_conflict", "meaning_preserved", "note"])
        for cid, pv in random.Random(50).sample(pool, min(50, len(pool))):
            v = out_cases[cid]["variables"][pv["symbol"]]
            w.writerow([cid, pv["symbol"], pv["role"], " | ".join(pv["descriptions"]), v["paraphrase"],
                        v["alt_symbol"], v["meaning_conflict"], "", ""])
    print(json.dumps({k: qc[k] for k in ("hard_issue_counts", "review_flag_counts", "meaning_conflict_ratio",
                                         "word_overlap_with_source", "served_models", "usd_total")},
                     ensure_ascii=False, indent=1))
    print(f"cases={len(out_cases)} missing={len(missing)} qc_failed={len(failed)} → {OUT_JSON}")
    if missing:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--no-fallbacks", action="store_true")
    sub.add_parser("submit")
    sub.add_parser("collect")
    r = sub.add_parser("retry")
    r.add_argument("--no-fallbacks", action="store_true")
    sub.add_parser("finalize")
    args = ap.parse_args()
    {"pilot": cmd_pilot, "submit": cmd_submit, "collect": cmd_collect,
     "retry": cmd_retry, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the offline tests to verify they pass**

Run: `python3 -m pytest tests/test_make_var_descriptions.py -q`
Expected: `4 passed`

- [ ] **Step 5: Ignore the raw-response directory**

Append this line to `.gitignore`:

```
experiments/var_descriptions_raw/
```

Run: `git check-ignore experiments/var_descriptions_raw/x.jsonl`
Expected: prints `experiments/var_descriptions_raw/x.jsonl`

- [ ] **Step 6: Run the 20-case pilot (real API, standard price, about 1 USD)**

Run: `python3 make_var_descriptions.py pilot --n 20`
Expected: 20 lines `toy…: OK/NG served=claude-opus-5-5 …`, then `pilot: k/20 合格…` and a line with the batch cost estimate.
If any line shows `err=400:` mentioning `fallbacks`, the API rejected the fallback parameter. Rerun with `--no-fallbacks` and note this in the report. Responses that pass QC are not resent later, because `submit` skips cases with a passing record.

- [ ] **Step 7: STOP — report to the user and wait for an explicit OK**

Report in Japanese: the pass count (k/20); the hard-issue codes seen; 3 example paraphrases next to their source descriptions (copied from `experiments/var_descriptions_raw/pilot.jsonl`); the average tokens; the batch cost estimate; the serving models. Ask: 「この見積もりでバッチを送ってよいですか？」. **Do not run `submit` before the user answers yes.** If many pilot cases fail the same check, propose a prompt change (bump `PROMPT_VERSION` to `v2`) before the batch.

---

### Task 6: Batch generation, retries, finalize, manual check (after the user's OK)

**Files:**
- Uses: `make_var_descriptions.py` (Task 5)
- Produces: `experiments/var_descriptions.json`, `experiments/var_descriptions_qc.json`, `experiments/var_descriptions_sample50.csv`

**Interfaces:**
- Consumes: Task 5 CLI.
- Produces: the three files above (shapes in Task 5 Interfaces).

- [ ] **Step 1: Submit the batch**

Run: `python3 make_var_descriptions.py submit`
Expected: `submitted batch msgbatch_… : N requests` with N = 1,823 minus the passing pilot cases.

- [ ] **Step 2: Collect the results (waits until the batch ends; most batches end within 1 hour)**

Run: `python3 make_var_descriptions.py collect` (run it in the background and wait for it to finish; do not poll with sleep loops of your own)
Expected: `collected msgbatch_…: {'succeeded': …}`

- [ ] **Step 3: Retry failures with the standard API**

Run: `python3 make_var_descriptions.py retry`
Expected: `retry: M cases` followed by OK/NG lines. Each case gets at most 3 records in total.

- [ ] **Step 4: Finalize**

Run: `python3 make_var_descriptions.py finalize`
Expected: a JSON summary and `cases=1823 missing=0 qc_failed=…`. Exit code 0.
If `missing` > 0, run `retry` again (cases without 3 records get new attempts). If a case already has 3 failed records, report it to the user instead of looping.

- [ ] **Step 5: Manual meaning check of 50 variables (reading only; generation stays with the API)**

Read `experiments/var_descriptions_sample50.csv`. For each row, judge whether `paraphrase` names the same quantity as `source_descriptions`, including the distinguishing qualifiers (inlet, jacket, species, and so on). Fill `meaning_preserved` with `Y` or `N` and give a short reason in `note` for every `N`. Count the `N` rows.

- [ ] **Step 6: Checkpoint (no commit)**

Run: `python3 -c "import json; m=json.load(open('experiments/var_descriptions.json'))['meta']; print(m)"`
Expected: `n_cases` 1823, `n_missing` 0, `served_models` mostly `claude-opus-5-5` (report any other model and its count).

---

### Task 7: Pre-check of E5 separability with the AUC gate (`analyze_var_description.py precheck`)

**Files:**
- Create: `analyze_var_description.py`
- Test: `tests/test_analyze_var_description.py`

**Interfaces:**
- Consumes: Task 1 `youden_tau`; Task 3 `load_data, load_var_descriptions, build_views, make_matcher, calibration_pairs, pair_sims`.
- Produces: `experiments/var_desc_precheck.json` = `{"spec", "n_cases", "D1": {...}, "D0": {...}, "gate": {"metric", "threshold", "pass"}}`. Each condition entry has `n_pos, n_neg, auc, tau_all_cases, sim_pos, sim_neg, hardest_negatives[20], hardest_positives[20]`. `sim_pos` is the E5 similarity between each paraphrase and its source descriptions, the distribution that spec §3.2 check 2 asks for. Helpers `quantiles(a) -> dict` and `AUC_GATE = 0.80`. The process exits with code 2 when the gate fails.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyze_var_description.py
"""analyze_var_description.py の補助関数のテスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.3, §5
"""
import numpy as np

from analyze_var_description import AUC_GATE, quantiles


def test_quantiles_and_gate():
    q = quantiles(np.arange(101, dtype=float))
    assert q["mean"] == 50.0 and q["p5"] == 5.0 and q["p50"] == 50.0 and q["p95"] == 95.0
    assert AUC_GATE == 0.80
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_analyze_var_description.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'analyze_var_description'`

- [ ] **Step 3: Write the precheck part of the script**

```python
# analyze_var_description.py
#!/usr/bin/env python3
"""変数の説明による照合の分析: 本実験の前の確認（precheck）、検定（stats）、図（figure）.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.3, §3.6, §5
  python3 analyze_var_description.py precheck   # E5 が同じ変数と同じ式の中の別の変数を分ける AUC（< 0.80 なら終了コード 2）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
SPEC = "docs/superpowers/specs/2026-10-01-variable-description-matching-design.md"
PRECHECK_JSON = EXP / "var_desc_precheck.json"
AUC_GATE = 0.80


def quantiles(a) -> dict:
    x = np.asarray(a, dtype=float)
    return {"mean": round(float(x.mean()), 4),
            **{f"p{q}": round(float(np.percentile(x, q)), 4) for q in (5, 25, 50, 75, 95)}}


def cmd_precheck(args) -> None:
    from sklearn.metrics import roc_auc_score
    from var_matching import youden_tau
    from var_conditions import (load_data, load_var_descriptions, build_views, make_matcher,
                                calibration_pairs, pair_sims)
    data = load_data("A")
    vd = load_var_descriptions()
    idx = [i for i in range(len(data["cases"])) if data["correct_lists"][i]]
    out = {"spec": SPEC, "n_cases": len(idx)}
    for cond in ("D1", "D0"):
        views = build_views(cond, data, vd)
        matcher = make_matcher(cond, data, views)
        pr = calibration_pairs(idx, views, data)
        sims = pair_sims(pr, matcher)
        y = pr["label"]

        def row(t):
            ci, k, o = int(pr["case"][t]), int(pr["item"][t]), int(pr["occ"][t])
            return {"case_id": data["cases"][ci]["case_id"], "query_symbol": views[ci].orig_syms[k],
                    "query_text": views[ci].texts[k], "db_symbol": data["occ_sym"][o],
                    "db_description": data["occ_desc"][o],
                    "eq_key": data["eq_keys"][int(data["occ_eq"][o])], "sim": round(float(sims[t]), 4)}

        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        out[cond] = {
            "n_pos": int(len(pos)), "n_neg": int(len(neg)),
            "auc": round(float(roc_auc_score(y, sims)), 4),
            "tau_all_cases": round(youden_tau(sims, y), 4),
            "sim_pos": quantiles(sims[pos]), "sim_neg": quantiles(sims[neg]),
            "hardest_negatives": [row(t) for t in neg[np.argsort(-sims[neg])][:20]],
            "hardest_positives": [row(t) for t in pos[np.argsort(sims[pos])][:20]],
        }
        print(f"{cond}: AUC={out[cond]['auc']:.4f} pos={len(pos)} neg={len(neg)} "
              f"tau(all)={out[cond]['tau_all_cases']:.4f}", flush=True)
    out["gate"] = {"metric": "D1 AUC", "threshold": AUC_GATE, "pass": out["D1"]["auc"] >= AUC_GATE}
    json.dump(out, open(PRECHECK_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {PRECHECK_JSON}")
    if not out["gate"]["pass"]:
        print(f"STOP: D1 の AUC {out['D1']['auc']:.4f} < {AUC_GATE}。本実験の前に本人に報告して相談する（仕様 §3.3）")
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("precheck")
    args = ap.parse_args()
    {"precheck": cmd_precheck}[args.cmd](args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_analyze_var_description.py -q`
Expected: `1 passed`

- [ ] **Step 5: Run the pre-check (computes and caches the E5 embeddings: DB 27,994 texts and about 25k paraphrases, a few minutes on MPS)**

Run: `python3 analyze_var_description.py precheck`
Expected: `D1: AUC=… pos=… neg=…` and `D0: AUC=…`, then `Saved: experiments/var_desc_precheck.json`. The exit code is 0 when the D1 AUC is ≥ 0.80.
**If the exit code is 2: STOP.** Report the AUC and the five hardest negatives to the user, and wait before running Task 8.

---

### Task 8: Main runs (`run_var_description.sh`)

**Files:**
- Create: `run_var_description.sh`
- Produces: `experiments/var_desc_A.json`, `experiments/var_desc_B.json`, `experiments/var_desc_tau-0.02.json`, `experiments/var_desc_tau+0.02.json`, `experiments/var_desc_coverage.json`, logs in `experiments/var_desc_logs/`

**Interfaces:**
- Consumes: Task 4 CLI; Task 6 `var_descriptions.json`; Task 7 cached embeddings.
- Produces: result JSONs in the Task 4 output shape.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# 変数の説明による照合の本実験（仕様 docs/superpowers/specs/2026-10-01-variable-description-matching-design.md）
#   設定 A・B を並列に、条件 S0,S1(+S1-r50/r100),D1,D0 × 10 seed。
#   続けて τ±0.02（設定 A・seed 42・D1）と、第 1 段の重みの掃引による被覆率（設定 A）。
set -u
cd "$(dirname "$0")"
LOG=experiments/var_desc_logs
mkdir -p "$LOG"
for S in A B; do
  python3 run_var_description.py --setting "$S" --conds S0,S1,D1,D0 --renames 0.5,1.0 \
    --save-per-case --output "experiments/var_desc_${S}.json" > "$LOG/main_${S}.log" 2>&1 &
done
wait
for D in -0.02 +0.02; do
  python3 run_var_description.py --setting A --conds D1 --seed-list 42 --tau-delta "$D" \
    --output "experiments/var_desc_tau${D}.json" > "$LOG/tau${D}.log" 2>&1
done
python3 run_var_description.py --setting A --conds S0,S1,D1,D0 --coverage \
  --output experiments/var_desc_coverage.json > "$LOG/coverage.log" 2>&1
echo "ALL DONE"
```

Run: `chmod +x run_var_description.sh`

- [ ] **Step 2: Launch in the background and estimate the time**

Run: `./run_var_description.sh` with `run_in_background: true`. After the first `[seed=42]` line appears in `experiments/var_desc_logs/main_A.log`, read its elapsed seconds and report an ETA. That is about 4 conditions × 10 seeds × the per-seed time. Wait for the background-task notification; do not poll with sleep loops.

- [ ] **Step 3: Verify the outputs**

Run:
```bash
python3 - <<'EOF'
import json
for s in "AB":
    r = json.load(open(f"experiments/var_desc_{s}.json"))["results"]
    print(s, {k: (len(v["per_seed"]), v["Recall@K_correct"]["mean"]) for k, v in r.items()})
a = json.load(open("experiments/var_desc_A.json"))["results"]["S0"]["per_seed"]
print("S0 seed42 A:", [p["Recall@K_correct"] for p in a if p["seed"] == 42])
c = json.load(open("experiments/var_desc_coverage.json"))["coverage"]
print({k: (round(v["w30-70"], 4), round(v["tau"], 4)) for k, v in c.items()})
EOF
```
Expected: for both settings, labels `S0, S1, S1-r50, S1-r100, D1, D0`, each with 10 seeds. `S0 seed42 A: [0.8402538556793877]` (within 1e-4). Coverage S0 w30-70 = 0.9671. If the τ files exist, they hold `D1-tau-0.02` and `D1-tau+0.02`.

---

### Task 9: Paired tests and figure (`analyze_var_description.py stats` / `figure`)

**Files:**
- Modify: `analyze_var_description.py` (add functions and subcommands)
- Modify: `tests/test_analyze_var_description.py` (add tests)
- Produces: `experiments/var_desc_stats.json`, `figures/fig_var_description.pdf`, `figures/fig_var_description.png`

**Interfaces:**
- Consumes: Task 8 JSONs; Task 6 `var_descriptions_qc.json` (`meaning_conflict_ratio`) and `var_descriptions.json`; `analyze_significance.paired_stats(base, rer)`, which returns `{n, mean_base, mean_rer, mean_delta, std_delta, p_ttest, p_wilcoxon, cohen_dz}` with delta = rer − base.
- Produces: `SEEDS`, `COMPARISONS`, `METRICS`, `family(variant) -> str`, `per_seed(res, label, metric) -> list`, `ci95(d) -> [lo, hi]`, `verdict(st, ci) -> str`, `safe_paired(base, rer) -> dict`; `var_desc_stats.json` = `{"A"|"B": {"comparisons": {"D1_vs_S1": {"purpose", metric: {...paired_stats, "ci95", "verdict"}}, …}, "family": {...}, "family_tests": {...}, "tau": {...}, "approx_mismatch": {...}, "tau_sensitivity"?: {...}, "excluding_meaning_conflict"?: {...}}, "meaning_conflict": {...}}`.

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_analyze_var_description.py`:

```python
from analyze_var_description import ci95, verdict, family


def test_ci95_and_verdict():
    d = [0.01, -0.01, 0.0, 0.005, -0.005, 0.0, 0.01, -0.01, 0.0, 0.0]
    lo, hi = ci95(d)
    assert -0.02 <= lo < 0 < hi <= 0.02
    assert verdict({"p_ttest": 0.5, "p_wilcoxon": 0.5, "mean_delta": 0.0}, [lo, hi]).startswith("equivalent")
    assert verdict({"p_ttest": 0.01, "p_wilcoxon": 0.002, "mean_delta": -0.05}, [-0.07, -0.03]) == "a lower (significant)"
    assert verdict({"p_ttest": 0.01, "p_wilcoxon": 0.002, "mean_delta": 0.05}, [0.03, 0.07]) == "a higher (significant)"
    assert verdict({"p_ttest": 0.2, "p_wilcoxon": 0.1, "mean_delta": -0.03}, [-0.08, 0.02]) == "inconclusive"


def test_family():
    assert family("original") == "single"
    assert family("multisource_v3") == "cross" and family("multisource_random_io") == "cross"
    assert family("dae_X3") == "synth"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_analyze_var_description.py -q`
Expected: `ImportError: cannot import name 'ci95'`

- [ ] **Step 3: Add stats and figure to `analyze_var_description.py`**

Insert after `quantiles` (module level):

```python
STATS_JSON = EXP / "var_desc_stats.json"
FIG_DIR = ROOT / "figures"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
METRICS = ["Recall@K_correct", "MAP", "Recall@20"]
COMPARISONS = [  # (a, b, 目的): a − b を検定する
    ("D1", "S1", "置き換え"),
    ("S1-r50", "S1", "頑健性（半分を替える）"),
    ("S1-r100", "S1", "頑健性（全部替える）"),
    ("D1", "S1-r100", "全部替えた記号版と説明版"),
    ("D0", "D1", "漏れの大きさ"),
    ("S1", "S0", "説明文から記号を消した影響"),
]
C_DESC, C_SYM, C_COPY = "#2a78d6", "#eb6834", "#1baf7a"  # dataviz の検証済み 3 色
INK = "#222222"


def family(variant: str) -> str:
    if variant == "original":
        return "single"
    if variant.startswith("multisource_"):
        return "cross"
    if variant.startswith("dae_"):
        return "synth"
    return "other"


def per_seed(res: dict, label: str, metric: str) -> list:
    return [p[metric] for p in sorted(res[label]["per_seed"], key=lambda p: SEEDS.index(p["seed"]))]


def ci95(d) -> list:
    from scipy import stats
    d = np.asarray(d, dtype=float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)]


def verdict(st: dict, ci: list) -> str:
    """結果の読み方（仕様 §5）。信頼区間が ±0.02 に収まれば同程度、両検定で p<0.05 なら有意."""
    if -0.02 <= ci[0] and ci[1] <= 0.02:
        return "equivalent: 95% CI within ±0.02"
    if st["p_ttest"] < 0.05 and st["p_wilcoxon"] < 0.05:
        return "a higher (significant)" if st["mean_delta"] > 0 else "a lower (significant)"
    return "inconclusive"


def safe_paired(base, rer) -> dict:
    from analyze_significance import paired_stats
    try:
        return paired_stats(base, rer)
    except ValueError as e:  # 差がすべて 0 などで Wilcoxon が計算できないとき
        return {"error": str(e)}


def compare(res: dict, a: str, b: str, purpose: str) -> dict:
    entry = {"purpose": purpose, "a": a, "b": b}
    for metric in METRICS:
        va, vb = per_seed(res, a, metric), per_seed(res, b, metric)
        st = safe_paired(vb, va)
        if "error" not in st:
            st["ci95"] = ci95(np.array(va) - np.array(vb))
            st["verdict"] = verdict(st, st["ci95"])
        entry[metric] = st
    return entry


def family_means(res: dict, label: str, exclude: set = frozenset()) -> dict:
    """族 → seed → その族のテストケースの R@K の平均."""
    by = {}
    for r in res[label].get("per_case", []):
        if r["case_id"] in exclude:
            continue
        by.setdefault(family(r["variant"]), {}).setdefault(r["seed"], []).append(r["Recall@K_correct"])
    return {f: {s: float(np.mean(v)) for s, v in d.items()} for f, d in by.items()}


def family_table(res: dict) -> dict:
    tab = {}
    for label in res:
        tab[label] = {}
        for f, d in family_means(res, label).items():
            v = list(d.values())
            tab[label][f] = {"mean": round(float(np.mean(v)), 4),
                             "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0,
                             "n_seeds": len(v)}
    return tab


def family_tests(res: dict, a: str = "D1", b: str = "S1") -> dict:
    if a not in res or b not in res:
        return {}
    fa, fb = family_means(res, a), family_means(res, b)
    out = {}
    for f in fa:
        seeds = [s for s in SEEDS if s in fa[f] and s in fb.get(f, {})]
        if len(seeds) >= 3:
            out[f] = safe_paired([fb[f][s] for s in seeds], [fa[f][s] for s in seeds])
    return {f"{a}_vs_{b}": out}


def excluding_conflict(res: dict, conflict: set, a: str = "D1", b: str = "S1") -> dict:
    def seed_means(label):
        by = {}
        for r in res[label].get("per_case", []):
            if r["case_id"] not in conflict:
                by.setdefault(r["seed"], []).append(r["Recall@K_correct"])
        return {s: float(np.mean(v)) for s, v in by.items()}
    ma, mb = seed_means(a), seed_means(b)
    seeds = [s for s in SEEDS if s in ma and s in mb]
    st = safe_paired([mb[s] for s in seeds], [ma[s] for s in seeds])
    if "error" not in st:
        st["ci95"] = ci95(np.array([ma[s] for s in seeds]) - np.array([mb[s] for s in seeds]))
        st["verdict"] = verdict(st, st["ci95"])
    return {f"{a}_vs_{b}": st, "n_excluded_cases": len(conflict)}


def cmd_stats(args) -> None:
    from var_conditions import load_var_descriptions
    vd = load_var_descriptions()
    qc = json.load(open(EXP / "var_descriptions_qc.json", encoding="utf-8"))
    conflict = {cid for cid, c in vd.items() if any(v["meaning_conflict"] for v in c["variables"].values())}
    ratio = qc["meaning_conflict_ratio"]
    out = {"spec": SPEC, "meaning_conflict": {"ratio": round(ratio, 4), "n_cases_with_conflict": len(conflict),
                                              "exclusion_applied": ratio > 0.05}}
    for setting in ("A", "B"):
        doc = json.load(open(EXP / f"var_desc_{setting}.json", encoding="utf-8"))
        res = doc["results"]
        comps = {f"{a}_vs_{b}": compare(res, a, b, p) for a, b, p in COMPARISONS if a in res and b in res}
        out[setting] = {"comparisons": comps, "family": family_table(res), "family_tests": family_tests(res),
                        "tau": doc["tau"], "approx_mismatch": doc["approx_mismatch"]}
        if ratio > 0.05:
            out[setting]["excluding_meaning_conflict"] = excluding_conflict(res, conflict)
        for a, b, _ in COMPARISONS:
            c = comps.get(f"{a}_vs_{b}", {}).get("Recall@K_correct", {})
            if "mean_delta" in c:
                print(f"[{setting}] {a} − {b}: Δ={c['mean_delta']:+.4f} CI={c['ci95']} "
                      f"p_t={c['p_ttest']} p_w={c['p_wilcoxon']} → {c['verdict']}")
    sens = {}
    for d in ("-0.02", "+0.02"):
        p = EXP / f"var_desc_tau{d}.json"
        if p.exists():
            r = json.load(open(p, encoding="utf-8"))["results"][f"D1-tau{d}"]["per_seed"][0]
            sens[d] = {k: r[k] for k in METRICS}
    a_d1 = json.load(open(EXP / "var_desc_A.json", encoding="utf-8"))["results"]["D1"]["per_seed"]
    sens["0"] = {k: next(p[k] for p in a_d1 if p["seed"] == 42) for k in METRICS}
    out["A"]["tau_sensitivity_seed42"] = sens
    json.dump(out, open(STATS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {STATS_JSON}")


def cmd_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    xs = [0, 50, 100]
    for ax, setting, title in zip(axes, ["A", "B"], ["(a) Setting A", "(b) Setting B"]):
        res = json.load(open(EXP / f"var_desc_{setting}.json", encoding="utf-8"))["results"]

        def ms(label):
            v = per_seed(res, label, "Recall@K_correct")
            return float(np.mean(v)), float(np.std(v, ddof=1))

        s = [ms(lab) for lab in ("S1", "S1-r50", "S1-r100")]
        d_m, d_sd = ms("D1")
        c_m, _ = ms("D0")
        ax.axhspan(d_m - d_sd, d_m + d_sd, color=C_DESC, alpha=0.15, lw=0, zorder=1)
        ax.axhline(d_m, color=C_DESC, lw=2, zorder=2, label="Matching by descriptions (D1)")
        ax.axhline(c_m, color=C_COPY, lw=1.5, ls="--", zorder=2,
                   label="Copied source descriptions (D0, reference)")
        ax.errorbar(xs, [m for m, _ in s], yerr=[sd for _, sd in s], color=C_SYM, lw=2, marker="o",
                    ms=7, capsize=4, zorder=3, label="Matching by symbols (S1)")
        for y, text in ((d_m, "D1"), (c_m, "D0"), (s[-1][0], "S1")):
            ax.annotate(text, xy=(100, y), xytext=(110, y), va="center", fontsize=10, color=INK)
        ax.set_xticks(xs)
        ax.set_xlim(-8, 124)
        ax.set_xlabel("Input/output variables given with a different symbol (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recall@K (mean ± SD over 10 seeds)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_var_description.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'fig_var_description.pdf'} (+ .png)")
```

Replace `main()` with:

```python
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("precheck", "stats", "figure"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"precheck": cmd_precheck, "stats": cmd_stats, "figure": cmd_figure}[args.cmd](args)
```

Also extend the module docstring's usage lines with:

```
  python3 analyze_var_description.py stats      # 対応あり検定・95% 信頼区間・判定・族別の内訳 → var_desc_stats.json
  python3 analyze_var_description.py figure     # figures/fig_var_description.pdf / .png
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_analyze_var_description.py -q`
Expected: `3 passed`

- [ ] **Step 5: Run stats and figure**

Run: `python3 analyze_var_description.py stats && python3 analyze_var_description.py figure`
Expected: six `[A] …` and six `[B] …` lines with Δ, CI, p values, and verdicts, then `Saved:` lines for the JSON and the figure.

- [ ] **Step 6: Look at the figure**

Open `figures/fig_var_description.png` with the Read tool. Check for overlapping direct labels (D1/D0/S1 at x=110), clipped error bars, and an overlapping legend. If labels collide (two y values within about 0.01), offset the lower label's `xytext` y by −0.012 and the upper one's by +0.012 inside `cmd_figure`, then re-run `figure` and look again.

- [ ] **Step 7: Full test suite**

Run: `python3 -m pytest tests/test_var_matching.py tests/test_var_desc_qc.py tests/test_var_conditions.py tests/test_run_var_description.py tests/test_make_var_descriptions.py tests/test_analyze_var_description.py -q`
Expected: `27 passed, 2 skipped`

---

### Task 10: Records (dev log) and final report

**Files:**
- Modify: `docs/development_log.tex` (new `\item` at the top of the itemize under `\section{更新履歴}`, line 358 onward)

**Interfaces:**
- Consumes: `experiments/var_descriptions_qc.json`, `experiments/var_desc_precheck.json`, `experiments/var_desc_stats.json`, `experiments/var_desc_coverage.json`, the Task 6 manual-check count.

- [ ] **Step 1: Add the dev-log entry**

Insert directly after the `\begin{itemize}[nosep]` that follows `\section{更新履歴}`. Fill every value from the JSON key listed in the table below (4 decimals, no rounding by hand), and use the day the runs finished in the item date.

```latex
  \item 2026-10-XX（変数を記号でなく説明で照合する実験：置き換えと頑健性）：
    09-27 報告の今後の計画（文脈が効く場面を示す評価）として、式とケースの変数を自然言語の説明で表し、変数の重複率などの特徴量を E5（\texttt{intfloat/e5-base-v2}）の類似度がしきい値 $\tau$ 以上の組の最大マッチングで測った（仕様 \texttt{docs/superpowers/specs/2026-10-01-variable-description-matching-design.md}）。
    ケース側の言い換え・記号なしの説明文・別の記号は Anthropic API（Claude Opus 5.5、バッチ）で生成した（\texttt{make\_var\_descriptions.py}、\texttt{experiments/var\_descriptions.json}、費用 QC1 USD）。品質：不合格 QC2 件、意味の食い違いの印 QC3、目視 50 変数のうち誤り MAN 件。
    E5 が同じ変数と同じ式の中の別の変数を分ける AUC は PRE1（D0 は PRE2）。$\tau$ は seed ごとに TAU1〜TAU2。
    設定 A：記号版 S1 の R@$K$ ST1 に対し説明版 D1 は ST2（差 ST3、95\% 信頼区間 ST4、判定 ST5）。テストだけ記号を 50\%・100\% 替えると記号版は ST6・ST7。正解式の説明をそのまま写した D0 は ST8。
    設定 B：S1 ST9、D1 ST10（判定 ST11）、記号を全部替えた記号版 ST12。第 1 段の被覆率（w30-70）は S1 COV1、D1 COV2。
    新規：\texttt{var\_matching.py}・\texttt{var\_desc\_qc.py}・\texttt{var\_conditions.py}・\texttt{run\_var\_description.py}・\texttt{make\_var\_descriptions.py}・\texttt{analyze\_var\_description.py}・\texttt{run\_var\_description.sh}、図 \texttt{figures/fig\_var\_description.pdf}。S0 は既報（seed 42 の R@$K$ 0.8403、被覆率 0.9671）を再現。
```

| Token | Source |
|---|---|
| QC1 | `var_descriptions_qc.json` → `usd_total` |
| QC2 | `var_descriptions_qc.json` → `len(qc_failed)` |
| QC3 | `var_descriptions_qc.json` → `meaning_conflict` / `n_variables` (write as "k / n 変数") |
| MAN | count of `N` in `var_descriptions_sample50.csv` (Task 6 Step 5) |
| PRE1, PRE2 | `var_desc_precheck.json` → `D1.auc`, `D0.auc` |
| TAU1, TAU2 | min and max over `var_desc_stats.json` → `A.tau.D1` values |
| ST1, ST2 | `A.comparisons.D1_vs_S1.Recall@K_correct.mean_base`, `.mean_rer` |
| ST3, ST4, ST5 | same entry `.mean_delta`, `.ci95`, `.verdict` |
| ST6, ST7 | `A.comparisons.S1-r50_vs_S1…mean_rer`, `A.comparisons.S1-r100_vs_S1…mean_rer` |
| ST8 | `A.comparisons.D0_vs_D1.Recall@K_correct.mean_rer` |
| ST9, ST10, ST11 | `B.comparisons.D1_vs_S1.Recall@K_correct.mean_base`, `.mean_rer`, `.verdict` |
| ST12 | `B.comparisons.S1-r100_vs_S1.Recall@K_correct.mean_rer` |
| COV1, COV2 | `var_desc_coverage.json` → `coverage.S1.w30-70`, `coverage.D1.w30-70` |

- [ ] **Step 2: Build the dev log (uplatex; platex drops circled digits)**

Run: `cd docs && uplatex -interaction=nonstopmode -halt-on-error development_log.tex > /dev/null && uplatex -interaction=nonstopmode -halt-on-error development_log.tex > /dev/null && dvipdfmx -q development_log.dvi && echo BUILD_OK`
Expected: `BUILD_OK`

- [ ] **Step 3: Final status and report (no commit)**

Run: `git status --short`
Expected: the new scripts, tests, `run_var_description.sh`, `experiments/var_desc_*.json`, `experiments/var_descriptions*.{json,csv}`, the figure, and the modified `.gitignore` and `docs/development_log.tex`.

Report to the user in Japanese, following the reading rules of spec §5 (no claim beyond the verdict strings):
1. 置き換え（D1 vs S1）の差・信頼区間・判定（A・B）
2. 頑健性（S1-r50/r100 の低下と有意性）
3. 漏れの大きさ（D0 − D1）
4. AUC と間違えやすい組の典型例 2〜3 件
5. 族別の内訳で目立つ点
6. 費用
Ask whether to commit (commit triggers auto-push). Note that how to phrase the positioning (whether to call variable descriptions 文脈, and the relation to [9] and to ProcessBERT) is decided together with the user (spec §5), not in this task.
