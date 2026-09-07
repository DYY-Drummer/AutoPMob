# A2 External Baselines (BM25 / E5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure BM25 and dense (E5) retrieval on the thesis evaluation protocol (Settings A/B, 10 stratified seeds, whole-database ranking) and add the rows to Table 3 of the thesis.

**Architecture:** One script `external_baselines.py` computes text-score matrices (TF-IDF / BM25 / E5) once per scorer, mixes them with the variable-overlap Jaccard matrix, ranks the whole database per case, and aggregates per seed with the metric functions of `evaluate_multi_eq.py` and the split of `set_aware_reranker.py`. A second script `analyze_external_baselines.py` runs the paired tests against `strat_A/B.json` and prints LaTeX rows. Thesis, devlog, and gap analysis are updated from the JSON.

**Tech Stack:** Python 3.12, numpy, scipy.sparse, scikit-learn (TfidfVectorizer / CountVectorizer), transformers 4.53 + torch 2.7 (MPS) for `intfloat/e5-base-v2`, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-07-a2-external-baselines-design.md`.
- Regression: `tfidf:none` with weights (0.7, 0.3) must reproduce `experiments/strat_{A,B}.json` `results.baseline.per_seed` exactly (Recall@K_correct, MAP, Recall@20).
- Seeds: `42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999`. Setting A variants `original,multisource_,dae_`; Setting B `dae_`.
- BM25: k1 = 1.5, b = 0.75, IDF = ln(1 + (N − n + 0.5)/(n + 0.5)), unigrams, sklearn default tokenizer.
- E5: `intfloat/e5-base-v2`, mean pooling, L2 normalization, prefixes `query: ` / `passage: `, max_length 512.
- Normalization for BM25/E5 mixing: per-query min–max over the whole database.
- No new pip packages. No commit (user commits).
- Thesis text: Strunk & White rules (active voice, omit needless words); Kato-lab rules (numbers from JSON, "significant" only after a test, abbreviations expanded at first use).

---

### Task 1: Core scoring utilities with tests

**Files:**
- Create: `external_baselines.py`
- Test: `tests/test_external_baselines.py`

**Interfaces (produced):**
- `class BM25(texts, k1=1.5, b=0.75)` with `.score(query) -> np.ndarray[N]` and `.score_many(queries) -> np.ndarray[Q, N]`
- `minmax_rows(S: np.ndarray[Q, N]) -> np.ndarray[Q, N]`
- `jaccard_matrix(query_sets: list[set], eq_sets: list[set]) -> np.ndarray[Q, N] float32`
- `coverage(order: list[int], corr: set[int], k: int) -> float`
- `parse_weights("1:0,0.7:0.3") -> [(1.0, 0.0), (0.7, 0.3)]`, `parse_scorers("bm25:minmax") -> [("bm25", "minmax")]`
- `label(scorer, norm, wt, wv) -> "bm25__minmax__w70-30"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_external_baselines.py
import math
from collections import Counter

import numpy as np

from external_baselines import (
    BM25, minmax_rows, jaccard_matrix, coverage, parse_weights, parse_scorers, label,
)
from two_stage_query_conditioned import jaccard


def _ref_bm25(docs, query, k1=1.5, b=0.75):
    toks = [d.split() for d in docs]
    N = len(toks); avgdl = sum(len(t) for t in toks) / N
    df = Counter(t for d in toks for t in set(d))
    out = []
    for d in toks:
        tf = Counter(d); s = 0.0
        for t in query.split():
            if t not in df: continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def test_bm25_matches_reference():
    docs = ["mass balance reactor", "energy balance heat exchanger reactor reactor", "reactor volume flow"]
    bm = BM25(docs)
    got = bm.score("reactor balance balance")
    exp = _ref_bm25(docs, "reactor balance balance")
    assert np.allclose(got, exp, atol=1e-6)
    assert bm.score("unknownterm").tolist() == [0.0, 0.0, 0.0]
    assert bm.score_many(["reactor", "flow"]).shape == (2, 3)


def test_minmax_rows():
    S = np.array([[1.0, 3.0, 2.0], [5.0, 5.0, 5.0]])
    M = minmax_rows(S)
    assert np.allclose(M[0], [0.0, 1.0, 0.5])
    assert np.allclose(M[1], [0.0, 0.0, 0.0])


def test_jaccard_matrix_matches_python():
    qs = [{"a", "b"}, set(), {"z"}]
    es = [{"a"}, {"b", "c"}, set(), {"a", "b"}]
    J = jaccard_matrix(qs, es)
    assert J.dtype == np.float32
    for i, q in enumerate(qs):
        for j, e in enumerate(es):
            assert J[i, j] == np.float32(jaccard(q, e))


def test_coverage():
    assert coverage([5, 1, 9, 2], {1, 2, 3}, 2) == 1 / 3
    assert coverage([5, 1, 9, 2], {1, 2, 3}, 4) == 2 / 3
    assert math.isnan(coverage([1], set(), 1))


def test_parsers_and_label():
    assert parse_weights("1:0,0.7:0.3") == [(1.0, 0.0), (0.7, 0.3)]
    assert parse_scorers("tfidf:none,bm25:minmax") == [("tfidf", "none"), ("bm25", "minmax")]
    assert label("bm25", "minmax", 0.7, 0.3) == "bm25__minmax__w70-30"
    assert label("e5", "minmax", 1.0, 0.0) == "e5__minmax__w100-00"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_external_baselines.py -q`
Expected: ImportError (module `external_baselines` missing).

- [ ] **Step 3: Implement the utilities**

```python
# external_baselines.py (part 1)
"""外部ベースライン（BM25・密ベクトル検索）を修論の評価枠で測る. ...docstring per spec..."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from two_stage_query_conditioned import (load_equations, load_cases, norm, eq_key, eq_text,
    eq_vars, case_text, io_vars, in_vars, out_vars, case_src, jaccard)
from evaluate_multi_eq import compute_all_ranks, case_metrics, aggregate_metrics

EXP = ROOT / "experiments"; EMB_DIR = EXP / "embeddings"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
SETTINGS = {"A": ["original", "multisource_", "dae_"], "B": ["dae_"]}
KS = [10, 25, 50, 100, 200, 400]
DEFAULT_SCORERS = "tfidf:none,tfidf:minmax,bm25:minmax,e5:minmax,e5io:minmax"
DEFAULT_WEIGHTS = "1:0,0.7:0.3,0.3:0.7"
E5_MODEL = "intfloat/e5-base-v2"

def parse_weights(s): return [(float(a), float(b)) for a, b in (x.split(":") for x in s.split(",") if x.strip())]
def parse_scorers(s): return [tuple(x.strip().split(":")) for x in s.split(",") if x.strip()]
def label(scorer, norm, wt, wv): return f"{scorer}__{norm}__w{round(wt*100):02d}-{round(wv*100):02d}"

class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        self.cv = CountVectorizer(lowercase=True)
        tf = self.cv.fit_transform(texts).astype(np.float64).tocoo()
        N = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel(); avgdl = dl.mean() if N else 0.0
        df = np.bincount(tf.col, minlength=tf.shape[1])
        self.idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        w = self.idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl))
        self.W = sp.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.analyzer = self.cv.build_analyzer(); self.vocab = self.cv.vocabulary_
    def score(self, query):
        toks = [self.vocab[t] for t in self.analyzer(query) if t in self.vocab]
        if not toks: return np.zeros(self.W.shape[0])
        cols, counts = np.unique(toks, return_counts=True)
        return np.asarray(self.W[:, cols] @ counts.astype(np.float64)).ravel()
    def score_many(self, queries): return np.stack([self.score(q) for q in queries])

def minmax_rows(S):
    S = np.asarray(S, dtype=np.float64)
    lo = S.min(axis=1, keepdims=True); hi = S.max(axis=1, keepdims=True); rng = hi - lo
    return np.where(rng > 0, (S - lo) / np.where(rng > 0, rng, 1.0), 0.0)

def jaccard_matrix(query_sets, eq_sets):
    vocab = {}
    for s in list(query_sets) + list(eq_sets):
        for v in s: vocab.setdefault(v, len(vocab))
    def mat(sets):
        rows, cols = [], []
        for i, s in enumerate(sets):
            for v in s: rows.append(i); cols.append(vocab[v])
        return sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(sets), max(1, len(vocab))))
    A, B = mat(query_sets), mat(eq_sets)
    inter = (A @ B.T).toarray().astype(np.int64)
    union = np.asarray(A.sum(axis=1)).reshape(-1, 1) + np.asarray(B.sum(axis=1)).reshape(1, -1) - inter
    J = np.zeros(inter.shape, dtype=np.float64)
    np.divide(inter, union, out=J, where=union > 0)
    return J.astype(np.float32)

def coverage(order, corr, k):
    if not corr: return float("nan")
    return len(corr & set(order[:k])) / len(corr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_external_baselines.py -q` → 5 passed.

---

### Task 2: Data loading, text scorers, E5 embedding, evaluation loop, CLI

**Files:**
- Modify: `external_baselines.py` (append)
- Modify: `.gitignore` (add `experiments/embeddings/`)
- Test: `tests/test_external_baselines.py` (append regression test)

**Interfaces (produced):**
- `load_setting(setting) -> dict(cases, eq_keys, eq_texts, eq_vars, correct_lists, case_srcs, feats)`
- `text_scores(scorer, eq_texts, cases, model_name) -> np.ndarray[Q, N]` (tfidf rows computed one by one, float64)
- `embed_texts(texts, model_name, kind) -> np.ndarray[Q, d]` (cached under `experiments/embeddings/`)
- `evaluate_setting(data, scorers, weights, seeds, model_name, do_coverage) -> (results: dict, coverage: dict)`

- [ ] **Step 1: Write the failing regression test**

```python
import json
from external_baselines import load_setting, evaluate_setting

def test_tfidf_none_reproduces_strat_B_seed42():
    data = load_setting("B")
    results, _ = evaluate_setting(data, [("tfidf", "none")], [(0.7, 0.3)], seeds=[42], model_name=None, do_coverage=False)
    got = results["tfidf__none__w70-30"]["per_seed"][0]
    ref = next(p for p in json.load(open("experiments/strat_B.json"))["results"]["baseline"]["per_seed"] if p["seed"] == 42)
    for m in ["Recall@K_correct", "MAP", "Recall@20"]:
        assert abs(got[m] - ref[m]) < 1e-9, (m, got[m], ref[m])
    assert got["n_cases"] == ref["n_cases"]
```

- [ ] **Step 2: Run it to verify it fails** (ImportError for `load_setting`).

- [ ] **Step 3: Implement**

```python
# external_baselines.py (part 2)
def variant_matches(v, wanted):
    return any(v.startswith(w) if w.endswith("_") else v == w for w in wanted)

def load_setting(setting):
    eqs = load_equations()
    cases = [c for c in load_cases() if variant_matches(c.get("variant_type", ""), SETTINGS[setting])]
    ek, et, ev = [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k: continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]
    feats = [(len(cl[i]), len(in_vars(cases[i])), len(out_vars(cases[i])),
              len({ek[j].split("__")[0] for j in cl[i]})) for i in range(len(cases))]
    return dict(setting=setting, cases=cases, eq_keys=ek, eq_texts=et, eq_vars=ev,
                correct_lists=cl, case_srcs=cs, feats=feats)

def embed_texts(texts, model_name, kind, batch_size=64, max_length=512):
    key = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = EMB_DIR / f"{model_name.replace('/', '_')}__{kind}__{key}.npy"
    if path.exists(): return np.load(path)
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            batch = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out[idx] = torch.nn.functional.normalize(e, p=2, dim=1).float().cpu().numpy()
    EMB_DIR.mkdir(parents=True, exist_ok=True); np.save(path, out)
    return out

def text_scores(scorer, eq_texts, cases, model_name):
    if scorer == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
        X_eq = tfidf.fit_transform(eq_texts); X_q = tfidf.transform([case_text(c) for c in cases])
        return np.stack([cosine_similarity(X_q[i], X_eq).ravel() for i in range(X_q.shape[0])])
    if scorer == "bm25":
        return BM25(eq_texts).score_many([case_text(c) for c in cases])
    if scorer in ("e5", "e5io"):
        io = scorer == "e5io"
        E = embed_texts(["passage: " + t for t in eq_texts], model_name, "passage")
        Q = embed_texts(["query: " + case_text(c, io=io) for c in cases], model_name, "query_io" if io else "query_ctx")
        return (Q @ E.T).astype(np.float64)
    raise ValueError(scorer)

SUMMARY_KEYS = [...same list as set_aware_reranker...]

def evaluate_setting(data, scorers, weights, seeds, model_name, do_coverage):
    from set_aware_reranker import stratified_src_split
    cases, cl, cs = data["cases"], data["correct_lists"], data["case_srcs"]
    n_eq = len(data["eq_keys"])
    J = jaccard_matrix([io_vars(c) for c in cases], data["eq_vars"])
    splits = {seed: stratified_src_split(cs, data["feats"], seed) for seed in seeds}
    tests = {seed: [i for i, s in enumerate(cs) if s in splits[seed]["test"] and cl[i]] for seed in seeds}
    evaluable = [i for i in range(len(cases)) if cl[i]]
    results, cov = {}, {}
    for scorer, nm in scorers:
        S = text_scores(scorer, data["eq_texts"], cases, model_name)
        S = minmax_rows(S) if nm == "minmax" else S
        for wt, wv in weights:
            lab = label(scorer, nm, wt, wv)
            cms, covs = {}, {k: [] for k in KS}
            for ci in evaluable:
                corr = set(cl[ci]); vs = J[ci]; ts = S[ci]
                order = np.argsort(-(wt * ts + wv * vs))
                ol = order.tolist()
                cm = case_metrics(compute_all_ranks(ol, corr)); cm["variant"] = ...; cm["case_id"] = ...
                cms[ci] = cm
                for k in KS: covs[k].append(coverage(ol, corr, k))
            per_seed = []
            for seed in seeds:
                agg = aggregate_metrics([cms[ci] for ci in tests[seed]]); agg["seed"] = seed; per_seed.append(agg)
            summary = {"mode": lab, "scorer": scorer, "norm": nm, "w_text": wt, "w_var": wv}
            for k in SUMMARY_KEYS: (mean/std over per_seed as in set_aware_reranker)
            summary["per_seed"] = per_seed; results[lab] = summary
            if do_coverage: cov[lab] = {str(k): mean_sem(covs[k]) for k in KS}
    return results, cov
```

CLI (`main`): `--setting {A,B}`, `--scorers`, `--weights`, `--seed-list`, `--model`, `--output`, `--coverage`, `--coverage-output`, `--save-per-case`. Writes `{"config": vars(args) + n_cases/n_eq/device, "results": results}`; merges coverage into the coverage JSON under `coverage[setting]`.

- [ ] **Step 4: Run the regression test** → PASS (values identical to strat_B seed 42).

- [ ] **Step 5: Smoke run** `python3 external_baselines.py --setting B --scorers tfidf:none,bm25:minmax --weights 0.7:0.3 --seed-list 42 --output /tmp/scratch/smoke.json` → prints per-seed lines.

---

### Task 3: Full run (both settings, all scorers, coverage) and analysis script

**Files:**
- Create: `run_external_baselines.sh`
- Create: `analyze_external_baselines.py`
- Create: `experiments/external_baselines_A.json`, `_B.json`, `external_baselines_coverage.json`, `external_baselines_stats.json`

- [ ] **Step 1: Run** `caffeinate -i bash run_external_baselines.sh` in the background (E5 download 438 MB + encoding ≈ 10 min on MPS). Check logs.

- [ ] **Step 2: Write `analyze_external_baselines.py`**

```python
"""外部ベースライン（BM25・E5）と古典 IR ベースライン・提案手法（reranker-10S）の対応あり検定."""
import json; from pathlib import Path
from analyze_significance import paired_stats
ROOT = Path(__file__).resolve().parent; EXP = ROOT / "experiments"
METRICS = ["Recall@K_correct", "MAP", "Recall@20"]
def per_seed(doc, mode, m): return [p[m] for p in doc["results"][mode]["per_seed"]]
def main():
    out = {}
    for s in ["A", "B"]:
        strat = json.load(open(EXP / f"strat_{s}.json")); ext = json.load(open(EXP / f"external_baselines_{s}.json"))
        seeds_ref = [p["seed"] for p in strat["results"]["baseline"]["per_seed"]]
        out[s] = {"seeds": seeds_ref, "labels": {}}
        for lab, res in ext["results"].items():
            assert [p["seed"] for p in res["per_seed"]] == seeds_ref
            entry = {}
            for m in METRICS:
                x = per_seed(ext, lab, m)
                entry[m] = {"mean": res[m]["mean"], "std": res[m]["std"],
                            "vs_baseline": paired_stats(per_seed(strat, "baseline", m), x),
                            "vs_proposed": paired_stats(x, per_seed(strat, "reranker-10S", m))}
            out[s]["labels"][lab] = entry
        # regression check
        reg = out[s]["labels"].get("tfidf__none__w70-30")
        if reg: assert abs(reg["Recall@K_correct"]["vs_baseline"]["mean_delta"]) < 1e-9
    json.dump(out, open(EXP / "external_baselines_stats.json", "w"), ensure_ascii=False, indent=1)
    # LaTeX rows
    for s in out: for lab, e in out[s]["labels"].items(): print(f"{s} & {lab} & ${e['Recall@K_correct']['mean']:.3f} \\pm {e['Recall@K_correct']['std']:.3f}$ & {e['MAP']['mean']:.3f} & {e['Recall@20']['mean']:.3f} \\\\  % p_vs_prop={...}")
```

- [ ] **Step 3: Run it**, read the table, apply the decision rules of spec §4.

---

### Task 4: Thesis update

**Files:**
- Modify: `thesis/master_thesis/Experiment.tex` §5.2 (after the internal/external baseline paragraph)
- Modify: `thesis/master_thesis/ResultsAndDiscussion.tex` §6.1 (Table 3 rows + paragraph)
- Modify: `thesis/master_thesis/Appendix.tex` (hyperparameter rows)
- Modify: `thesis/master_thesis/main.tex` (`\bibitem{E5}` — verify on arXiv 2212.03533 first; insert in first-citation order)
- Conditionally: `ResultsAndDiscussion.tex` §6.8 (coverage sentence)

- [ ] Steps: verify E5 bibliography with WebFetch → write the paragraphs with `% source:` comments → `cd thesis/master_thesis && latexmk -gg main.tex` → exit 0, no undefined citations, page count noted.

---

### Task 5: Records

**Files:**
- Modify: `docs/development_log.tex` (dated entry, uplatex build check)
- Modify: `docs/論文化ギャップ分析_2026-08-30.md` (A2 → ✅ with results)
- Modify: `README.md` if it lists scripts.

- [ ] Steps: append entry → build devlog with `cd docs && latexmk -gg development_log.tex` (or uplatex + dvipdfmx) → update gap analysis → run full pytest → report.
