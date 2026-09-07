#!/usr/bin/env python3
"""外部ベースライン（BM25・密ベクトル検索）を修論の評価枠で測る（A2）.

古典 IR ベースライン（TF-IDF 0.7 ＋ 変数 Jaccard 0.3、全 DB 順位付け、学習なし）と
同じ土俵で、文章スコアを BM25 / E5 埋め込みに差し替えたときの
Recall@K_correct / MAP / Recall@20 を、設定 A・B × 層化分割 10 seed で測る。

  score(q, e) = w_text * S~(q, e) + w_var * Jaccard(io_vars(q), vars(e))

  S~ : 文章スコアをクエリごとに正規化したもの
       tfidf = TF-IDF (1-2 gram) のコサイン類似度（現行ベースラインと同一）
       bm25  = Okapi BM25（k1=1.5, b=0.75, 単語ユニグラム, Lucene 形式の非負 IDF）
       e5    = intfloat/e5-base-v2 の平均プーリング埋め込みのコサイン（クエリ=context）
       e5io  = 同上、クエリ=context ＋ 入出力変数列（case_text(c, io=True)）
       正規化 none = 生値 / minmax = 全 DB にわたりクエリごとに [0, 1]

分割・指標は set_aware_reranker.py / evaluate_multi_eq.py の関数をそのまま使うので、
tfidf:none・(0.7, 0.3) は experiments/strat_{A,B}.json の baseline を完全に再現する
（tests/test_external_baselines.py で確認）。順位付けは seed に依存しないため、
各ケースの順位を 1 回だけ計算し、seed ごとにテストケース集合で集計する。

出力:
  experiments/external_baselines_{A,B}.json      （strat_A.json と同じ形: results[label]）
  experiments/external_baselines_coverage.json   （--coverage: 第 1 段の被覆率 k=10..400）
  experiments/embeddings/*.npy                   （E5 埋め込みのキャッシュ, 非追跡）
仕様: docs/superpowers/specs/2026-09-07-a2-external-baselines-design.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from two_stage_query_conditioned import (  # noqa: E402
    load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
    case_text, io_vars, in_vars, out_vars, case_src,
)
from evaluate_multi_eq import (  # noqa: E402
    case_metrics, aggregate_metrics,
)

EXP = ROOT / "experiments"
EMB_DIR = EXP / "embeddings"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
SETTINGS = {"A": ["original", "multisource_", "dae_"], "B": ["dae_"]}
KS = [10, 25, 50, 100, 200, 400]
DEFAULT_SCORERS = "tfidf:none,tfidf:minmax,bm25:minmax,e5:minmax,e5io:minmax"
DEFAULT_WEIGHTS = "1:0,0.9:0.1,0.8:0.2,0.7:0.3,0.6:0.4,0.5:0.5,0.4:0.6,0.3:0.7,0.2:0.8,0.1:0.9,0.05:0.95,0:1"
E5_MODEL = "intfloat/e5-base-v2"
SUMMARY_KEYS = [
    "MRR_first", "MRR_worst", "MRR_avg", "MAP",
    "Precision@C", "Recall@K_correct",
    "FullRecall@3", "FullRecall@10",
    "Recall@3", "Recall@5", "Recall@10", "Recall@20",
    "multi_only__MRR_first", "multi_only__MRR_worst",
    "multi_only__MAP", "multi_only__FullRecall@3", "multi_only__FullRecall@10",
    "multi_only__Recall@3", "multi_only__Recall@10",
    "multi_only__Recall@K_correct",
]


# ---------------------------------------------------------------------------
# 小道具
# ---------------------------------------------------------------------------

def parse_weights(s: str) -> list[tuple[float, float]]:
    """"1:0,0.7:0.3" → [(1.0, 0.0), (0.7, 0.3)]（w_text:w_var）."""
    out = []
    for x in s.split(","):
        if not x.strip():
            continue
        a, b = x.split(":")
        out.append((float(a), float(b)))
    return out


def parse_scorers(s: str) -> list[tuple[str, str]]:
    """"tfidf:none,bm25:minmax" → [("tfidf", "none"), ("bm25", "minmax")]."""
    out = []
    for x in s.split(","):
        if not x.strip():
            continue
        scorer, nm = x.strip().split(":")
        out.append((scorer, nm))
    return out


def label(scorer: str, nm: str, wt: float, wv: float) -> str:
    return f"{scorer}__{nm}__w{round(wt * 100):02d}-{round(wv * 100):02d}"


def variant_matches(v: str, wanted: list[str]) -> bool:
    """set_aware_reranker.main と同じ規則（接尾辞 '_' は前方一致）."""
    return any(v.startswith(w) if w.endswith("_") else v == w for w in wanted)


def mean_sem(vals) -> dict:
    a = np.asarray([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "sem": sem, "n": int(a.size)}


def coverage(order: list, corr: set, k: int) -> float:
    """上位 k 件が正解式を含む割合（analyze_stage1_coverage.py と同じ定義）."""
    if not corr:
        return float("nan")
    return len(corr & set(order[:k])) / len(corr)


def ranks_of(order: np.ndarray, corr) -> list[int]:
    """全 DB の順列 order（降順）における各正解式の 1 始まり順位.

    evaluate_multi_eq.compute_all_ranks(order.tolist(), corr) と同じ値を、
    11k 要素の辞書を作らずに逆順列で求める（全式が order に含まれるので欠落はない）。
    """
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(1, len(order) + 1)
    return [int(inv[c]) for c in corr]


def coverage_from_ranks(ranks: list[int], k: int) -> float:
    """coverage(order, corr, k) と同じ値を順位から求める."""
    if not ranks:
        return float("nan")
    return sum(1 for r in ranks if r <= k) / len(ranks)


# ---------------------------------------------------------------------------
# 文章スコア
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25（scipy sparse でベクトル化）.

    IDF は Lucene 形式 ln(1 + (N - n + 0.5) / (n + 0.5))（非負）。
    トークナイズは sklearn 既定（小文字化・\\b\\w\\w+\\b・単語ユニグラム）＝ TF-IDF と同じ規則。
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        self.k1, self.b = k1, b
        self.cv = CountVectorizer(lowercase=True)
        tf = self.cv.fit_transform(texts).astype(np.float64).tocoo()
        n_doc = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n_doc else 0.0
        df = np.bincount(tf.col, minlength=tf.shape[1])
        self.idf = np.log(1.0 + (n_doc - df + 0.5) / (df + 0.5))
        denom = tf.data + k1 * (1.0 - b + b * dl[tf.row] / avgdl)
        w = self.idf[tf.col] * tf.data * (k1 + 1.0) / denom
        self.W = sp.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)  # N x V
        self.analyzer = self.cv.build_analyzer()
        self.vocab = self.cv.vocabulary_

    def score(self, query: str) -> np.ndarray:
        toks = [self.vocab[t] for t in self.analyzer(query) if t in self.vocab]
        if not toks:
            return np.zeros(self.W.shape[0], dtype=np.float64)
        cols, counts = np.unique(toks, return_counts=True)
        return np.asarray(self.W[:, cols] @ counts.astype(np.float64)).ravel()

    def score_many(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.score(q) for q in queries])


def minmax_rows(S) -> np.ndarray:
    """行（クエリ）ごとに [0, 1] へ min-max 正規化。定数行は全 0."""
    S = np.asarray(S, dtype=np.float64)
    lo = S.min(axis=1, keepdims=True)
    hi = S.max(axis=1, keepdims=True)
    rng = hi - lo
    safe = np.where(rng > 0, rng, 1.0)
    return np.where(rng > 0, (S - lo) / safe, 0.0)


def jaccard_matrix(query_sets: list[set], eq_sets: list[set]) -> np.ndarray:
    """Jaccard(query_sets[i], eq_sets[j]) の Q x N 行列（float32）.

    two_stage_query_conditioned.jaccard を全ペアで呼んだ結果と同じ値になる
    （整数の共通部分・和集合を float64 で割ってから float32 に落とす）。
    """
    vocab: dict[str, int] = {}
    for s in list(query_sets) + list(eq_sets):
        for v in s:
            vocab.setdefault(v, len(vocab))

    def mat(sets):
        rows, cols = [], []
        for i, s in enumerate(sets):
            for v in s:
                rows.append(i)
                cols.append(vocab[v])
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int64), (rows, cols)),
            shape=(len(sets), max(1, len(vocab))),
        )

    A, B = mat(query_sets), mat(eq_sets)
    inter = (A @ B.T).toarray().astype(np.int64)
    union = (np.asarray(A.sum(axis=1)).reshape(-1, 1)
             + np.asarray(B.sum(axis=1)).reshape(1, -1) - inter)
    J = np.zeros(inter.shape, dtype=np.float64)
    np.divide(inter, union, out=J, where=union > 0)
    return J.astype(np.float32)


def embed_texts(texts: list[str], model_name: str, kind: str,
                batch_size: int = 64, max_length: int = 512) -> np.ndarray:
    """E5 系モデルで平均プーリング＋L2 正規化した埋め込み（キャッシュ付き）.

    texts には接頭辞（"query: " / "passage: "）を付けた文字列を渡す。
    """
    key = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = EMB_DIR / f"{model_name.replace('/', '_')}__{kind}__{key}.npy"
    if path.exists():
        return np.load(path)
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    order = np.argsort([len(t) for t in texts])  # 長さ順でパディングを減らす
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            batch = tok([texts[i] for i in idx], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
            h = model(**batch).last_hidden_state
            m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            e = torch.nn.functional.normalize(e, p=2, dim=1)
            out[idx] = e.float().cpu().numpy()
    print(f"  [embed] {kind}: {len(texts)} texts on {device} in {time.time() - t0:.0f}s "
          f"→ {path.name}")
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def text_scores(scorer: str, eq_texts: list[str], cases: list[dict],
                model_name: str | None) -> np.ndarray:
    """文章スコア行列 S（ケース数 × 式数, float64, 正規化前）."""
    if scorer == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
        X_eq = tfidf.fit_transform(eq_texts)
        X_q = tfidf.transform([case_text(c) for c in cases])
        # 現行ベースラインと同じく 1 ケースずつ計算する（ビット単位で同じ値にするため）
        return np.stack([cosine_similarity(X_q[i], X_eq).ravel() for i in range(X_q.shape[0])])
    if scorer == "bm25":
        return BM25(eq_texts).score_many([case_text(c) for c in cases])
    if scorer in ("e5", "e5io"):
        model_name = model_name or E5_MODEL
        io = scorer == "e5io"
        E = embed_texts(["passage: " + t for t in eq_texts], model_name, "passage")
        Q = embed_texts(["query: " + case_text(c, io=io) for c in cases], model_name,
                        "query_io" if io else "query_ctx")
        return (Q @ E.T).astype(np.float64)
    raise ValueError(f"unknown scorer: {scorer}")


# ---------------------------------------------------------------------------
# データと評価
# ---------------------------------------------------------------------------

def load_setting(setting: str) -> dict:
    """設定 A/B のケース・式 DB・正解索引・層化用特徴量（set_aware_reranker.main と同じ手順）."""
    eqs = load_equations()
    cases = [c for c in load_cases() if variant_matches(c.get("variant_type", ""), SETTINGS[setting])]
    ek, et, ev = [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k)
        et.append(eq_text(e))
        ev.append(eq_vars(e))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]
    feats = [(len(cl[i]), len(in_vars(cases[i])), len(out_vars(cases[i])),
              len({ek[j].split("__")[0] for j in cl[i]})) for i in range(len(cases))]
    return dict(setting=setting, cases=cases, eq_keys=ek, eq_texts=et, eq_vars=ev,
                correct_lists=cl, case_srcs=cs, feats=feats)


def evaluate_setting(data: dict, scorers: list[tuple[str, str]], weights: list[tuple[float, float]],
                     seeds: list[int], model_name: str | None, do_coverage: bool,
                     save_per_case: bool = False, verbose: bool = True) -> tuple[dict, dict]:
    """各 (scorer, norm, weight) について全ケースを 1 回順位付けし、seed ごとに集計する."""
    from set_aware_reranker import stratified_src_split

    cases, cl, cs = data["cases"], data["correct_lists"], data["case_srcs"]
    J = jaccard_matrix([io_vars(c) for c in cases], data["eq_vars"])
    splits = {seed: stratified_src_split(cs, data["feats"], seed) for seed in seeds}
    tests = {seed: [i for i, s in enumerate(cs) if s in splits[seed]["test"] and cl[i]]
             for seed in seeds}
    evaluable = [i for i in range(len(cases)) if cl[i]]

    results, cov = {}, {}
    for scorer, nm in scorers:
        t0 = time.time()
        S = text_scores(scorer, data["eq_texts"], cases, model_name)
        if nm == "minmax":
            S = minmax_rows(S)
        elif nm != "none":
            raise ValueError(f"unknown normalization: {nm}")
        if verbose:
            print(f"[{scorer}:{nm}] text scores {S.shape} in {time.time() - t0:.0f}s", flush=True)
        for wt, wv in weights:
            lab = label(scorer, nm, wt, wv)
            cms: dict[int, dict] = {}
            covs = {k: [] for k in KS}
            for ci in evaluable:
                corr = set(cl[ci])
                ts = S[ci]
                vs = J[ci]
                order = np.argsort(-(wt * ts + wv * vs))
                ranks = ranks_of(order, corr)
                cm = case_metrics(ranks)
                cm["variant"] = norm(cases[ci].get("variant_type") or "?")
                cm["case_id"] = cases[ci].get("case_id", f"idx_{ci}")
                cm["n_input"] = len(in_vars(cases[ci]))
                cm["n_output"] = len(out_vars(cases[ci]))
                cms[ci] = cm
                if do_coverage:
                    for k in KS:
                        covs[k].append(coverage_from_ranks(ranks, k))
            per_seed, per_case = [], []
            for seed in seeds:
                agg = aggregate_metrics([cms[ci] for ci in tests[seed]])
                agg["seed"] = seed
                per_seed.append(agg)
                if save_per_case:
                    for ci in tests[seed]:
                        rec = {"seed": seed, "mode": lab, **cms[ci]}
                        rec.pop("FullRecall", None)
                        rec.pop("Recall", None)
                        per_case.append(rec)
                if verbose:
                    print(f"  [seed={seed}] {lab:24s}  R@K_c={agg['Recall@K_correct']:.4f}  "
                          f"MAP={agg['MAP']:.4f}  R@20={agg['Recall@20']:.4f}  n={agg['n_cases']}",
                          flush=True)
            summary = {"mode": lab, "scorer": scorer, "norm": nm, "w_text": wt, "w_var": wv,
                       "n_features": 0}
            for k in SUMMARY_KEYS:
                vals = [a.get(k) for a in per_seed if a.get(k) is not None]
                if vals:
                    summary[k] = {
                        "mean": round(float(np.mean(vals)), 4),
                        "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0, 4),
                    }
            summary["per_seed"] = per_seed
            if save_per_case:
                summary["per_case"] = per_case
            results[lab] = summary
            if do_coverage:
                cov[lab] = {str(k): mean_sem(covs[k]) for k in KS}
    return results, cov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=list(SETTINGS), default="A")
    ap.add_argument("--scorers", default=DEFAULT_SCORERS,
                    help="scorer:norm のカンマ区切り（scorer ∈ tfidf,bm25,e5,e5io / norm ∈ none,minmax）")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="w_text:w_var のカンマ区切り")
    ap.add_argument("--seed-list", default=None, help="seed のカンマ区切り（既定: 正典 10 個）")
    ap.add_argument("--model", default=E5_MODEL, help="密ベクトルのモデル ID（Hugging Face）")
    ap.add_argument("--output", default=None,
                    help="結果 JSON（既定: experiments/external_baselines_<setting>.json）")
    ap.add_argument("--coverage", action="store_true", help="第 1 段の被覆率も測る")
    ap.add_argument("--coverage-output", default=str(EXP / "external_baselines_coverage.json"))
    ap.add_argument("--save-per-case", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seed_list.split(",") if s.strip()] if args.seed_list else SEEDS
    scorers = parse_scorers(args.scorers)
    weights = parse_weights(args.weights)
    out_path = Path(args.output) if args.output else EXP / f"external_baselines_{args.setting}.json"

    t0 = time.time()
    data = load_setting(args.setting)
    n_evaluable = sum(1 for cl in data["correct_lists"] if cl)
    print(f"Setting {args.setting}: variants={SETTINGS[args.setting]}  cases={len(data['cases'])} "
          f"(evaluable {n_evaluable})  equations={len(data['eq_keys'])}  seeds={seeds}")

    results, cov = evaluate_setting(data, scorers, weights, seeds, args.model, args.coverage,
                                    save_per_case=args.save_per_case)

    device = None
    if any(s in ("e5", "e5io") for s, _ in scorers):
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = {**vars(args), "seeds": seeds, "n_cases": len(data["cases"]),
              "n_evaluable": n_evaluable, "n_equations": len(data["eq_keys"]),
              "bm25": {"k1": 1.5, "b": 0.75, "idf": "ln(1+(N-n+0.5)/(n+0.5))", "tokens": "unigram"},
              "e5": {"model": args.model, "pooling": "mean", "normalize": "L2", "max_length": 512,
                     "prefix": {"query": "query: ", "passage": "passage: "}, "device": device},
              "elapsed_sec": round(time.time() - t0, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\nSaved: {out_path}")

    if args.coverage:
        cov_path = Path(args.coverage_output)
        doc = json.load(open(cov_path)) if cov_path.exists() else {"config": {}, "coverage": {}}
        doc["config"][args.setting] = {"n_cases": len(data["cases"]), "n_evaluable": n_evaluable,
                                       "n_eq": len(data["eq_keys"]), "ks": KS}
        doc["coverage"][args.setting] = cov
        with open(cov_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"Saved: {cov_path}")

    print(f"\n{'label':26s} | {'R@K_c':>14s} | {'MAP':>7s} | {'R@20':>7s} | cov@50")
    print("-" * 80)
    for lab, r in results.items():
        c50 = cov.get(lab, {}).get("50", {}).get("mean", float("nan"))
        print(f"{lab:26s} | {r['Recall@K_correct']['mean']:.4f}±{r['Recall@K_correct']['std']:.3f} | "
              f"{r['MAP']['mean']:.4f} | {r['Recall@20']['mean']:.4f} | {c50:.4f}")


if __name__ == "__main__":
    main()
