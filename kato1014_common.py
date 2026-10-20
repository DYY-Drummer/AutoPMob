"""2026-10-14 の加藤先生の指示（3・4・5）による再実験で共通に使う指標と集計.

指標は報告書 §3.5 の定義（実装から書き起こしたもの）:
  Recall@20(q) = |R_q ∩ Top_20(q)| / K
  AP(q)        = (1/K) Σ_{i=1..K'} i / r_(i)   （r_(1) < … < r_(K') は並べた中にある正解式の順位。
                 並べた中にない正解式は 0 として K で割る＝標準の定義）
  MAP          = テストのケースでの AP の平均（1 つの分割の値）
  MAP@20       = 上位 20 件で打ち切った AP の平均（20 位より後の正解式は P(e) = 0、K で割る。LLM と並べるため）
分割（乱数の種）10 通りの平均と不偏標準偏差を報告し、2 手法の差は分割を対にした
Wilcoxon の符号順位検定（両側、scipy の既定）で調べる。

per-case の記録（set_aware_reranker.py / external_baselines.py の --save-per-case）は、
各正解式の順位 "ranks"（昇順）を持つ。並べた件数 n_list（提案手法は候補の件数 N1、
baseline は DB の全式数）を超える順位は「並べた中にない」印（10,000 か 10^9）である。
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"
OUT = EXP / "kato1014"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
N_EQ = 11_146


def recall_at(ranks, k: int = 20) -> float:
    return sum(1 for r in ranks if r <= k) / len(ranks)


def ap_std(ranks, n_list: int) -> float:
    found = sorted(r for r in ranks if r <= n_list)
    return sum(i / r for i, r in enumerate(found, 1)) / len(ranks)


def ap_at(ranks, k: int = 20) -> float:
    """MAP@k の 1 ケース分：上位 k 件で打ち切った AP（k 位より後の正解式は P(e) = 0、K で割る）.

    LLM 直接生成（最大 20 本を返す）と並べるための指標（2026-10-15 本人の依頼）。K ≤ 10 < k なので
    K で割るのと min(K, k) で割るのは同じ。上位 k 件に全正解式があれば標準の AP と同じ値になる。
    """
    found = sorted(r for r in ranks if r <= k)
    return sum(i / r for i, r in enumerate(found, 1)) / len(ranks)


def rk(ranks) -> float:
    """旧主指標 Recall@K（K = 正解式の数）。対応表のためだけに使う."""
    K = len(ranks)
    return sum(1 for r in ranks if r <= K) / K


METRIC_FNS = {
    "Recall@20": lambda rec, n: recall_at(rec["ranks"], 20),
    "MAP": lambda rec, n: ap_std(rec["ranks"], n),
    "MAP@20": lambda rec, n: ap_at(rec["ranks"], 20),
    "Recall@K": lambda rec, n: rk(rec["ranks"]),
    "R-Precision": lambda rec, n: rk(rec["ranks"]),   # 2026-10-20：旧 Recall@K（K = 正解式の数）の呼び名
    "coverage": lambda rec, n: sum(1 for r in rec["ranks"] if r <= n) / len(rec["ranks"]),
}


def seed_means(records, metric: str, n_list: int, sub=None) -> dict:
    by = defaultdict(list)
    fn = METRIC_FNS[metric]
    for r in records:
        if sub is not None and not sub(r):
            continue
        by[r["seed"]].append(fn(r, n_list))
    return {s: float(np.mean(v)) for s, v in by.items() if v}


def seed_summary(sm: dict) -> dict:
    v = [sm[s] for s in sorted(sm)]
    return {"mean": float(np.mean(v)) if v else None,
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "n_seeds": len(v), "per_seed": {str(s): sm[s] for s in sorted(sm)}}


def paired(a: dict, b: dict) -> dict:
    """a − b を分割で対にして比べる（Wilcoxon の符号順位検定、両側）."""
    from scipy.stats import wilcoxon
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    out = {"n_seeds": len(seeds), "mean_a": float(np.mean([a[s] for s in seeds])),
           "mean_b": float(np.mean([b[s] for s in seeds])), "mean_delta": float(d.mean()),
           "std_delta": float(d.std(ddof=1)) if len(d) > 1 else None,
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    if len(d) > 1 and np.any(d != 0):
        out["p_wilcoxon"] = float(wilcoxon(d).pvalue)
    else:
        out["p_wilcoxon"] = None
    return out


def load_records(pattern: str, mode: str | None = None) -> tuple[list, int | None]:
    """--save-per-case の出力（seed ごとの JSON）から per-case の記録を集める.

    返り値: (記録のリスト, 並べた件数 n_list)。n_list は config の top_k（baseline は N_EQ）。
    """
    recs, n_list = [], None
    for f in sorted(glob.glob(str(pattern))):
        d = json.load(open(f, encoding="utf-8"))
        cfg = d.get("config", {})
        for m, r in d.get("results", {}).items():
            if mode is not None and m != mode:
                continue
            n = N_EQ if m == "baseline" or not r.get("n_features") else int(cfg.get("top_k", 50))
            n_list = n if n_list is None else n_list
            assert n_list == n, f"並べた件数がファイルで異なる: {f}"
            recs.extend(r.get("per_case", []))
    return recs, n_list


def summarize(records, n_list, metrics=("Recall@20", "MAP"), sub=None) -> dict:
    return {m: seed_summary(seed_means(records, m, n_list, sub)) for m in metrics}


def compare(rec_a, n_a, rec_b, n_b, metrics=("Recall@20", "MAP"), sub=None) -> dict:
    return {m: paired(seed_means(rec_a, m, n_a, sub), seed_means(rec_b, m, n_b, sub))
            for m in metrics}


def seed_of(path: str) -> int:
    return int(re.search(r"__(\d+)\.json$", path).group(1))


def dump(obj, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {p}")
    return p
