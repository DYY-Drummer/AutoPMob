"""分野ラベル合併が domain 特徴量の精度寄与を変えるかの検証（負の結果の補強）.

背景：基本版 domain 特徴（feature6）も集合版 gDom も、照合対象は 32 収集分野でなく
抽出時に LLM が式ごとに付けた自由記述ラベル（11,146 式に 1,998 種）。完全一致・部分
文字列一致がまばらにしか発火せず寄与ゼロ（修論 §6.6）。本スクリプトは「ラベルを合併
（粗く）すれば効くようになるか」を検証する。

方法：コミット済み set_aware_reranker.run_mode を呼び、eq_domains(ed) だけを合併ラベルに
差し替えて reranker-2+Dom（基本 2 特徴量 + 基本版 domain 特徴）を再学習。base は
domain 不使用の reranker-2。設定 A・層化分割・top_k=50・10 seed（run_base_ablation.sh と同一条件）。

合併粒度：
  orig : 元の自由記述ラベル（1,998 種）  ← 既存 experiments/xd/reranker-2_Dom__*.json を再利用
  L1   : 軽い正規化（小文字化・記号除去・語尾複数形統合）
  K100 : L2 正規化 TF-IDF 上の KMeans で 100 群
  K40  : 同 40 群（32 収集分野スケール）

使い方：
  python run_domain_merge_ablation.py                 # 全 (merge, seed) を逐次実行 → 集計
  python run_domain_merge_ablation.py --merge K40 --seed 42 --out out.json   # 単発（並列用）
出力：experiments/domain_merge_stats.json（per-seed・平均・Δ・t/Wilcoxon p）
"""
from __future__ import annotations
import sys, json, re, argparse, statistics
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from two_stage_query_conditioned import (  # type: ignore
    load_equations, load_cases, eq_key, eq_text, eq_vars, norm, case_src,
)
from set_aware_reranker import run_mode, get_src  # type: ignore

SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
VARIANTS = ["original", "multisource_", "dae_"]
XD = ROOT / "experiments" / "xd"


def vmatch(cv):
    return any(cv.startswith(w) if w.endswith("_") else cv == w for w in VARIANTS)


def l1_key(s):
    t = re.sub(r"[^a-z0-9 ]+", " ", s.strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    w = t.split(" ")
    if len(w[-1]) > 3 and w[-1].endswith("s") and not w[-1].endswith("ss"):
        w[-1] = w[-1][:-1]
    return " ".join(w)


def build_maps(labels):
    """labels: 非空 distinct ラベルのソート済みリスト。決定論的。"""
    maps = {"orig": {d: d for d in labels}, "L1": {d: l1_key(d) for d in labels}}
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    X = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1).fit_transform(labels)
    for K in (100, 40):
        cids = KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(X)
        maps[f"K{K}"] = {d: f"C{int(c)}" for d, c in zip(labels, cids)}
    for m in maps.values():
        m[""] = ""
    return maps


def load_lists():
    eqs = load_equations(); cases = load_cases()
    cases = [c for c in cases if vmatch(c.get("variant_type", ""))]
    ek, et, ev, ed, es = [], [], [], [], []
    for e in eqs:
        if not eq_key(e):
            continue
        ek.append(eq_key(e)); et.append(eq_text(e)); ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or "")); es.append(get_src(e))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]
    return cases, ek, et, ev, ed, es, cl, cs


def run_seed(mode, ed_use, ctx, seed):
    cases, ek, et, ev, _, es, cl, cs = ctx
    r = run_mode(mode, [seed], cases, ek, et, ev, ed_use, es, cl, cs,
                 50, 15, 1e-3, save_per_case=False, hidden_dim=64, margin=0.1,
                 batch_size=16, n_neg_samples=8, weight_decay=1e-4,
                 loss_type="pairwise", hard_neg=False, greedy=False,
                 split_mode="stratified", train_greedy=False,
                 greedy_train_cap=8, stop_dof=False)
    return float(r["Recall@K_correct"]["mean"])


def xd_val(mode_file, seed):
    d = json.load(open(XD / f"{mode_file}__{seed}.json"))["results"]
    m = d.get(mode_file.replace("_", "+")) or list(d.values())[0]
    return m["Recall@K_correct"]["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", default=None, help="単発実行する合併粒度")
    ap.add_argument("--seed", type=int, default=None, help="単発実行する seed")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ctx = load_lists()
    ed = ctx[4]
    labels = sorted({d for d in ed if d})
    maps = build_maps(labels)
    ed_by = {n: [mm.get(d, d) for d in ed] for n, mm in maps.items()}
    ng = {n: len({v for v in e if v}) for n, e in ed_by.items()}

    if a.merge and a.seed is not None:      # 単発（並列用）
        v = run_seed("reranker-2+Dom", ed_by[a.merge], ctx, a.seed)
        out = a.out or f"domain_merge_{a.merge}_{a.seed}.json"
        json.dump({"merge": a.merge, "seed": a.seed, "recall_k": v}, open(out, "w"))
        print(f"OK reranker-2+Dom@{a.merge} seed={a.seed}: {v:.4f}")
        return

    # 全逐次実行 + 集計
    base = [xd_val("reranker-2", s) for s in SEEDS]           # 既存再利用（domain不使用）
    conds = {"orig": [xd_val("reranker-2_Dom", s) for s in SEEDS]}  # 既存再利用
    for merge in ("L1", "K100", "K40"):
        conds[merge] = [run_seed("reranker-2+Dom", ed_by[merge], ctx, s) for s in SEEDS]

    from scipy import stats as st
    out = {"description": __doc__.strip().splitlines()[0], "base_mode": "reranker-2",
           "base_mean": statistics.mean(base), "base_per_seed": base, "seeds": SEEDS,
           "n_groups": ng, "conditions": {}}
    for mg, v in conds.items():
        delta = [x - y for x, y in zip(v, base)]
        out["conditions"][mg] = {"mode": "reranker-2+Dom", "n_groups": ng[mg],
            "mean": statistics.mean(v), "mean_delta": statistics.mean(delta),
            "p_ttest": st.ttest_rel(v, base).pvalue,
            "p_wilcoxon": st.wilcoxon(v, base).pvalue, "per_seed": v}
    json.dump(out, open(ROOT / "experiments" / "domain_merge_stats.json", "w"),
              ensure_ascii=False, indent=2)
    print("saved experiments/domain_merge_stats.json")


if __name__ == "__main__":
    main()
