"""Permutation Feature Importance (PFI) — 最良モデル reranker-10S 1 つに対して
どの特徴量が精度（Recall@K_correct）を担っているかを、土台を複数訓練せずに測る.

背景（この実験の位置づけ）:
  これまでの内訳分解はアブレーション（特徴量部分集合ごとにモデルを再訓練）で行った。
  PFI は「訓練済みの最良モデルがどれだけその特徴量に依存しているか（reliance）」を、
  モデル 1 つのまま測る独立手法。2 手法が一致すれば頑健性の裏付けになる。

設計上の注意（相関・冗長な特徴量向け）:
  補完性/一貫性・特化度/jaccard/被覆は変数重なり情報を共有し冗長。
  1 特徴だけ置換すると相棒が信号を復元するため個別 PFI は過小評価になる。
  → (1) 群 PFI（変数重なり群 vs 話題群を丸ごと置換）を主役にする。
     (2) 群は「共有置換」（群内列に同一の行置換を適用）で群内相関を保つ。
     (3) 冗長性指標 = 群 PFI − 群内個別 PFI の和（正に大きいほど冗長）。

置換の範囲:
  Recall@K_correct はケース単位の順位付け指標なので、既定は「ケース内置換」
  （各ケースの候補行内で対象列をシャッフル）。頑健性確認として「全体置換」も併走。

出力: experiments/pfi_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from two_stage_query_conditioned import (  # type: ignore
    load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
    case_text, io_vars, in_vars, out_vars, case_src, Reranker, stage1,
)
from set_aware_reranker import (
    get_src, stratified_src_split, compute_features_with_set,
)
from evaluate_multi_eq import compute_all_ranks, case_metrics, aggregate_metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

# --- 最良モデルと同一のハイパーパラメータ（set_aware_reranker.py の既定）---
SET_MASK = ("Comp", "Coh", "Dom")   # reranker-10S
TOP_K, EPOCHS, LR = 50, 15, 1e-3
HIDDEN, MARGIN, BATCH, N_NEG, WD = 64, 0.1, 16, 8, 1e-4
DATA_SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
N_PERM = 20  # 各特徴の置換繰り返し回数（エラーバー用）
METRIC = "Recall@K_correct"

# 特徴量メタ（列 index, 名前, 群）
FEATURES = [
    (0, "text_sim",    "topic"),
    (1, "io_jaccard",  "var"),
    (2, "svd_sim",     "topic"),
    (3, "input_cov",   "var"),
    (4, "output_cov",  "var"),
    (5, "specificity", "var"),
    (6, "domain",      "topic"),
    (7, "gComp",       "var"),
    (8, "gCoh",        "var"),
    (9, "gDom",        "topic"),
]
VAR_COLS = [c for c, _, g in FEATURES if g == "var"]      # [1,3,4,5,7,8]
TOPIC_COLS = [c for c, _, g in FEATURES if g == "topic"]  # [0,2,6,9]

# 置換対象（label -> 列リスト）。単独 10 個 + 群 + 冗長ペア。
PERM_TARGETS = {name: [c] for c, name, _ in FEATURES}
PERM_TARGETS["GROUP_var"] = VAR_COLS
PERM_TARGETS["GROUP_topic"] = TOPIC_COLS
PERM_TARGETS["PAIR_Comp_Coh"] = [7, 8]  # 冗長ペア：joint vs 個別和


def per_case_signal(base_R: float, perm_Rs: list) -> dict:
    """置換前 Recall と N_PERM 回の置換後 Recall から per-case 信号を計算.

    flip: base_R==1.0（baseline 解済み）かつ半数以上の置換で Recall<1.0 になったら 1。
    drop: 期待低下 base_R - mean(perm_Rs)（全ケース対象、連続の主信号）。
    """
    perm = np.asarray(perm_Rs, dtype=float)
    perm_mean = float(perm.mean()) if perm.size else float("nan")
    n_below = int(np.sum(perm < 1.0))
    flip_rate = n_below / perm.size if perm.size else 0.0
    flip = 1 if (base_R == 1.0 and flip_rate >= 0.5) else 0
    return {"perm_R_mean": perm_mean, "drop": float(base_R - perm_mean),
            "flip": flip, "flip_rate": float(flip_rate)}


def sep_for_feature(feats, cands, corr, col: int) -> float:
    """そのケースで特徴 col が正解を不正解からどれだけ分離しているか.

    sep = mean(正解候補行の col) - mean(不正解候補行の col)。
    正解/不正解いずれかの行が無ければ nan。feats 行は cands と同順。
    """
    corr_mask = np.array([c in corr for c in cands], dtype=bool)
    pos = feats[corr_mask, col]
    neg = feats[~corr_mask, col]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(pos.mean() - neg.mean())


def keep_setting_A(v: str) -> bool:
    return v == "original" or v.startswith("multisource_") or v.startswith("dae_")


def train_and_cache(seed, cases, ek, et, ev, ed, es, cl, cs):
    """reranker-10S を run_mode と同一手順で訓練し、テストケースの
    (cands, feats[n×10], corr) をキャッシュして返す."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    n_eq = len(ek)
    n_feat = 7 + len(SET_MASK)

    feats_strat = [(len(cl[i]), len(in_vars(cases[i])), len(out_vars(cases[i])),
                    len({ek[j].split("__")[0] for j in cl[i]}))
                   for i in range(len(cases))]
    split = stratified_src_split(cs, feats_strat, seed)
    tr = [i for i, s in enumerate(cs) if s in split["train"] and cl[i]]
    te = [i for i, s in enumerate(cs) if s in split["test"] and cl[i]]

    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    X_ctx = tfidf.transform([case_text(c) for c in cases])
    ios = [io_vars(c) for c in cases]

    svd = TruncatedSVD(n_components=256, random_state=seed)
    E = svd.fit_transform(X_eq); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    Q = svd.transform(tfidf.transform([case_text(c, io=True) for c in cases]))
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    svd_sim = Q @ E.T

    model = Reranker(n_feat, HIDDEN)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rng = random.Random(seed)
    for ep in range(EPOCHS):
        rng.shuffle(tr)
        for s in range(0, len(tr), BATCH):
            batch = tr[s:s + BATCH]
            losses = []
            for ci in batch:
                corr = set(cl[ci])
                if not corr:
                    continue
                cands = stage1(ci, X_ctx, X_eq, ios[ci], ev, TOP_K)
                pos = [j for j in cands if j in corr]
                neg = [j for j in cands if j not in corr]
                if not pos or not neg:
                    continue
                ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                feats = compute_features_with_set(
                    cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                    svd_sim[ci], ev, ed, case_text(cases[ci]),
                    None, ios[ci], False, SET_MASK,
                )
                c2k = {j: k for k, j in enumerate(cands)}
                scores = model(torch.tensor(feats, dtype=torch.float32))
                pos_k = [c2k[p] for p in pos]
                ns = min(N_NEG, len(neg))
                chosen = rng.sample(neg, ns)
                for pk in pos_k:
                    for ng in chosen:
                        losses.append(F.relu(MARGIN - scores[pk] + scores[c2k[ng]]))
            if losses:
                loss = torch.stack(losses).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    cache = []
    with torch.no_grad():
        for ci in te:
            corr = set(cl[ci])
            if not corr:
                continue
            cands = stage1(ci, X_ctx, X_eq, ios[ci], ev, TOP_K)
            ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
            feats = compute_features_with_set(
                cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                svd_sim[ci], ev, ed, case_text(cases[ci]),
                None, ios[ci], False, SET_MASK,
            )
            cache.append({
                "feats": np.array(feats, dtype=np.float32),
                "corr": corr, "cands": cands,
                "variant": norm(cases[ci].get("variant_type") or "?"),
                "case_id": cases[ci].get("case_id", f"idx_{ci}"),
                "n_input": len(in_vars(cases[ci])),
                "n_output": len(out_vars(cases[ci])),
                "n_sources": len({ek[j].split("__")[0] for j in corr}),
            })
    return model, cache


def score_to_RK(model, cache, feat_override=None, return_per_case=False):
    """キャッシュ各ケースを（任意で置換後 feats で）採点し集約 Recall@K_correct を返す.

    return_per_case=True のとき (aggregate, per_case_list) を返す。
    per_case_list[i] は cache 順のケース単位 Recall@K_correct。
    """
    case_results = []
    with torch.no_grad():
        for idx, rec in enumerate(cache):
            feats = rec["feats"] if feat_override is None else feat_override[idx]
            scores = model(torch.tensor(feats, dtype=torch.float32)).numpy().ravel()
            order = sorted(range(len(rec["cands"])), key=lambda k: -scores[k])
            ranked = [rec["cands"][k] for k in order]
            ranks = compute_all_ranks(ranked, rec["corr"], miss_rank=10_000)
            cm = case_metrics(ranks)
            for kf in ("variant", "case_id", "n_input", "n_output", "n_sources"):
                cm[kf] = rec[kf]
            case_results.append(cm)
    agg = aggregate_metrics(case_results).get(METRIC, 0.0)
    if return_per_case:
        per_case = [cm.get(METRIC, 0.0) for cm in case_results]
        return agg, per_case
    return agg


def permute_feats(cache, cols, perm_seed, scope):
    """cache の各ケース feats をコピーし、対象 cols を置換（群は共有行置換）した行列列を返す."""
    prng = np.random.RandomState(perm_seed)
    if scope == "global":
        # 全テスト候補行をプールして単一の行置換（列ブロックは共有）
        blocks = [rec["feats"] for rec in cache]
        sizes = [b.shape[0] for b in blocks]
        stacked = np.vstack(blocks)
        perm = prng.permutation(stacked.shape[0])
        stacked_perm = stacked.copy()
        stacked_perm[:, cols] = stacked[np.ix_(perm, cols)]
        out, o = [], 0
        for n in sizes:
            out.append(stacked_perm[o:o + n]); o += n
        return out
    # case: 各ケース内で行置換
    out = []
    for rec in cache:
        f = rec["feats"].copy()
        n = f.shape[0]
        perm = prng.permutation(n)
        f[:, cols] = rec["feats"][np.ix_(perm, cols)]
        out.append(f)
    return out


def run(n_seeds=None, n_perm=None, force=False):
    from scipy import stats
    # --seeds/--n-perm を指定した smoke 実行は正本を破壊しないよう別ファイルに退避する
    # （--force 指定時のみ正本パスへ縮小スケールで上書きすることを許す）。
    smoke = (n_seeds is not None or n_perm is not None) and not force
    seeds = DATA_SEEDS[:n_seeds] if n_seeds is not None else DATA_SEEDS
    n_perm = n_perm or N_PERM
    eqs = load_equations(); cases_all = load_cases()
    cases = [c for c in cases_all if keep_setting_A(str(c.get("variant_type", "")))]
    ek, et, ev, ed, es = [], [], [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or "")); es.append(get_src(e))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]
    print(f"設定A ケース {len(cases)} 件 / 式 {len(ek)} 件")

    scopes = ["case", "global"]
    drops = {sc: {lab: [] for lab in PERM_TARGETS} for sc in scopes}
    baselines = []

    # per-case（scope="case" のみ、単独10特徴＋GROUP_var）
    PC_LABELS = [name for _, name, _ in FEATURES] + ["GROUP_var"]
    COL_OF = {name: c for c, name, _ in FEATURES}
    percase_records = []

    for si, seed in enumerate(seeds):
        model, cache = train_and_cache(seed, cases, ek, et, ev, ed, es, cl, cs)
        base, base_pc = score_to_RK(model, cache, return_per_case=True)
        baselines.append(base)
        print(f"[seed={seed}] baseline {METRIC} = {base:.4f}  (test {len(cache)} ケース)")

        # sep を seed 内で1回（単独特徴のみ）
        seps = {name: [sep_for_feature(rec["feats"], rec["cands"], rec["corr"], COL_OF[name])
                       for rec in cache]
                for name in COL_OF}
        # per-case 置換後 Recall の蓄積器
        pc_perm = {lab: [[] for _ in cache] for lab in PC_LABELS}

        for sc in scopes:
            for lab, cols in PERM_TARGETS.items():
                ds = []
                collect_pc = (sc == "case" and lab in PC_LABELS)
                for pi in range(n_perm):
                    pf = permute_feats(cache, cols, perm_seed=seed * 1000 + pi, scope=sc)
                    if collect_pc:
                        rk, rk_pc = score_to_RK(model, cache, feat_override=pf, return_per_case=True)
                        for i, v in enumerate(rk_pc):
                            pc_perm[lab][i].append(v)
                    else:
                        rk = score_to_RK(model, cache, feat_override=pf)
                    ds.append(base - rk)
                drops[sc][lab].append(float(np.mean(ds)))

        # per-case レコード生成
        for lab in PC_LABELS:
            for i, rec in enumerate(cache):
                sig = per_case_signal(base_pc[i], pc_perm[lab][i])
                percase_records.append({
                    "seed": seed, "case_id": rec["case_id"], "feature": lab,
                    "base_R": float(base_pc[i]), "perm_R_mean": sig["perm_R_mean"],
                    "drop": sig["drop"], "flip": sig["flip"], "flip_rate": sig["flip_rate"],
                    "sep": (seps[lab][i] if lab in seps else None),
                    "n_correct": len(rec["corr"]), "n_input": rec["n_input"],
                    "n_output": rec["n_output"], "n_sources": rec["n_sources"],
                    "variant": rec["variant"],
                })

    # --- 集約・出力 ---
    out = {"metric": METRIC, "n_seeds": len(seeds), "n_perm": n_perm,
           "baseline_mean": float(np.mean(baselines)),
           "baseline_std": float(np.std(baselines, ddof=1)),
           "results": {}}
    for sc in scopes:
        res = {}
        for lab in PERM_TARGETS:
            arr = np.array(drops[sc][lab])
            t, p = stats.ttest_1samp(arr, 0.0)
            res[lab] = {"importance_mean": round(float(arr.mean()), 4),
                        "importance_std": round(float(arr.std(ddof=1)), 4),
                        "t": round(float(t), 3), "p": round(float(p), 4)}
        # 冗長性指標
        sum_var = sum(res[n]["importance_mean"] for _, n, g in FEATURES if g == "var")
        sum_topic = sum(res[n]["importance_mean"] for _, n, g in FEATURES if g == "topic")
        res["_redundancy"] = {
            "group_var": res["GROUP_var"]["importance_mean"],
            "sum_individual_var": round(sum_var, 4),
            "var_redundancy(group-sum)": round(res["GROUP_var"]["importance_mean"] - sum_var, 4),
            "group_topic": res["GROUP_topic"]["importance_mean"],
            "sum_individual_topic": round(sum_topic, 4),
            "pair_CompCoh_joint": res["PAIR_Comp_Coh"]["importance_mean"],
            "sum_Comp_Coh": round(res["gComp"]["importance_mean"] + res["gCoh"]["importance_mean"], 4),
        }
        out["results"][sc] = res

    outp = ROOT / "experiments" / ("pfi_results_smoke.json" if smoke else "pfi_results.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp, "w"), ensure_ascii=False, indent=2)

    pc_out = {
        "config": {"setting": "A", "set_mask": list(SET_MASK), "top_k": TOP_K, "epochs": EPOCHS},
        "metric": METRIC, "n_seeds": len(seeds), "n_perm": n_perm, "scope": "case",
        "features": PC_LABELS, "records": percase_records,
    }
    pc_path = ROOT / "experiments" / ("pfi_per_case_smoke.json" if smoke else "pfi_per_case.json")
    json.dump(pc_out, open(pc_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved per-case: {pc_path}  ({len(percase_records)} records)")
    if smoke:
        print(f"[smoke] --seeds/--n-perm 指定のため正本を保護し {outp.name} / {pc_path.name} に退避出力した。"
              f"正本 experiments/pfi_results.json ・ experiments/pfi_per_case.json は変更していない。"
              f"正本を縮小スケールで上書きするには --force を指定すること。")

    # --- 見やすい表示 ---
    print(f"\n{'='*72}")
    print(f"baseline {METRIC} = {out['baseline_mean']:.4f} ± {out['baseline_std']:.4f}  "
          f"({len(seeds)} seeds, N_perm={n_perm})")
    for sc in scopes:
        print(f"\n=== 置換範囲: {sc} ===")
        res = out["results"][sc]
        print(f"{'特徴量/群':<16s} {'群':>6s} {'PFI(低下)':>12s} {'±std':>8s} {'p':>8s}")
        # 個別 → 大きい順
        singles = [(n, g) for _, n, g in FEATURES]
        for n, g in sorted(singles, key=lambda x: -res[x[0]]["importance_mean"]):
            r = res[n]
            print(f"{n:<16s} {g:>6s} {r['importance_mean']:>+12.4f} "
                  f"{r['importance_std']:>8.4f} {r['p']:>8.4f}")
        print("  --- 群 ---")
        for lab in ("GROUP_var", "GROUP_topic", "PAIR_Comp_Coh"):
            r = res[lab]
            print(f"{lab:<16s} {'':>6s} {r['importance_mean']:>+12.4f} "
                  f"{r['importance_std']:>8.4f} {r['p']:>8.4f}")
        rd = res["_redundancy"]
        print(f"  冗長性: 変数群PFI {rd['group_var']:+.4f} vs 個別和 {rd['sum_individual_var']:+.4f} "
              f"→ 差 {rd['var_redundancy(group-sum)']:+.4f}")
        print(f"          Comp+Coh 同時 {rd['pair_CompCoh_joint']:+.4f} vs 個別和 {rd['sum_Comp_Coh']:+.4f}")
    print(f"\nSaved: {outp}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None, help="先頭N seedのみ使用（smoke）")
    ap.add_argument("--n-perm", type=int, default=None, help="置換回数（smoke）")
    ap.add_argument("--force", action="store_true",
                    help="--seeds/--n-perm 指定時でも正本パス（pfi_results.json 等）に書き込む")
    a = ap.parse_args()
    run(n_seeds=a.seeds, n_perm=a.n_perm, force=a.force)
