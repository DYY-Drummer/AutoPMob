"""補完性 gComp と一貫性 gCoh がなぜ同程度の寄与になるのかの診断.

(A) 既存の per-case 結果（無料）：
    各ケースで ΔComp = R@K(7+Comp) - R@K(7)、ΔCoh = R@K(7+Coh) - R@K(7) を計算し、
    「両者が同じケースで効いているか」を相関で見る。高相関＝冗長（同じ信号）。
(B) 特徴量値そのもの（1〜数シードで再計算）：
    テストケースの候補について gComp・gCoh を計算し、
    - corr(gComp, gCoh)：本当に「逆の指標」か（負相関か）
    - 正解/不正解を分ける力（AUC）：どちらも同じだけ効くか
"""
from __future__ import annotations
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent


# ---------- (A) per-case の改善が同じケースで起きるか ----------
def load_percase(mode, pat):
    """(seed, case_id) -> Recall@K_correct"""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "experiments" / "xd" / f"{pat}__*.json"))):
        d = json.load(open(f))
        for r in d["results"][mode].get("per_case", []):
            out[(r["seed"], r["case_id"])] = r["Recall@K_correct"]
    return out


def part_A():
    base = load_percase("reranker-7", "reranker-7")
    comp = load_percase("reranker-7+Comp", "reranker-7_Comp")
    coh = load_percase("reranker-7+Coh", "reranker-7_Coh")
    keys = sorted(set(base) & set(comp) & set(coh))
    dC = np.array([comp[k] - base[k] for k in keys])
    dK = np.array([coh[k] - base[k] for k in keys])
    print(f"[A] 対象ケース数（seed×case）: {len(keys)}")
    print(f"    平均 ΔComp = {dC.mean():+.4f},  平均 ΔCoh = {dK.mean():+.4f}")
    # 改善が起きたケースだけで相関（0 が多いと相関が薄まるため両方見る）
    r_all = np.corrcoef(dC, dK)[0, 1]
    nz = (dC != 0) | (dK != 0)
    r_nz = np.corrcoef(dC[nz], dK[nz])[0, 1] if nz.sum() > 2 else float("nan")
    print(f"    corr(ΔComp, ΔCoh) 全ケース = {r_all:.3f},  改善が動いたケースのみ = {r_nz:.3f}  (n={int(nz.sum())})")
    # 同符号率：Comp が助けたケースで Coh も助けたか
    both_help = ((dC > 0) & (dK > 0)).sum()
    comp_help = (dC > 0).sum(); coh_help = (dK > 0).sum()
    print(f"    Comp が改善したケース {comp_help}、Coh が改善したケース {coh_help}、両方改善 {both_help}")
    print(f"    → Comp改善のうち Coh も改善した割合 = {both_help/comp_help:.2f}" if comp_help else "")


# ---------- (B) 特徴量値の相関・弁別力 ----------
def part_B(seeds=(42, 123, 456)):
    import sys
    sys.path.insert(0, str(ROOT))
    from two_stage_query_conditioned import (
        load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
        case_text, io_vars, in_vars, out_vars, case_src,
    )
    from set_aware_reranker import (
        compute_set_aware_features, stratified_src_split,
    )
    import random
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    eqs = load_equations(); cases_all = load_cases()
    # 設定A
    def keep(v):
        return v == "original" or v.startswith("multisource_") or v.startswith("dae_")
    cases = [c for c in cases_all if keep(str(c.get("variant_type", "")))]
    ek, et, ev = [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k: continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
    ki = {k: i for i, k in enumerate(ek)}
    ed = [""] * len(ek)
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]
    ios = [io_vars(c) for c in cases]

    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    X_ctx = tfidf.transform([case_text(c) for c in cases])
    n_eq = len(ek)

    comp_vals, coh_vals, labels = [], [], []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed)
        feats = [(len(cl[i]), len(in_vars(cases[i])), len(out_vars(cases[i])),
                  len({ek[j].split("__")[0] for j in cl[i]})) for i in range(len(cases))]
        split = stratified_src_split(cs, feats, seed)
        te = [i for i, s in enumerate(cs) if s in split["test"] and cl[i]]
        for ci in te:
            corr = set(cl[ci])
            ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
            vs = np.array([len(ios[ci] & ev[j]) / (len(ios[ci] | ev[j]) or 1) for j in range(n_eq)])
            order = np.argsort(-(0.7 * ts + 0.3 * vs))
            cands = order[:50].tolist()
            setf = compute_set_aware_features(cands, ios[ci], ev, ed, output_vars=out_vars(cases[ci]))
            for k, j in enumerate(cands):
                comp_vals.append(setf[k, 0]); coh_vals.append(setf[k, 1])
                labels.append(1 if j in corr else 0)

    comp_vals = np.array(comp_vals); coh_vals = np.array(coh_vals); labels = np.array(labels)
    print(f"\n[B] 候補総数 {len(labels)}（正解 {labels.sum()}, 不正解 {len(labels)-labels.sum()}）  seeds={list(seeds)}")
    print(f"    corr(gComp, gCoh) = {np.corrcoef(comp_vals, coh_vals)[0,1]:+.3f}  （負なら『逆の指標』・正なら同方向）")

    def auc(x, y):
        # Mann-Whitney U を AUC に換算
        from scipy.stats import mannwhitneyu
        pos = x[y == 1]; neg = x[y == 0]
        if len(pos) == 0 or len(neg) == 0: return float("nan")
        u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
        return u / (len(pos) * len(neg))
    aC = auc(comp_vals, labels); aK = auc(coh_vals, labels)
    print(f"    正解を分ける力 AUC：gComp = {aC:.3f}, gCoh = {aK:.3f}  （0.5=無力、1 or 0=完全）")
    print(f"    正解式での平均：gComp={comp_vals[labels==1].mean():.3f} gCoh={coh_vals[labels==1].mean():.3f}")
    print(f"    不正解式での平均：gComp={comp_vals[labels==0].mean():.3f} gCoh={coh_vals[labels==0].mean():.3f}")


if __name__ == "__main__":
    print("=== (A) 補完性・一貫性は同じケースで効くか（既存 per-case, 無料）===")
    part_A()
    print("\n=== (B) 特徴量値の相関・弁別力（3 シードのテスト候補で再計算）===")
    part_B()
