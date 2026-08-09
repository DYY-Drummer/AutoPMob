"""第1段の候補窓が第2段の上限を決めることの定量化.

第1段（0.7*文章類似度 + 0.3*変数の一致度 の上位 k 件）が正解式を候補に含める
割合＝被覆率を、k と絞り込み方式ごとに測る。被覆率は第2段 Recall@K_correct の
上限なので、第2段をどれだけ改良しても超えられない天井を与える。

方式: blend（現行）/ 文章類似度のみ / 変数の一致度のみ / 意味の近さのみ

出力: experiments/stage1_coverage.json
      docs/figures/fig_stage1_coverage.png / .pdf（(a) 被覆率, (b) 救済可能性）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
KS = [10, 25, 50, 100, 200, 400]
METHODS = [
    ("blend", "現行（文章 0.7 ＋ 変数 0.3）", "#1f77b4"),
    ("jaccard", "変数の一致度のみ", "#2ca02c"),
    ("text", "文章類似度のみ", "#d62728"),
    ("svd", "意味の近さのみ", "#ff7f0e"),
]


def coverage(order: list, corr: set, k: int) -> float:
    """上位 k 件が正解式を含む割合（正解が空なら nan）."""
    if not corr:
        return float("nan")
    return len(corr & set(order[:k])) / len(corr)


def mean_sem(vals) -> dict:
    a = np.asarray([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "sem": sem, "n": int(a.size)}


def run(out_json="experiments/stage1_coverage.json",
        out_fig="docs/figures/fig_stage1_coverage.png", seed=42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity

    from two_stage_query_conditioned import (
        load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
        case_text, io_vars, jaccard,
    )
    from analyze_pfi import keep_setting_A

    eqs = load_equations()
    cases = [c for c in load_cases() if keep_setting_A(str(c.get("variant_type", "")))]
    ek, et, ev = [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki]
          for c in cases]
    keep = [i for i in range(len(cases)) if cl[i]]
    cases = [cases[i] for i in keep]; cl = [cl[i] for i in keep]
    n_eq = len(ek)
    print(f"設定A ケース {len(cases)} 件 / 式 {n_eq} 件")

    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    X_ctx = tfidf.transform([case_text(c) for c in cases])
    svd = TruncatedSVD(n_components=256, random_state=seed)
    E = svd.fit_transform(X_eq); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    Q = svd.transform(tfidf.transform([case_text(c, io=True) for c in cases]))
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    svd_sim = (Q @ E.T).astype(np.float32)
    ios = [io_vars(c) for c in cases]

    cov = {m: {k: [] for k in KS} for m, _, _ in METHODS}
    full = {m: {k: [] for k in KS} for m, _, _ in METHODS}
    for ci, c in enumerate(cases):
        corr = set(cl[ci])
        ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
        vs = np.array([jaccard(ios[ci], ev[j]) for j in range(n_eq)], dtype=np.float32)
        orders = {
            "blend": np.argsort(-(0.7 * ts + 0.3 * vs)).tolist(),
            "jaccard": np.argsort(-vs).tolist(),
            "text": np.argsort(-ts).tolist(),
            "svd": np.argsort(-svd_sim[ci]).tolist(),
        }
        for m, order in orders.items():
            for k in KS:
                h = coverage(order, corr, k)
                cov[m][k].append(h)
                full[m][k].append(1.0 if h == 1.0 else 0.0)

    out = {
        "config": {"setting": "A", "n_cases": len(cases), "n_eq": n_eq, "svd_seed": seed},
        "note": "被覆率は第2段 Recall@K_correct の上限",
        "coverage": {m: {str(k): mean_sem(cov[m][k]) for k in KS} for m, _, _ in METHODS},
        "full_coverage_rate": {m: {str(k): float(np.mean(full[m][k])) for k in KS}
                               for m, _, _ in METHODS},
    }
    Path(ROOT / out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")
    _figure(out, ROOT / out_fig)
    print(f"Saved figure: {out_fig}")
    for m, lab, _ in METHODS:
        row = "  ".join(f"k={k}:{out['coverage'][m][str(k)]['mean']:.3f}" for k in KS)
        print(f"  {lab:<22s} {row}")
    return out


def _figure(out, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import japanize_matplotlib  # noqa
    except Exception:
        for fam in ["Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "AppleGothic"]:
            try:
                plt.rcParams["font.family"] = fam; break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # (a) 被覆率 vs k
    for m, lab, col in METHODS:
        ys = [out["coverage"][m][str(k)]["mean"] for k in KS]
        es = [out["coverage"][m][str(k)]["sem"] for k in KS]
        axes[0].errorbar(KS, ys, yerr=es, marker="o", lw=2.0, capsize=3, color=col, label=lab)
    axes[0].axvline(50, color="#888888", ls=":", lw=1.2)
    axes[0].set_ylim(0.16, 1.03)
    axes[0].text(50, 0.18, "本実験の設定 k=50", fontsize=9, color="#555555",
                 ha="center", va="bottom")
    axes[0].set_xscale("log")
    axes[0].set_xticks(KS); axes[0].set_xticklabels([str(k) for k in KS])
    axes[0].set_title("(a) 第1段の被覆率＝第2段の上限")
    axes[0].set_xlabel("候補件数 k（対数軸）")
    axes[0].set_ylabel("正解式が候補に入る割合（平均±標準誤差）")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=9, loc="lower right")

    # (b) 救済可能性
    pc = json.load(open(ROOT / "experiments/case_outcomes_seed42.json", encoding="utf-8"))
    r = pc["topic_rescue_fraction"]
    names = [("svd_sim", "意味の近さ"), ("domain", "分野の一致"),
             ("specificity", "式の特化度"), ("io_jaccard", "変数の一致度"),
             ("text_sim", "文章類似度")]
    y = np.arange(len(names))
    vals = [r[k]["mean"] for k, _ in names]
    errs = [r[k]["sem"] for k, _ in names]
    cols = ["#d62728" if k in ("svd_sim", "domain", "text_sim") else "#1f77b4"
            for k, _ in names]
    axes[1].barh(y, vals, xerr=errs, color=cols, error_kw=dict(capsize=3, lw=1))
    axes[1].axvline(0.5, color="#555555", lw=1.2, ls="--")
    axes[1].text(0.508, 0.03, "0.5＝偶然と同じ", fontsize=9, color="#555555",
                 transform=axes[1].get_xaxis_transform())
    axes[1].set_yticks(y); axes[1].set_yticklabels([lab for _, lab in names])
    axes[1].set_xlim(0, 0.8)
    axes[1].set_title("(b) その特徴量を強めれば取り逃しを救えるか")
    axes[1].set_xlabel("見逃した正解式が誤って上位に来た式を上回る割合（平均±標準誤差）")
    axes[1].grid(alpha=0.25, axis="x")

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure-only", action="store_true",
                    help="計測をやり直さず、保存済み JSON から作図のみ行う")
    a = ap.parse_args()
    if a.figure_only:
        doc = json.load(open(ROOT / "experiments/stage1_coverage.json", encoding="utf-8"))
        _figure(doc, ROOT / "docs/figures/fig_stage1_coverage.png")
        print("Saved figure (figure-only): docs/figures/fig_stage1_coverage.png")
    else:
        run()
