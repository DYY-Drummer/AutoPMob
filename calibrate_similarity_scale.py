"""コサイン類似度の物差しを作る：0.218 はどのくらい「違う」のか.

ケース間のベクトル類似度の中央値 0.218 は、それ自体では近いとも遠とも判断できない。
そこで同じ空間の中に「明らかに同じ内容の組」と「明らかに無関係な組」を置き、
両者の間のどこに 0.218 が来るかを測る。

比較する組:
  言い換え     : 同じケースの説明文を書き換えただけの組（内容は同一）
  同じ核モデル : 同じ元モデルから入出力を変えて作った組（説明文は同じ）
  同じ文献     : 正解式が同じ文献に由来する別々のケースの組
  無関係       : 出典文献が重ならず、同じ物理モデルにも由来しない組 ← この中央値が基準
  （無作為な組も JSON には残す。無関係な組とほぼ同じ分布になる）

ベクトルの作り方は diagnose_topic_features.py と同一（式本文で TF-IDF と SVD を
学習し、ケースの説明文＋入出力変数名をその空間へ射影して L2 正規化）。

出力: experiments/similarity_scale.json
      docs/figures/fig_similarity_scale.png / .pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent

FS = 1.5  # 図中の文字サイズの倍率（報告書に貼ったとき小さすぎたため拡大）

GROUPS = [
    ("paraphrase", "言い換え（同じケースの説明文を書き換え）", "#2ca02c"),
    ("same_core", "同じ物理モデルで入出力だけ変えた組（説明文は同じ）", "#1f77b4"),
    ("same_source", "同じ文献に由来する別ケースの組", "#ff7f0e"),
    ("unrelated", "無関係な 2 ケースの組（出典文献が重ならない）", "#d62728"),
]


# ---------------------------------------------------------------------------
# 純関数（tests/test_similarity_scale.py で検証）
# ---------------------------------------------------------------------------

def paraphrase_base(case_id: str) -> str | None:
    """言い換えケースの ID から元ケースの ID を返す（例 core_058_v1_para1 → core_058_v1）."""
    if "_para" not in case_id:
        return None
    head, _, tail = case_id.rpartition("_para")
    return head if tail.isdigit() and head else None


def quantiles(vals) -> dict:
    a = np.asarray([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return {"n": 0, "median": float("nan"), "q25": float("nan"),
                "q75": float("nan"), "mean": float("nan")}
    q = np.percentile(a, [25, 50, 75])
    return {"n": int(a.size), "q25": float(q[0]), "median": float(q[1]),
            "q75": float(q[2]), "mean": float(a.mean())}


def cos_pairs(V: np.ndarray, pairs: list) -> np.ndarray:
    """L2 正規化済み行ベクトル集合から、指定した添字の組のコサイン類似度を返す."""
    if not pairs:
        return np.zeros(0)
    i = np.array([p[0] for p in pairs]); j = np.array([p[1] for p in pairs])
    return np.einsum("ij,ij->i", V[i], V[j])


def percentile_of(value: float, vals) -> float:
    """value が vals の中で下から何割の位置にあるか（0〜1）."""
    a = np.asarray([v for v in vals if v == v], dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.mean(a <= value))


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------

def run(seed: int = 42, n_random: int = 200_000,
        out_json: str = "experiments/similarity_scale.json",
        out_fig: str = "docs/figures/fig_similarity_scale.png"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    from two_stage_query_conditioned import (
        load_equations, load_cases, norm, eq_key, eq_text, case_text,
    )
    from set_aware_reranker import get_src
    from analyze_pfi import keep_setting_A

    eqs = load_equations()
    cases = load_cases()
    ek, et, es = [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); es.append(get_src(e))
    ki = {k: i for i, k in enumerate(ek)}

    # diagnose_topic_features.py と同一の空間（式本文で学習し、ケースを射影）
    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    svd = TruncatedSVD(n_components=256, random_state=seed)
    svd.fit(X_eq)
    Q = svd.transform(tfidf.transform([case_text(c, io=True) for c in cases]))
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

    idx_of = {c.get("case_id"): i for i, c in enumerate(cases)}
    core_of = [str(c.get("original_core_id") or "") for c in cases]
    is_A = [keep_setting_A(str(c.get("variant_type", ""))) for c in cases]
    srcs = []
    for c in cases:
        ids = [norm(m) for m in (c.get("correct_model_ids") or [])]
        srcs.append(frozenset(es[ki[m]] for m in ids if m in ki))

    rng = np.random.RandomState(0)

    # 言い換えの組
    para = []
    for i, c in enumerate(cases):
        b = paraphrase_base(str(c.get("case_id") or ""))
        if b and b in idx_of:
            para.append((i, idx_of[b]))

    # 同じ核モデル（言い換えを除く）の組
    para_ids = {c.get("case_id") for c in cases if paraphrase_base(str(c.get("case_id") or ""))}
    by_core: dict = {}
    for i, c in enumerate(cases):
        if c.get("case_id") in para_ids or not core_of[i]:
            continue
        by_core.setdefault(core_of[i], []).append(i)
    same_core = []
    for members in by_core.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                same_core.append((members[a], members[b]))

    # 評価対象（設定 A）の添字
    A_idx = [i for i in range(len(cases)) if is_A[i]]

    # 同じ文献に由来する別ケース（設定 A 内、核モデルが異なる組）
    same_source = []
    for _ in range(400_000):
        i, j = rng.choice(A_idx, 2, replace=False)
        if core_of[i] and core_of[i] == core_of[j]:
            continue
        if srcs[i] and srcs[j] and (srcs[i] & srcs[j]):
            same_source.append((int(i), int(j)))
        if len(same_source) >= 20_000:
            break

    # 無作為な組（設定 A 内）
    ii = rng.choice(A_idx, n_random); jj = rng.choice(A_idx, n_random)
    keep = ii != jj
    random_pairs = list(zip(ii[keep].tolist(), jj[keep].tolist()))

    # 無関係な組＝無作為な組から、出典文献が重なる組と同じ物理モデルに由来する組を除く
    unrelated_pairs = [
        (i, j) for i, j in random_pairs
        if not (core_of[i] and core_of[i] == core_of[j]) and not (srcs[i] & srcs[j])
    ]

    vals = {
        "paraphrase": cos_pairs(Q, para),
        "same_core": cos_pairs(Q, same_core),
        "same_source": cos_pairs(Q, same_source),
        "unrelated": cos_pairs(Q, unrelated_pairs),
        "random": cos_pairs(Q, random_pairs),
    }
    stats = {k: quantiles(v) for k, v in vals.items()}
    med_random = stats["unrelated"]["median"]
    out = {
        "config": {"svd_components": 256, "svd_seed": seed,
                   "space": "式本文で学習した TF-IDF→SVD 空間にケースを射影"},
        "groups": stats,
        "unrelated_median": med_random,
        "random_median": stats["random"]["median"],
        "share_gt_0.5": {k: float(np.mean(np.asarray(v) > 0.5)) if len(v) else 0.0
                          for k, v in vals.items()},
        "percentile_of_unrelated_median_within": {
            k: percentile_of(med_random, v) for k, v in vals.items()
        },
    }

    # 具体例：無関係な組から、類似度が各水準に近いものと最大の組を取り出す
    examples = []
    rv = vals["unrelated"]
    targets = [0.05, med_random, 0.40, 0.70, float(rv.max())]
    for target in targets:
        k = int(np.argmin(np.abs(rv - target)))
        i, j = unrelated_pairs[k]
        examples.append({
            "target": float(target), "cosine": float(rv[k]),
            "case_a": cases[i].get("case_id"), "context_a": (cases[i].get("context") or "")[:200],
            "case_b": cases[j].get("case_id"), "context_b": (cases[j].get("context") or "")[:200],
        })
    out["examples"] = examples

    Path(ROOT / out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")
    _figure(vals, med_random, ROOT / out_fig)
    print(f"Saved figure: {out_fig}")

    print("\n=== コサイン類似度の物差し（中央値・四分位） ===")
    for key, label, _ in GROUPS:
        s = stats[key]
        print(f"  {label:<34s} 中央値 {s['median']:.3f}  "
              f"（四分位 {s['q25']:.3f}–{s['q75']:.3f}、n={s['n']:,}）")
    print(f"\n無関係な組の中央値 {med_random:.3f} は、")
    for key, label, _ in GROUPS[:3]:
        p = out["percentile_of_unrelated_median_within"][key]
        print(f"  「{label}」の分布の下から {p:.1%} の位置")
    print("\n=== 無関係な組の具体例 ===")
    for ex in examples:
        print(f"\n  コサイン {ex['cosine']:.3f}")
        print(f"    A: {ex['context_a'][:110]}")
        print(f"    B: {ex['context_b'][:110]}")
    return out


def _figure(vals, med_random, out_png):
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

    fig, ax = plt.subplots(figsize=(11, 3.4))
    data = [vals[k] for k, _, _ in GROUPS][::-1]
    labels = [f"{lab}\n(n={len(vals[k]):,})" for k, lab, _ in GROUPS][::-1]
    cols = [c for _, _, c in GROUPS][::-1]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.6))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.set_yticklabels(labels, fontsize=9 * FS)
    ax.axvline(med_random, color="#d62728", ls="--", lw=1.4)
    ax.text(med_random + 0.01, 0.5, f"無関係な組の中央値 {med_random:.3f}",
            fontsize=9 * FS, color="#d62728")
    ax.set_ylim(0.3, 4.6)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("ケースどうしのベクトルのコサイン類似度", fontsize=10 * FS)
    ax.tick_params(axis="x", labelsize=10 * FS)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    run()
