"""PFIケース別依存プロファイル：置換で正解→不正解に転じるケースを属性で特徴づける.

入力 : experiments/pfi_per_case.json（analyze_pfi.py が出力）
出力 : experiments/pfi_profile_stats.json, docs/figures/fig_pfi_dependence_profile.png/.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent
SETAWARE = [("gComp", "＋補完性", "#1f77b4"),
            ("gCoh", "＋一貫性", "#2ca02c"),
            ("gDom", "＋ドメイン一致", "#d62728")]
ATTRS = ["n_correct", "n_input", "n_output", "n_sources", "sep"]
ATTR_LABELS = {"n_correct": "正解式数", "n_input": "入力変数数", "n_output": "出力変数数",
               "n_sources": "出典数", "sep": "分離度 sep"}
FEATURE_LABELS = {name: label.lstrip("＋") for name, label, _ in SETAWARE}


def load_per_case(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["records"], d.get("config", {}), d.get("n_seeds")


def collapse_to_cases(records, feature, solved_only=True):
    """seed をまたぐ重複を潰し、1 case_id = 1 観測にする.

    同一 case_id は複数 seed のテストに出現し、属性は seed 間で同一。
    記録をそのまま観測とすると疑似反復になり p 値が過大評価される。
    solved_only=True のとき baseline 解済み（base_R==1.0）の記録のみを対象に集約する。

    返す各 dict: case_id, sep, drop, flip_frac, flip, および属性（seed 間で不変）。
    flip: そのケースが（テストされた seed のうち）過半で反転したなら 1。
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in records:
        if r["feature"] != feature:
            continue
        if solved_only and r["base_R"] != 1.0:
            continue
        buckets[r["case_id"]].append(r)
    out = []
    for cid, rs in buckets.items():
        seps = [r["sep"] for r in rs if r.get("sep") is not None]
        flip_frac = float(np.mean([r["flip"] for r in rs]))
        row = {
            "case_id": cid,
            "n_obs": len(rs),
            "sep": (float(np.mean(seps)) if seps else None),
            "drop": float(np.mean([r["drop"] for r in rs])),
            "flip_frac": flip_frac,
            "flip": 1 if flip_frac >= 0.5 else 0,
        }
        for a in ("n_correct", "n_input", "n_output", "n_sources", "variant"):
            row[a] = rs[0][a]
        out.append(row)
    return out


def split_dependent_robust(records, feature):
    """依存（置換で反転）／頑健（反転しない）ケースに二分する.

    collapse_to_cases で seed 重複を先に集約するため、同一 case_id が
    依存・頑健の両方に出現することはない（多数決で一方に確定する）。
    """
    cases = collapse_to_cases(records, feature, solved_only=True)
    dep = [c for c in cases if c["flip"] == 1]
    rob = [c for c in cases if c["flip"] == 0]
    return dep, rob


def attr_array(recs, attr):
    """recs から attr 列を取り出す（欠損値・非有限値は除外）.

    非有限値（nan/inf）が1件でも混入すると平均・標準偏差が nan に伝播し、
    描画側でその特徴の曲線全体が消える恐れがあるため、ここで弾く。
    """
    arr = np.array([r[attr] for r in recs if r.get(attr) is not None], dtype=float)
    return arr[np.isfinite(arr)]


def mannwhitney_effect(dep, rob):
    def _sem(a):
        if a.size == 0:
            return float("nan")     # 群が空 → 未定義
        if a.size == 1:
            return 0.0              # 単一点 → 慣例的に0（未定義だが nan にはしない）
        return float(a.std(ddof=1) / np.sqrt(a.size))

    res = {"n_dep": int(dep.size), "n_rob": int(rob.size),
           "mean_dep": float(dep.mean()) if dep.size else float("nan"),
           "sem_dep": _sem(dep),
           "mean_rob": float(rob.mean()) if rob.size else float("nan"),
           "sem_rob": _sem(rob),
           "mannwhitney_p": float("nan"), "rank_biserial_r": float("nan")}
    if dep.size and rob.size:
        U, p = stats.mannwhitneyu(dep, rob, alternative="two-sided")
        res["mannwhitney_p"] = float(p)
        res["rank_biserial_r"] = float(2.0 * U / (dep.size * rob.size) - 1.0)
    return res


def standardized_mean_diff(dep, rob):
    n1, n2 = dep.size, rob.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = np.sqrt(((n1 - 1) * dep.var(ddof=1) + (n2 - 1) * rob.var(ddof=1)) / (n1 + n2 - 2))
    # sp==0（両群とも定数）は「差が無い」のではなく真の SMD が無限大なので nan にする。
    return float((dep.mean() - rob.mean()) / sp) if sp > 0 else float("nan")


def smd_stderr(dep, rob):
    """標準化平均差（Cohen's d）の閉形式標準誤差.

    SE(d) = sqrt((n1+n2)/(n1*n2) + d^2 / (2*(n1+n2)))
    いずれかの群が 2 未満なら nan。
    """
    n1, n2 = dep.size, rob.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    d = standardized_mean_diff(dep, rob)
    if d != d:  # nan
        return float("nan")
    return float(np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2.0 * (n1 + n2))))


def spearman_drop_sep(records, feature):
    """drop と sep の Spearman 順位相関.

    collapse_to_cases で baseline 解済み（base_R == 1.0）の記録に限定して
    ケース単位に集約してから相関を取る。この部分集合では perm_R <= 1.0 のため
    drop >= 0 が構造的に保証され、「置換が助ける」側（drop < 0）が定義上
    現れない。母集団を混ぜると符号が反転するため、図（sep_profile）・係数
    ともこの母集団に統一する（FIX C2）。
    """
    cases = collapse_to_cases(records, feature, solved_only=True)
    recs = [c for c in cases if c["sep"] is not None]
    if len(recs) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(recs)}
    rho, p = stats.spearmanr([r["sep"] for r in recs], [r["drop"] for r in recs])
    return {"rho": float(rho), "p": float(p), "n": len(recs)}


def sep_profile(records, feature, n_bins=5):
    """sep の分位ビンごとの平均 drop ±SEM.

    spearman_drop_sep と同一の母集団（collapse_to_cases による baseline
    解済みケース、seed 重複を集約済み）を使う。母集団が異なると図と係数の
    符号が食い違うため（FIX C2）。
    各ビンの x 座標はビン区間の中点ではなく、そのビンに実際に入った sep
    値の中央値を使う。区間中点は端ビンで外れ値に強く引きずられ、見かけの
    傾きを作ってしまうため（FIX I2）。
    """
    cases = collapse_to_cases(records, feature, solved_only=True)
    recs = [c for c in cases if c["sep"] is not None
            and np.isfinite(c["sep"]) and np.isfinite(c["drop"])]
    if len(recs) < n_bins:
        return [], [], []
    seps = np.array([r["sep"] for r in recs])
    drops = np.array([r["drop"] for r in recs])
    edges = np.quantile(seps, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    xs, ys, es = [], [], []
    for b in range(n_bins):
        m = (seps >= edges[b]) & (seps < edges[b + 1])
        if m.sum() == 0:
            continue
        xs.append(float(np.median(seps[m])))
        ys.append(float(drops[m].mean()))
        es.append(float(drops[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else 0.0)
    return xs, ys, es


def cancellation_check(records, feature, solved_only=True):
    """集約 PFI ≈ 0 が「寄与なし」か「相殺/希釈」かを判別する.

    sep>0（正解式で値が高い）と sep<0（逆分離）に分けて平均 drop を出す。
    寄与なしなら両方 0 付近。相殺なら符号が逆に割れる。

    solved_only=True（既定）では collapse_to_cases が base_R==1.0 の記録に
    限定するため、perm_R<=1.0 から drop>=0 が構造的に保証される（FIX I1）。
    この母集団では符号反転（真の相殺）は原理的に観測できず、測れるのは
    大きさの非対称（希釈）のみ。真の符号反転を検出するには
    solved_only=False を指定し、base_R<1.0 の記録（drop が負になりうる）
    も含める必要がある。
    """
    cases = [c for c in collapse_to_cases(records, feature, solved_only=solved_only)
             if c["sep"] is not None]
    pos = np.array([c["drop"] for c in cases if c["sep"] > 0])
    neg = np.array([c["drop"] for c in cases if c["sep"] < 0])
    res = {"n_sep_pos": int(pos.size), "n_sep_neg": int(neg.size),
           "mean_drop_sep_pos": float(pos.mean()) if pos.size else float("nan"),
           "mean_drop_sep_neg": float(neg.mean()) if neg.size else float("nan"),
           "sem_drop_sep_pos": float(pos.std(ddof=1)/np.sqrt(pos.size)) if pos.size > 1 else 0.0,
           "sem_drop_sep_neg": float(neg.std(ddof=1)/np.sqrt(neg.size)) if neg.size > 1 else 0.0,
           "mannwhitney_p": float("nan")}
    if pos.size and neg.size:
        _, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        res["mannwhitney_p"] = float(p)
    return res


def build_stats(records, n_seeds=None):
    feats = {}
    all_feats = sorted(set(r["feature"] for r in records))
    for name in all_feats:
        dep, rob = split_dependent_robust(records, name)
        solved = len(dep) + len(rob)
        entry = {"n_dependent": len(dep), "n_robust": len(rob),
                 "n_cases": solved,
                 "flip_fraction_of_solved": (len(dep) / solved) if solved else 0.0,
                 "spearman_drop_sep": spearman_drop_sep(records, name),
                 "cancellation": cancellation_check(records, name, solved_only=True),
                 "cancellation_all_cases": cancellation_check(records, name, solved_only=False),
                 "attrs": {}}
        for attr in ATTRS:
            entry["attrs"][attr] = mannwhitney_effect(attr_array(dep, attr), attr_array(rob, attr))
        feats[name] = entry
    return {"metric": "Recall@K_correct", "n_features": len(all_feats),
            "n_seeds": n_seeds, "features": feats}


def _make_figure(records, out_png):
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # (a) 分離度ビン × 平均drop
    for name, label, col in SETAWARE:
        xs, ys, es = sep_profile(records, name, n_bins=5)
        if xs:
            axes[0].errorbar(xs, ys, yerr=es, marker="o", lw=2.2, capsize=3, color=col, label=label)
    axes[0].axhline(0, color="#bbbbbb", lw=1, zorder=0)
    axes[0].set_title("(a) 分離度別の期待Recall低下")
    axes[0].set_xlabel("分離度 sep（正解式−不正解式の特徴値差）")
    axes[0].set_ylabel("置換によるRecall@K_correctの期待低下")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=9, loc="upper left")

    # (b) gComp/gCoh の 依存−頑健 標準化平均差（属性別・横棒）
    bar_feats = [("gComp", "#1f77b4"), ("gCoh", "#2ca02c")]
    y = np.arange(len(ATTRS)); h = 0.38
    for bi, (name, col) in enumerate(bar_feats):
        dep, rob = split_dependent_robust(records, name)
        smds = [standardized_mean_diff(attr_array(dep, a), attr_array(rob, a)) for a in ATTRS]
        ses = [smd_stderr(attr_array(dep, a), attr_array(rob, a)) for a in ATTRS]
        smds = [0.0 if (s != s) else s for s in smds]  # nan→0（描画のため）
        ses = [0.0 if (e != e) else e for e in ses]    # nan→0（描画のため）
        axes[1].barh(y + (bi - 0.5) * h, smds, height=h, xerr=ses,
                     error_kw=dict(capsize=3, lw=1), color=col, label=FEATURE_LABELS.get(name, name))
    axes[1].axvline(0, color="#888888", lw=1)
    axes[1].set_yticks(y); axes[1].set_yticklabels([ATTR_LABELS.get(a, a) for a in ATTRS])
    axes[1].set_title("(b) 依存ケースを特徴づける属性")
    axes[1].set_xlabel("標準化平均差（依存−頑健）")
    axes[1].grid(alpha=0.25, axis="x"); axes[1].legend(fontsize=9)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/pfi_per_case.json")
    ap.add_argument("--out-json", default="experiments/pfi_profile_stats.json")
    ap.add_argument("--out-fig", default="docs/figures/fig_pfi_dependence_profile.png")
    a = ap.parse_args()
    records, config, n_seeds = load_per_case(ROOT / a.input)
    stats_out = build_stats(records, n_seeds=n_seeds)
    stats_out["config"] = config
    json.dump(stats_out, open(ROOT / a.out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Saved stats: {a.out_json}")
    _make_figure(records, ROOT / a.out_fig)
    print(f"Saved figure: {a.out_fig}")
    # コンソール要約
    for name, label, _ in SETAWARE:
        g = stats_out["features"].get(name, {})
        s = g.get("spearman_drop_sep", {})
        print(f"{name:6s} 依存={g.get('n_dependent')}/{g.get('n_dependent',0)+g.get('n_robust',0)} "
              f"Spearman(drop,sep) rho={s.get('rho')} p={s.get('p')}")


if __name__ == "__main__":
    main()
