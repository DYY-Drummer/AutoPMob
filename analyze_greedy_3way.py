"""3者比較の集計：静的参照 / 推論のみgreedy / 学習版greedy.

入力 : experiments/xg/{config}__{seed}.json（config = static|infer|train）
出力 :
  - experiments/greedy_3way_stats.json
  - docs/figures/fig_greedy_3way.png
検定 : seed対応 paired t / Wilcoxon（train vs static, train vs infer, infer vs static）。
層別 : 正解式数 X=1..10 で Recall@K を比較（飽和端 X8-10 が持ち上がるか）。
"""
from __future__ import annotations
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
XG = ROOT / "experiments" / "xg"
METRIC = "Recall@K_correct"
CONFIGS = ["static", "infer", "train"]
LABEL = {"static": "静的参照 (既定)", "infer": "推論のみgreedy", "train": "学習版greedy"}
COLOR = {"static": "#9e9e9e", "infer": "#1f77b4", "train": "#d62728"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="experiments/xg", help="入力ディレクトリ（{config}__{seed}.json）")
    p.add_argument("--out-json", default="experiments/greedy_3way_stats.json", help="集計結果の出力先JSON")
    p.add_argument("--out-fig", default="docs/figures/fig_greedy_3way.png", help="図の出力先PNG")
    p.add_argument("--tag", default="DAEのみ", help="図中のデータセット表記（既定はDAEのみ実行時の表記と同一）")
    return p.parse_args()


def resolve(path_str: str) -> Path:
    """相対パスはROOT（本スクリプトの場所）基準で解決する。"""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def load(xg_dir: Path = XG):
    """config -> [per_case records] （seedを含む）."""
    out = defaultdict(list)
    for f in sorted(glob.glob(str(xg_dir / "*.json"))):
        m = re.match(r"(static|infer|train)__(\d+)\.json", Path(f).name)
        if not m:
            continue
        cfg, seed = m.group(1), int(m.group(2))
        d = json.load(open(f, encoding="utf-8"))
        for mode, r in d.get("results", {}).items():
            for rec in r.get("per_case", []):
                rec = dict(rec); rec["config"] = cfg
                out[cfg].append(rec)
    return out


def seed_means(records, sub=None):
    by = defaultdict(list)
    for r in records:
        if METRIC not in r:
            continue
        if sub is not None and not sub(r):
            continue
        by[r["seed"]].append(r[METRIC])
    return {s: float(np.mean(v)) for s, v in by.items() if v}


def paired(a_rec, b_rec, sub=None):
    a, b = seed_means(a_rec, sub), seed_means(b_rec, sub)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return {"n_seeds": len(seeds)}
    x = np.array([a[s] for s in seeds]); y = np.array([b[s] for s in seeds])
    d = x - y
    res = {"n_seeds": len(seeds), "mean_a": round(float(x.mean()), 4),
           "mean_b": round(float(y.mean()), 4), "mean_delta": round(float(d.mean()), 4),
           "std_delta": round(float(d.std(ddof=1)), 4)}
    if d.std() > 0:
        t, p = stats.ttest_rel(x, y)
        res["t"] = round(float(t), 3); res["p_ttest"] = round(float(p), 5)
        res["cohen_dz"] = round(float(d.mean() / d.std(ddof=1)), 3)
        try:
            res["p_wilcoxon"] = round(float(stats.wilcoxon(x, y).pvalue), 5)
        except Exception:
            res["p_wilcoxon"] = None
    return res


def main():
    args = parse_args()
    xg_dir = resolve(args.dir)
    out_json_path = resolve(args.out_json)
    out_fig_path = resolve(args.out_fig)
    tag = args.tag
    # xlabelの短縮表記: 既定タグ "DAEのみ" は元コードでは "(DAE)" と略記されていたため、
    # 末尾の「のみ」を除いた短縮形を使う（既定タグ以外はそのまま使う）。
    short_tag = tag[:-2] if tag.endswith("のみ") else tag

    data = load(xg_dir)
    for c in CONFIGS:
        print(f"{c:8s}: {len(data.get(c, []))} per-case recs, seeds={sorted({r['seed'] for r in data.get(c,[])})}")
    out = {"metric": METRIC, "n_per_config": {c: len(data.get(c, [])) for c in CONFIGS}}

    # 全体の平均（seed平均の平均±sd）
    out["overall"] = {}
    for c in CONFIGS:
        sm = seed_means(data.get(c, []))
        vals = list(sm.values())
        out["overall"][c] = {"mean": round(float(np.mean(vals)), 4),
                             "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                             "n_seeds": len(vals)}
    # 対応のある検定
    out["tests"] = {
        "train_vs_static": paired(data.get("train", []), data.get("static", [])),
        "train_vs_infer":  paired(data.get("train", []), data.get("infer", [])),
        "infer_vs_static": paired(data.get("infer", []), data.get("static", [])),
    }
    # 最難端 (X>=8) だけの train vs static
    out["tests_hard_X8plus"] = {
        "train_vs_static": paired(data.get("train", []), data.get("static", []),
                                  sub=lambda r: r.get("n_correct", 1) >= 8),
        "train_vs_infer": paired(data.get("train", []), data.get("infer", []),
                                 sub=lambda r: r.get("n_correct", 1) >= 8),
    }

    # 正解式数 X=1..10 別（seed平均→層平均）
    out["by_n_correct"] = {}
    for c in CONFIGS:
        rows = []
        for x in range(1, 11):
            sm = seed_means(data.get(c, []), sub=lambda r, x=x: r.get("n_correct") == x)
            vals = list(sm.values())
            n_cases = sum(1 for r in data.get(c, []) if r.get("n_correct") == x)
            rows.append({"X": x, "n": n_cases,
                         "mean": round(float(np.mean(vals)), 4) if vals else None,
                         "sem": round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 4) if len(vals) > 1 else None})
        out["by_n_correct"][c] = rows

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- 図 ----
    for fam in ["Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "AppleGothic"]:
        try:
            plt.rcParams["font.family"] = fam; break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # (a) 全体バー
    ax = axes[0]
    xs = np.arange(len(CONFIGS))
    means = [out["overall"][c]["mean"] for c in CONFIGS]
    errs = [out["overall"][c]["std"] for c in CONFIGS]
    bars = ax.bar(xs, means, yerr=errs, capsize=5, color=[COLOR[c] for c in CONFIGS])
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[c] for c in CONFIGS])
    ax.set_ylabel(f"{METRIC}（{tag}・seed平均）")
    ax.set_title("(a) 全体：DAE（最難）での Recall@K")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.005, f"{m:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(means) * 1.18)

    # (b) X別
    ax = axes[1]
    for c in CONFIGS:
        rows = [r for r in out["by_n_correct"][c] if r["mean"] is not None]
        xs = [r["X"] for r in rows]; ys = [r["mean"] for r in rows]
        es = [r["sem"] or 0 for r in rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=2, color=COLOR[c], label=LABEL[c])
    ax.set_xlabel(f"正解式数 X ({short_tag})"); ax.set_ylabel(METRIC)
    ax.set_title("(b) 正解式数別：飽和端で差が開くか")
    ax.grid(alpha=0.25); ax.legend(fontsize=9); ax.set_xticks(range(1, 11))
    fig.suptitle(f"学習版greedy 3者比較（{tag}・層化分割）", fontsize=13, y=1.02)
    fig.tight_layout()
    out_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig_path, dpi=150, bbox_inches="tight")

    # ---- コンソール ----
    print(f"\n=== 全体（{tag}, seed平均±sd）===")
    for c in CONFIGS:
        o = out["overall"][c]
        print(f"  {LABEL[c]:16s} {o['mean']:.4f} ± {o['std']:.4f}  (n_seeds={o['n_seeds']})")
    print("\n=== 対応のある検定 ===")
    for name, r in out["tests"].items():
        if "mean_delta" in r:
            print(f"  {name:18s} Δ={r['mean_delta']:+.4f}±{r.get('std_delta',0):.4f} "
                  f"t={r.get('t','-')} p={r.get('p_ttest','-')} dz={r.get('cohen_dz','-')} pW={r.get('p_wilcoxon','-')}")
    print("\n=== 最難端 X>=8 ===")
    for name, r in out["tests_hard_X8plus"].items():
        if "mean_delta" in r:
            print(f"  {name:18s} Δ={r['mean_delta']:+.4f} t={r.get('t','-')} p={r.get('p_ttest','-')} dz={r.get('cohen_dz','-')}")
    print(f"\nSaved: {args.out_json}, {args.out_fig}")


if __name__ == "__main__":
    main()
