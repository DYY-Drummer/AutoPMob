"""DoF=0 停止規則の分析：K を与えずに集合を予測できるか.

入力 : experiments/xd_dof/{setting}__{seed}.json（setting = dae|A）
出力 : experiments/dof_stop_stats.json, docs/figures/fig_dof_stop.png

主張の検証:
  - set_f1_dof（DoF停止で予測した集合のF1）が set_f1_oracleK（正解K既知の上限）に迫るか
  - closed_oracleK：スコア上位K件が閉じている割合（DoF停止集合は構成上ほぼ1）
  - K予測：k_pred_dof - n_correct の分布・的中率（※本データは K≈|出力| で自明な点は正直に併記）
"""
from __future__ import annotations
import glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
XD = ROOT / "experiments" / "xd_dof"


def load():
    out = defaultdict(list)
    for f in sorted(glob.glob(str(XD / "*.json"))):
        m = re.match(r"(dae|A)__(\d+)\.json", Path(f).name)
        if not m:
            continue
        setting, seed = m.group(1), int(m.group(2))
        d = json.load(open(f, encoding="utf-8"))
        for mode, r in d.get("results", {}).items():
            for rec in r.get("per_case", []):
                out[setting].append(rec)
    return out


def seed_mean(records, key, sub=None):
    by = defaultdict(list)
    for r in records:
        if key not in r:
            continue
        if sub is not None and not sub(r):
            continue
        by[r["seed"]].append(r[key])
    return {s: float(np.mean(v)) for s, v in by.items() if v}


def summ(records, key, sub=None):
    sm = list(seed_mean(records, key, sub).values())
    if not sm:
        return None
    return {"mean": round(float(np.mean(sm)), 4),
            "std": round(float(np.std(sm, ddof=1)) if len(sm) > 1 else 0.0, 4)}


def paired(records, k1, k2, sub=None):
    a = seed_mean(records, k1, sub); b = seed_mean(records, k2, sub)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return {}
    x = np.array([a[s] for s in seeds]); y = np.array([b[s] for s in seeds])
    d = x - y
    res = {"mean_delta": round(float(d.mean()), 4), "std_delta": round(float(d.std(ddof=1)), 4)}
    if d.std() > 0:
        t, p = stats.ttest_rel(x, y)
        res.update(t=round(float(t), 3), p_ttest=round(float(p), 5),
                   cohen_dz=round(float(d.mean() / d.std(ddof=1)), 3))
    return res


def main():
    data = load()
    out = {}
    for setting, recs in data.items():
        seeds = sorted({r["seed"] for r in recs})
        # K予測の的中/誤差
        kerr = [r["k_pred_dof"] - r["n_correct"] for r in recs if "k_pred_dof" in r]
        o = {
            "n_per_case": len(recs), "seeds": seeds,
            "set_f1_oracleK": summ(recs, "set_f1_oracleK"),
            "set_f1_dof": summ(recs, "set_f1_dof"),
            "set_exact_oracleK": summ(recs, "set_exact_oracleK"),
            "set_exact_dof": summ(recs, "set_exact_dof"),
            "closed_oracleK": summ(recs, "closed_oracleK"),
            "closed_dof": summ(recs, "closed_dof"),
            "Recall@K_correct": summ(recs, "Recall@K_correct"),
            "K_exact_rate": round(float(np.mean([e == 0 for e in kerr])), 4) if kerr else None,
            "K_within1_rate": round(float(np.mean([abs(e) <= 1 for e in kerr])), 4) if kerr else None,
            "K_MAE": round(float(np.mean(np.abs(kerr))), 4) if kerr else None,
            "K_median_abs_err": round(float(np.median(np.abs(kerr))), 4) if kerr else None,
            "K_bias": round(float(np.mean(kerr)), 4) if kerr else None,
            "dof_vs_oracle_f1": paired(recs, "set_f1_dof", "set_f1_oracleK"),
        }
        out[setting] = o

    # DAE の X 別
    if "dae" in data:
        recs = data["dae"]
        out["dae_by_X"] = []
        for x in range(1, 11):
            sub = lambda r, x=x: r.get("n_correct") == x
            row = {"X": x, "n": sum(1 for r in recs if r.get("n_correct") == x)}
            for key in ["set_f1_oracleK", "set_f1_dof", "set_exact_dof", "closed_oracleK"]:
                s = summ(recs, key, sub)
                row[key] = s["mean"] if s else None
            kerr = [r["k_pred_dof"] - r["n_correct"] for r in recs if r.get("n_correct") == x and "k_pred_dof" in r]
            row["K_exact_rate"] = round(float(np.mean([e == 0 for e in kerr])), 3) if kerr else None
            row["k_pred_mean"] = round(float(np.mean([r["k_pred_dof"] for r in recs if r.get("n_correct") == x])), 2)
            out["dae_by_X"].append(row)

    json.dump(out, open(ROOT / "experiments" / "dof_stop_stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- 図 ----
    for fam in ["Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "AppleGothic"]:
        try:
            plt.rcParams["font.family"] = fam; break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # (a) set-F1: oracle-K vs DoF停止（設定ごと）＋ 閉包率注記
    ax = axes[0]
    settings = [s for s in ["dae", "A"] if s in out]
    labs = {"dae": "DAEのみ", "A": "設定A"}
    w = 0.35; xs = np.arange(len(settings))
    orc = [out[s]["set_f1_oracleK"]["mean"] for s in settings]
    dof = [out[s]["set_f1_dof"]["mean"] for s in settings]
    orc_e = [out[s]["set_f1_oracleK"]["std"] for s in settings]
    dof_e = [out[s]["set_f1_dof"]["std"] for s in settings]
    ax.bar(xs - w/2, orc, w, yerr=orc_e, capsize=4, color="#9e9e9e", label="正解式数Kを教えた場合（上位K件）")
    ax.bar(xs + w/2, dof, w, yerr=dof_e, capsize=4, color="#d62728", label="自由度ゼロ停止（Kを教えない）")
    for i, s in enumerate(settings):
        ax.text(xs[i]-w/2, orc[i]+0.01, f"{orc[i]:.3f}", ha="center", fontsize=9)
        ax.text(xs[i]+w/2, dof[i]+0.01, f"{dof[i]:.3f}", ha="center", fontsize=9)
        cl = out[s]["closed_oracleK"]["mean"]
        ax.text(xs[i], 0.03, f"上位K件が解ける系\nである割合={cl:.2f}", ha="center", fontsize=8, color="#555")
    ax.set_xticks(xs); ax.set_xticklabels([labs[s] for s in settings])
    ax.set_ylabel("集合F1"); ax.set_ylim(0, 1.0)
    ax.set_title("(a) 集合F1")
    ax.legend(fontsize=9, loc="upper right")

    # (b) DAE X別：set_f1 と K予測
    if "dae_by_X" in out:
        ax = axes[1]
        rows = [r for r in out["dae_by_X"] if r["set_f1_dof"] is not None]
        xs = [r["X"] for r in rows]
        ax.plot(xs, [r["set_f1_oracleK"] for r in rows], "o-", color="#9e9e9e", lw=2, label="集合F1（Kを教えた場合）")
        ax.plot(xs, [r["set_f1_dof"] for r in rows], "o-", color="#d62728", lw=2, label="集合F1（自由度ゼロ停止）")
        ax.plot(xs, [r["set_exact_dof"] for r in rows], "s--", color="#1f77b4", lw=1.6, label="完全一致率（自由度ゼロ停止）")
        ax.set_xlabel("正解式数"); ax.set_ylabel("集合F1 / 完全一致率")
        ax.set_title("(b) DAEの正解式数別"); ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.suptitle("自由度ゼロ停止と、正解式数Kを教えた場合の比較", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "fig_dof_stop.png", dpi=150, bbox_inches="tight")

    # ---- コンソール ----
    for setting in [s for s in ["dae", "A"] if s in out]:
        o = out[setting]
        print(f"\n=== 設定 {setting}（{o['n_per_case']} per-case, seeds={len(o['seeds'])}）===")
        print(f"  集合F1 : oracle-K {o['set_f1_oracleK']['mean']:.4f}  vs  DoF停止 {o['set_f1_dof']['mean']:.4f}  "
              f"(Δ={o['dof_vs_oracle_f1'].get('mean_delta','-')}, p={o['dof_vs_oracle_f1'].get('p_ttest','-')})")
        print(f"  完全一致: oracle-K {o['set_exact_oracleK']['mean']:.4f}  vs  DoF停止 {o['set_exact_dof']['mean']:.4f}")
        print(f"  閉包率  : oracle-K集合 {o['closed_oracleK']['mean']:.4f}  vs  DoF停止集合 {o['closed_dof']['mean']:.4f}")
        print(f"  K予測   : 的中率 {o['K_exact_rate']}  ±1以内 {o['K_within1_rate']}  中央絶対誤差 {o['K_median_abs_err']}  MAE {o['K_MAE']}  bias {o['K_bias']}  (※K≈|出力|で自明な点に留意)")
        print(f"  Recall@K_correct = {o['Recall@K_correct']['mean']:.4f}")
    print("\nSaved: experiments/dof_stop_stats.json, docs/figures/fig_dof_stop.png")


if __name__ == "__main__":
    main()
