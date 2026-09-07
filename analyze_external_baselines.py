#!/usr/bin/env python3
"""外部ベースライン（BM25・E5）と古典 IR ベースライン・提案手法の対応あり検定（A2）.

入力: experiments/strat_{A,B}.json（baseline / reranker-10S の per_seed）
      experiments/external_baselines_{A,B}.json（external_baselines.py の出力）
      experiments/external_baselines_coverage.json（任意; 被覆率）
出力: experiments/external_baselines_stats.json
      標準出力に修論 表 3 用の LaTeX 行（転記ミス防止）

検定は analyze_significance.paired_stats（Wilcoxon 符号順位・対応あり t・Cohen's d_z）を
そのまま使う。vs_baseline は「外部 − 古典 IR」、vs_proposed は「提案 − 外部」の差。
tfidf__none__w70-30 が strat の baseline と完全一致することを確認する（回帰）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_significance import paired_stats  # noqa: E402

EXP = ROOT / "experiments"
METRICS = ["Recall@K_correct", "MAP", "Recall@20"]
SETTINGS = ["A", "B"]
REGRESSION_LABEL = "tfidf__none__w70-30"


def per_seed(doc: dict, mode: str, metric: str) -> list[float]:
    return [p[metric] for p in doc["results"][mode]["per_seed"]]


def analyze(strat: dict, ext: dict) -> dict:
    seeds_ref = [p["seed"] for p in strat["results"]["baseline"]["per_seed"]]
    out = {"seeds": seeds_ref, "reference": {}, "labels": {}}
    for mode in ("baseline", "reranker-10S"):
        out["reference"][mode] = {m: strat["results"][mode][m] for m in METRICS}
    for lab, res in ext["results"].items():
        seeds = [p["seed"] for p in res["per_seed"]]
        if seeds != seeds_ref:
            raise ValueError(f"{lab}: seeds {seeds} != strat seeds {seeds_ref}")
        entry = {"scorer": res["scorer"], "norm": res["norm"],
                 "w_text": res["w_text"], "w_var": res["w_var"]}
        for m in METRICS:
            x = per_seed(ext, lab, m)
            entry[m] = {
                "mean": res[m]["mean"], "std": res[m]["std"],
                "vs_baseline": paired_stats(per_seed(strat, "baseline", m), x),
                "vs_proposed": paired_stats(x, per_seed(strat, "reranker-10S", m)),
            }
        out["labels"][lab] = entry
    reg = out["labels"].get(REGRESSION_LABEL)
    if reg is not None:
        for m in METRICS:
            d = reg[m]["vs_baseline"]["mean_delta"]
            if abs(d) > 1e-9:
                raise AssertionError(f"regression failed: {REGRESSION_LABEL} {m} delta={d}")
        out["regression_ok"] = True
    return out


def best_per_scorer(stats: dict) -> dict:
    """(scorer, norm) ごとに Recall@K_correct の平均が最大の重みを選ぶ（テスト集合で選ぶので
    外部ベースラインに有利＝提案手法には保守的な比較）。w_text = 0 は変数のみの順位付けで
    スコアラーによらず同一なので除く."""
    best = {}
    for lab, e in stats["labels"].items():
        if e["w_text"] <= 0:
            continue
        key = f"{e['scorer']}__{e['norm']}"
        cur = best.get(key)
        if cur is None or e["Recall@K_correct"]["mean"] > cur["Recall@K_correct"]["mean"]:
            best[key] = {"label": lab, **e}
    return best


def summary_table(stats: dict, setting: str) -> list[str]:
    """scorer ごとに 文章のみ / (0.7,0.3) / (0.3,0.7) / 最良重み を 1 行に並べる."""
    lines = [f"{'scorer':16s} | {'text only':>10s} | {'w70-30':>10s} | {'w30-70':>10s} | best (w_text) "]
    seen = []
    for lab, e in stats["labels"].items():
        key = f"{e['scorer']}__{e['norm']}"
        if key not in seen:
            seen.append(key)
    best = best_per_scorer(stats)
    for key in seen:
        def val(wt, wv):
            lab = f"{key}__w{round(wt*100):02d}-{round(wv*100):02d}"
            e = stats["labels"].get(lab)
            return f"{e['Recall@K_correct']['mean']:.4f}" if e else "   -  "
        b = best[key]
        lines.append(f"{key:16s} | {val(1, 0):>10s} | {val(0.7, 0.3):>10s} | {val(0.3, 0.7):>10s} | "
                     f"{b['Recall@K_correct']['mean']:.4f} ({b['w_text']:.2f})  "
                     f"vs prop p_w={b['Recall@K_correct']['vs_proposed']['p_wilcoxon']:.5f}")
    return lines


def latex_rows(stats: dict, setting: str) -> list[str]:
    rows = []
    for lab, e in stats["labels"].items():
        rk, mp, r20 = e["Recall@K_correct"], e["MAP"], e["Recall@20"]
        rows.append(
            f"{setting} & {lab:26s} & ${rk['mean']:.3f} \\pm {rk['std']:.3f}$ & "
            f"{mp['mean']:.3f} & {r20['mean']:.3f} \\\\  "
            f"% vs base: d={rk['vs_baseline']['mean_delta']:+.4f} p_w={rk['vs_baseline']['p_wilcoxon']:.5f}; "
            f"vs prop: d={rk['vs_proposed']['mean_delta']:+.4f} p_w={rk['vs_proposed']['p_wilcoxon']:.5f} "
            f"d_z={rk['vs_proposed']['cohen_dz']:.2f}"
        )
    return rows


def main() -> None:
    out = {}
    for s in SETTINGS:
        strat_path, ext_path = EXP / f"strat_{s}.json", EXP / f"external_baselines_{s}.json"
        if not ext_path.exists():
            print(f"skip setting {s}: {ext_path} not found")
            continue
        out[s] = analyze(json.load(open(strat_path)), json.load(open(ext_path)))
        out[s]["files"] = {"strat": strat_path.name, "external": ext_path.name}
        out[s]["best_per_scorer"] = {k: v["label"] for k, v in best_per_scorer(out[s]).items()}
    cov_path = EXP / "external_baselines_coverage.json"
    if cov_path.exists():
        cov = json.load(open(cov_path))["coverage"]
        for s in out:
            if s in cov:
                out[s]["coverage@50"] = {lab: v["50"]["mean"] for lab, v in cov[s].items()}
                out[s]["coverage@200"] = {lab: v["200"]["mean"] for lab, v in cov[s].items()}
    dst = EXP / "external_baselines_stats.json"
    # 回帰ラベル（差が全 0）の検定は NaN になるので JSON では null にする
    out = json.loads(json.dumps(out).replace("NaN", "null"))
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {dst}\n")
    for s in out:
        ref = out[s]["reference"]
        print(f"=== Setting {s} (regression_ok={out[s].get('regression_ok')}) ===")
        for mode in ("baseline", "reranker-10S"):
            r = ref[mode]
            print(f"{s} & {mode:26s} & ${r['Recall@K_correct']['mean']:.3f} \\pm "
                  f"{r['Recall@K_correct']['std']:.3f}$ & {r['MAP']['mean']:.3f} & "
                  f"{r['Recall@20']['mean']:.3f} \\\\  % strat_{s}.json")
        for row in latex_rows(out[s], s):
            print(row)
        print()
        for line in summary_table(out[s], s):
            print(line)
        if "coverage@50" in out[s]:
            print("coverage@50: " + ", ".join(f"{k}={v:.4f}" for k, v in out[s]["coverage@50"].items()))
            print("coverage@200: " + ", ".join(f"{k}={v:.4f}" for k, v in out[s]["coverage@200"].items()))
        print()


if __name__ == "__main__":
    main()
