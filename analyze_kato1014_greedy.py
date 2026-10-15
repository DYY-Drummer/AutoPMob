"""§6（2026-10-14 の指示 5）：一括の並べ直し・逐次選択（推論のみ）・逐次選択（訓練も逐次）の比較を新しい設定で.

入力: --rundir の {static,infer,train}_{A,B}__{seed}.json（run_kato1014.sh の C の出力）
旧版: analyze_greedy_3way.py（N1 = 200）と analyze_greedy_3way_k50.py（N1 = 50）。第 1 段の重み 0.7/0.3、Recall@K。
ここでは Recall@20 と MAP（標準）で、全ケースと正解式 8 本以上のケースについて、分割を対にした
Wilcoxon の符号順位検定（と対応のある t 検定）で比べる。
出力: experiments/kato1014/greedy_<tag>.json
"""
from __future__ import annotations

import argparse

from analyze_kato1014_mech import with_t
from kato1014_common import dump, load_records, seed_means, seed_summary

METRICS = ("Recall@20", "MAP", "MAP@20")
PAIRS = (("infer", "static"), ("train", "infer"), ("train", "static"))
SUBSETS = {"all": None, "K>=8": (lambda r: r["n_correct"] >= 8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = {"rundir": a.rundir, "summary": {}, "tests": {}}
    for S in ("A", "B"):
        G = {k: load_records(f"{a.rundir}/{k}_{S}__*.json") for k in ("static", "infer", "train")}
        G = {k: v for k, v in G.items() if v[0]}
        out["summary"][S], out["tests"][S] = {}, {}
        for sub_name, sub in SUBSETS.items():
            sm = {k: {m: seed_means(r, m, n, sub) for m in METRICS} for k, (r, n) in G.items()}
            out["summary"][S][sub_name] = {k: {m: seed_summary(sm[k][m]) for m in METRICS} for k in sm}
            out["summary"][S][sub_name]["n_records"] = {k: sum(1 for x in r if sub is None or sub(x))
                                                        for k, (r, _) in G.items()}
            out["tests"][S][sub_name] = {f"{x} - {y}": {m: with_t(sm[x][m], sm[y][m]) for m in METRICS}
                                         for x, y in PAIRS if x in sm and y in sm}
        for sub_name in SUBSETS:
            s = out["summary"][S][sub_name]
            print(f"[{S} {sub_name}] " + "  ".join(
                f"{k} R@20 {s[k]['Recall@20']['mean']:.4f} MAP {s[k]['MAP']['mean']:.4f}" for k in G))
            for k, v in out["tests"][S][sub_name].items():
                print(f"    {k:16s} " + "  ".join(
                    f"{m} {v[m]['mean_delta']:+.4f} ({v[m]['wins']}/10, W p={v[m]['p_wilcoxon']})" for m in METRICS))
    dump(out, f"greedy_{a.tag}.json")


if __name__ == "__main__":
    main()
