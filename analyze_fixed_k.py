"""提示件数 K の決め方の比較：K を正解式の数に合わせる場合と、K を固定する場合.

加藤先生の問い「モデルを構成する数式の数は事前にわからないはず。どうやって決めるのか」
（2026-09-29）を受け、本人の指示で検証する。上位 c 件を提示したとき、
  - c = K（正解式の数。正典の Recall@K と同じ）
  - c = 5, 10, 20（固定。テストデータの正解式の数は最大 10 なので、10 で全モデルの大きさを覆う）
  - c = 50（第 1 段の候補数 N1。ここでは古典的手法と本手法の上位 50 件が同じ集合になる）
の各場合に、次を計算する（ケースごとの平均を乱数 10 通りで平均）。
  recall@c    : 正解式のうち上位 c 件に入った割合
  full@c      : 正解式が全部上位 c 件に入ったケースの割合（提示された一覧から正解モデルを組める）
  precision@c : 上位 c 件のうち正解式の割合
  wrong@c     : 上位 c 件に入った不正解式の数（ユーザが見分ける必要がある式の数）

入力 : experiments/strat_A.json（per_case あり）、experiments/strat_B_percase.json（設定 B の再実行）
出力 : experiments/fixed_k_stats.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"
INPUTS = {
    "A": EXP / "strat_A.json",
    "B": EXP / "strat_B_percase.json",
}
MODES = ["baseline", "reranker-10S"]
CUTS = ["K", 5, 10, 20, 50]
K_GROUPS = {"K=1": (1, 1), "K=2": (2, 2), "K=3-5": (3, 5), "K=6-10": (6, 10)}


def case_metrics(ranks: list[int], K: int, cut) -> dict[str, float]:
    c = K if cut == "K" else int(cut)
    hits = sum(1 for r in ranks if r <= c)
    return {
        "recall": hits / K,
        "full": float(hits == K),
        "precision": hits / c,
        "wrong": float(c - hits),
    }


def per_seed_table(records: list[dict], k_range=None) -> dict:
    """{cut: {metric: [seed 平均, ...]}}（seed 昇順）"""
    by_seed = defaultdict(list)
    for r in records:
        K = r["n_correct"]
        if k_range and not (k_range[0] <= K <= k_range[1]):
            continue
        by_seed[r["seed"]].append(r)
    out = {}
    for cut in CUTS:
        out[str(cut)] = defaultdict(list)
        for seed in sorted(by_seed):
            rows = [case_metrics(r["ranks"], r["n_correct"], cut) for r in by_seed[seed]]
            for m in ("recall", "full", "precision", "wrong"):
                out[str(cut)][m].append(float(np.mean([x[m] for x in rows])))
    return out, sorted(by_seed), sum(len(v) for v in by_seed.values())


def summarize(setting: str, path: Path) -> dict | None:
    if not path.exists():
        print(f"[{setting}] {path.name} がないので省略")
        return None
    data = json.loads(path.read_text())["results"]
    recs = {m: data[m]["per_case"] for m in MODES}
    # 2 手法のレコードが同じ (seed, case) の並びであることを確かめる
    assert all(a["case_id"] == b["case_id"] and a["seed"] == b["seed"]
               for a, b in zip(recs[MODES[0]], recs[MODES[1]]))
    result = {"n_records": len(recs[MODES[0]]), "overall": {}, "by_K": {}}

    def compare(k_range=None):
        tabs = {}
        for m in MODES:
            tabs[m], seeds, n = per_seed_table(recs[m], k_range)
        block = {"n_records": n, "seeds": seeds}
        for cut in map(str, CUTS):
            block[cut] = {}
            for metric in ("recall", "full", "precision", "wrong"):
                b = np.array(tabs["baseline"][cut][metric])
                p = np.array(tabs["reranker-10S"][cut][metric])
                diff = p - b
                if np.allclose(diff, 0):
                    pval = 1.0
                else:
                    pval = float(stats.wilcoxon(p, b).pvalue)
                block[cut][metric] = {
                    "baseline": [round(float(b.mean()), 4), round(float(b.std(ddof=1)), 4)],
                    "proposed": [round(float(p.mean()), 4), round(float(p.std(ddof=1)), 4)],
                    "diff": round(float(diff.mean()), 4),
                    "n_seeds_proposed_better": int((diff > 1e-12).sum()),
                    "p_wilcoxon": pval,
                }
        return block

    result["overall"] = compare()
    for g, rng in K_GROUPS.items():
        result["by_K"][g] = compare(rng)
    # 上位 50 件は第 1 段の候補そのもの：両手法で正解の入り方が同じかをケースごとに確認
    same50 = sum(
        1 for a, b in zip(recs["baseline"], recs["reranker-10S"])
        if sum(1 for r in a["ranks"] if r <= 50) == sum(1 for r in b["ranks"] if r <= 50)
    )
    result["same_hits_at_50"] = f"{same50}/{len(recs['baseline'])}"
    return result


def main():
    out = {}
    for setting, path in INPUTS.items():
        res = summarize(setting, path)
        if res is None:
            continue
        out[setting] = res
        ov = res["overall"]
        print(f"\n=== 設定 {setting}（延べ {res['n_records']} 件、上位 50 件で正解数が一致: {res['same_hits_at_50']}）")
        print(f"{'c':>4} | {'recall 古典→本手法 (差, 勝ち数, p)':<42} | {'full':<26} | {'precision':<26} | wrong")
        for cut in map(str, CUTS):
            row = ov[cut]
            def f(m):
                x = row[m]
                return f"{x['baseline'][0]:.3f}→{x['proposed'][0]:.3f} ({x['diff']:+.3f}, {x['n_seeds_proposed_better']}/10, p={x['p_wilcoxon']:.3g})"
            w = row["wrong"]
            print(f"{cut:>4} | {f('recall'):<42} | {row['full']['baseline'][0]:.3f}→{row['full']['proposed'][0]:.3f} ({row['full']['diff']:+.3f}) "
                  f"| {row['precision']['baseline'][0]:.3f}→{row['precision']['proposed'][0]:.3f} | {w['baseline'][0]:.2f}→{w['proposed'][0]:.2f}")
        print("  正解式の数 K ごと（recall / full、古典→本手法）:")
        for g, blk in res["by_K"].items():
            cells = []
            for cut in ("K", "10", "20"):
                r = blk[cut]["recall"]; fu = blk[cut]["full"]
                cells.append(f"c={cut}: R {r['baseline'][0]:.3f}→{r['proposed'][0]:.3f}, full {fu['baseline'][0]:.3f}→{fu['proposed'][0]:.3f}")
            print(f"   {g:<6} (延べ {blk['n_records']:>4}) " + " | ".join(cells))
    (EXP / "fixed_k_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\nSaved: experiments/fixed_k_stats.json")


if __name__ == "__main__":
    main()
