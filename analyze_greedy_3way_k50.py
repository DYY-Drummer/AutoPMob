"""逐次選択の 3 通りの比較を候補数 k=50 で集計する（加藤先生の注釈「なぜ k=50 で行わないのか」への対応）.

入力:
  - 一括（静的参照）: experiments/strat_A.json / strat_B_percase.json の reranker-10S per_case（k=50）
  - 推論のみ逐次（infer）/ 学習も逐次（train）: experiments/xk50/{infer,train}_{A,B}__{seed}.json
    （run_kato_0831_followup.sh。条件は run_greedy_3way.sh と同じで top_k だけ 50）
出力: experiments/greedy_3way_k50_stats.json
検定・集計は analyze_greedy_3way.py の seed_means / paired をそのまま使う（seed 平均の対応のある検定）。
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from analyze_greedy_3way import paired, seed_means

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"
STATIC = {"A": EXP / "strat_A.json", "B": EXP / "strat_B_percase.json"}


def load_static(setting):
    d = json.load(open(STATIC[setting], encoding="utf-8"))
    return [dict(r) for r in d["results"]["reranker-10S"]["per_case"]]


def load_cfg(setting, cfg):
    recs = []
    for f in sorted(glob.glob(str(EXP / "xk50" / f"{cfg}_{setting}__*.json"))):
        if not re.search(rf"{cfg}_{setting}__\d+\.json$", f):
            continue
        d = json.load(open(f, encoding="utf-8"))
        for r in d["results"].values():
            recs += [dict(x) for x in r.get("per_case", [])]
    return recs


def main():
    out = {"metric": "Recall@K_correct", "top_k": 50}
    for setting in ("A", "B"):
        data = {"static": load_static(setting), "infer": load_cfg(setting, "infer"),
                "train": load_cfg(setting, "train")}
        res = {"n_records": {c: len(v) for c, v in data.items()}, "overall": {}}
        for c, recs in data.items():
            vals = list(seed_means(recs).values())
            mean = sum(vals) / len(vals) if vals else None
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else None
            res["overall"][c] = {"mean": round(mean, 4) if mean is not None else None,
                                 "std": round(sd, 4) if sd is not None else None, "n_seeds": len(vals)}
        res["tests"] = {
            "infer_vs_static": paired(data["infer"], data["static"]),
            "train_vs_infer": paired(data["train"], data["infer"]),
            "train_vs_static": paired(data["train"], data["static"]),
        }
        hard = lambda r: r.get("n_correct", 1) >= 8  # noqa: E731
        res["tests_hard_X8plus"] = {
            "train_vs_static": paired(data["train"], data["static"], sub=hard),
            "train_vs_infer": paired(data["train"], data["infer"], sub=hard),
            "infer_vs_static": paired(data["infer"], data["static"], sub=hard),
        }
        out[setting] = res
        print(setting, json.dumps(res["overall"], ensure_ascii=False))
        for k, v in res["tests"].items():
            print("  ", k, v)
        for k, v in res["tests_hard_X8plus"].items():
            print("   X>=8", k, v)
    path = EXP / "greedy_3way_k50_stats.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
