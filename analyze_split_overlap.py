"""分割の性質を数える：テスト用のケースのうち、正解式を訓練用のケースと共有するものの割合.

分割は、各ケースを「正解式を最も多く出している出典」に割り当て、出典ごとに訓練用・検証用・テスト用に分ける
（set_aware_reranker.stratified_src_split）。同じ出典に割り当てたケースは同じ側に入るが、複数の出典の式から成る
正解集合（出典横断、合成の一部）は、割り当て先でない出典の式を含むので、その式を訓練用のケースの正解集合と共有しうる。
このスクリプトは、その件数を数える（式は ID で照合する。式の出典は ID の「出典__式番号」の前半）。

入力: experiments/llm_db_baseline/splits.json（設定 A・B、10 通り）、training_cases.json
出力: experiments/llm_db_baseline/split_overlap.json。使い方: python3 analyze_split_overlap.py
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "experiments" / "llm_db_baseline" / "splits.json"
OUT = ROOT / "experiments" / "llm_db_baseline" / "split_overlap.json"


def source_of(eq_id: str) -> str:
    return eq_id.split("__", 1)[0]


def ground_truth_ids(case: dict) -> list:
    """ケースの正解式の ID（リスト、またはリストを文字列にしたもの）を、前後の空白を除いたリストで返す."""
    ids = case.get("correct_model_ids") or []
    ids = ast.literal_eval(ids) if isinstance(ids, str) else ids
    return [i.strip() for i in ids]


def shared_with_training(split: dict, gt_of: dict) -> dict:
    """テスト用のケースのうち、(a) 正解式（同じ ID）を訓練用のケースの正解集合と共有するもの、
    (b) 正解式の出典が訓練用のケースの正解式の出典でもあるもの、を数える。検証用のケースは数えない."""
    train_eq = {i for c in split["train"] for i in gt_of[c]}
    train_src = {source_of(i) for i in train_eq}
    shared = [c for c in split["test"] if any(i in train_eq for i in gt_of[c])]
    shared_src = [c for c in split["test"] if any(source_of(i) in train_src for i in gt_of[c])]
    return {"n_test": len(split["test"]),
            "shared_equation": len(shared),
            "shared_source": len(shared_src),
            "cases_shared_equation": shared,
            "cases_shared_source": shared_src}


def summarize(per_seed: dict) -> dict:
    """分割ごとの件数を、10 通りを合わせた延べの割合と、分割ごとの割合の最小・最大にまとめる."""
    n = sum(v["n_test"] for v in per_seed.values())
    e = sum(v["shared_equation"] for v in per_seed.values())
    s = sum(v["shared_source"] for v in per_seed.values())
    shares = [v["shared_equation"] / v["n_test"] for v in per_seed.values()]
    return {"n_test_records": n, "shared_equation": e, "shared_equation_share": e / n,
            "shared_equation_share_min": min(shares), "shared_equation_share_max": max(shares),
            "shared_source": s, "shared_source_share": s / n}


def case_kind(case: dict) -> str:
    v = case.get("variant_type", "")
    return "synth" if v.startswith("dae_") else "cross" if v.startswith("multisource") else "single"


def main() -> None:
    cases = {c["case_id"]: c for c in json.loads((ROOT / "training_cases.json").read_text(encoding="utf-8"))}

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    gt_of = {c: ground_truth_ids(cases[c]) for s in splits.values() for d in s.values()
             for part in ("train", "val", "test") for c in d[part]}
    out = {"definition": __doc__.split("入力:")[0].strip(), "settings": {}}
    for setting, by_seed in splits.items():
        per_seed = {seed: shared_with_training(d, gt_of) for seed, d in by_seed.items()}
        kinds = defaultdict(lambda: [0, 0])
        for seed, d in by_seed.items():
            shared = set(per_seed[seed]["cases_shared_equation"])
            for c in d["test"]:
                kinds[case_kind(cases[c])][0] += 1
                kinds[case_kind(cases[c])][1] += c in shared
        out["settings"][setting] = {
            **summarize(per_seed),
            "by_kind": {k: {"n_test_records": n, "shared_equation": e, "shared_equation_share": e / n}
                        for k, (n, e) in sorted(kinds.items())},
            "per_seed": {seed: {k: v for k, v in d.items() if not k.startswith("cases_")} for seed, d in per_seed.items()},
        }
        s = out["settings"][setting]
        print(f"setting {setting}: {s['shared_equation']}/{s['n_test_records']} test records share a ground-truth equation "
              f"with a training case ({s['shared_equation_share']:.1%}; {s['shared_equation_share_min']:.1%} to "
              f"{s['shared_equation_share_max']:.1%} per split); by kind "
              + ", ".join(f"{k} {v['shared_equation_share']:.1%}" for k, v in s["by_kind"].items()))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
