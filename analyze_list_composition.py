"""考察の 3 つの問いのための集計（API は呼ばない。保存してある結果と式データベースだけを使う）.

問い（2026-11-17、加藤先生）：提案手法を基準に、(1) なぜ 100% に至らないか、(2) baseline との差とその理由、
(3) LLM との差とその理由。

数えるもの
- ケースごと：クエリの入出力変数に、変数がすべて含まれる式（特化度が 1 の式）がデータベースに何本あるか（n_cov）。
  提示するのは 20 本なので、n_cov が 20 を超えるケースでは、特化度だけでは正解式を選び切れない。
- 手法ごと：上位 20 件の内訳（正解式、変数がすべてクエリに含まれる不正解の式、クエリにない変数を含む不正解の式、
  データベースにない ID、埋まっていない枠）。正解式をすべて上位 20 件に入れたケースの割合。n_cov が 20 以下の
  ケースと 20 を超えるケースに分けた Recall@20。
- 正解式の数ごとの Recall@20（4 手法）。入力変数の数ごとの、正解式の I/O Jaccard の平均。

集計：分割ごとにテスト用のケースで平均し、10 通りの分割の平均と標準偏差（ddof=1）を出す。LLM のケースの値は 3 回の平均。
正解式の数ごとの値は、修論と論文の特性別の図と同じく、10 通りの分割のテスト用のケースを延べで平均する。

入力: results_claude-{opus,haiku}-5-5__eqd__ex-none__high__v4.json、proposed_top20/{設定}__{種}.json、
      experiments/kato1021/raw/baselines_percase_{設定}.json、unified_equations.json、training_cases.json
出力: experiments/llm_db_baseline/list_composition.json。使い方: python3 analyze_list_composition.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT_NAME = "list_composition.json"
TOP_N = 20
LLM_RESULTS = {"Opus 5.5": "results_claude-opus-5-5__eqd__ex-none__high__v4.json",
               "Haiku 5.5": "results_claude-haiku-5-5__eqd__ex-none__high__v4.json"}
PARTS = ("correct", "covered", "outside", "invalid", "empty")


def covered(eq_vars: set, query_vars: set) -> bool:
    """式の変数が 1 つ以上あり、そのすべてがクエリの入出力変数に含まれる（特化度が 1）."""
    return bool(eq_vars) and eq_vars <= query_vars


def composition(keys: list, correct: set, is_covered, top_n: int = TOP_N) -> dict:
    """1 つのリストの上位 top_n 件の内訳。keys は式のキーの並び（データベースにない ID は None）.

    correct：正解式。covered：不正解で、変数がすべてクエリに含まれる。outside：不正解で、クエリにない変数を含む
    （変数の記録がない式もここ）。invalid：データベースにない ID。empty：埋まっていない枠。足すと top_n."""
    out = dict.fromkeys(PARTS, 0)
    head = list(keys)[:top_n]
    for k in head:
        if k is None:
            out["invalid"] += 1
        elif k in correct:
            out["correct"] += 1
        elif is_covered(k):
            out["covered"] += 1
        else:
            out["outside"] += 1
    out["empty"] = top_n - len(head)
    return out


def k_bin(k: int) -> str:
    return "1" if k == 1 else "2-3" if k <= 3 else "4-6" if k <= 6 else "7-10"


def input_bin(n: int) -> str:
    return "<=5" if n <= 5 else "6-10" if n <= 10 else "11-15" if n <= 15 else "16-20" if n <= 20 else ">=21"


def mean_sd(values) -> dict:
    """平均と標準偏差（ddof=1）。None は除き、使った個数を n に書く。値が 1 つなら標準偏差は None."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v)}


def group_stats(records: list) -> dict:
    """ケースの記録 [{recall, complete, n_gt, missed}] → 件数、Recall@20 の平均、正解式をすべて入れた割合、取りこぼした式の数."""
    if not records:
        return {"n": 0, "recall": None, "complete": None, "missed": 0.0, "n_gt": 0}
    return {"n": len(records), "recall": float(np.mean([r["recall"] for r in records])),
            "complete": float(np.mean([r["complete"] for r in records])),
            "missed": float(sum(r["missed"] for r in records)), "n_gt": int(sum(r["n_gt"] for r in records))}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import run_llm_db_baseline as rl
    from analyze_opus_vs_proposed import load_top
    from two_stage_query_conditioned import eq_key, eq_vars, io_vars, in_vars, jaccard, load_equations, norm

    cases, splits = rl.load_case_index(), rl.load_splits()
    keys, var_sets = [], []
    for e in load_equations():
        k = eq_key(e)
        if k:
            keys.append(k)
            var_sets.append(eq_vars(e))
    vars_of = dict(zip(keys, var_sets))
    llm = {}
    for name, f in LLM_RESULTS.items():
        by = defaultdict(list)
        for r in json.loads((rl.OUT_DIR / f).read_text(encoding="utf-8"))["per_case"]:
            by[r["case_id"]].append(r)
        llm[name] = by

    out = {"definition": __doc__.split("入力:")[0].strip(), "top_n": TOP_N, "settings": {},
           "provenance": {"unified_equations.json": _sha(rl.EQ_JSON), "training_cases.json": _sha(rl.CASES_JSON)}}
    for s in ("A", "B"):
        seeds = [str(x) for x in rl.ALL_SEEDS]
        first = splits[s][seeds[0]]
        all_cases = sorted(set(first["train"]) | set(first["val"]) | set(first["test"]))
        info = {}
        for c in all_cases:
            io = io_vars(cases[c])
            correct = list(dict.fromkeys(norm(x) for x in cases[c]["correct_model_ids"] if norm(x) in vars_of))
            info[c] = {"io": io, "correct": set(correct), "K": len(correct), "n_in": len(in_vars(cases[c])),
                       "kind": rl.case_kind(cases[c]["variant_type"]),
                       "n_cov": sum(1 for v in var_sets if len(v) <= len(io) and covered(v, io)),
                       "gt_all_covered": all(covered(vars_of[k], io) for k in correct),
                       "gt_jaccard": [jaccard(io, vars_of[k]) for k in correct],
                       "gt_n_vars": [len(vars_of[k]) for k in correct]}
        n_cov = np.array([info[c]["n_cov"] for c in all_cases])
        st = {"cases": {"n": len(all_cases), "n_cov_median": float(np.median(n_cov)), "n_cov_mean": float(n_cov.mean()),
                        "n_cov_over": int((n_cov > TOP_N).sum()), "share_cov_over": float((n_cov > TOP_N).mean()),
                        "gt_all_covered": sum(info[c]["gt_all_covered"] for c in all_cases),
                        "by_kind": {kd: {"n": sum(1 for c in all_cases if info[c]["kind"] == kd),
                                         "n_cov_over": sum(1 for c in all_cases if info[c]["kind"] == kd and info[c]["n_cov"] > TOP_N)}
                                    for kd in ("single", "cross", "synth")}}}
        # 正解式の I/O Jaccard を、入力変数の数ごとに（ケースの延べなし）
        bins = defaultdict(lambda: {"cases": 0, "jac": [], "n_vars": [], "n_query": []})
        for c in all_cases:
            b = bins[input_bin(info[c]["n_in"])]
            b["cases"] += 1
            b["jac"] += info[c]["gt_jaccard"]
            b["n_vars"] += info[c]["gt_n_vars"]
            b["n_query"].append(len(info[c]["io"]))
        st["gt_jaccard_by_inputs"] = {k: {"n_cases": b["cases"], "n_gt": len(b["jac"]), "jaccard": float(np.mean(b["jac"])),
                                          "n_vars": float(np.mean(b["n_vars"])), "n_query_vars": float(np.mean(b["n_query"]))}
                                      for k, b in bins.items()}

        # 手法ごとの、ケースの記録（分割ごと）
        top = load_top(s)
        base = defaultdict(dict)
        pc = json.loads((rl.REF_RUN / "raw" / f"baselines_percase_{s}.json").read_text(encoding="utf-8"))
        for r in pc["results"]["baseline"]["per_case"]:
            base[str(r["seed"])][r["case_id"]] = r["ranks"]

        def record(c, recall, comp=None, complete=None):
            i = info[c]
            return {"case": c, "recall": recall, "complete": float(recall == 1.0) if complete is None else complete,
                    "n_gt": i["K"], "missed": (1.0 - recall) * i["K"], "over": i["n_cov"] > TOP_N, "comp": comp}

        per = {"proposed": {}, "baseline": {}, **{name: {} for name in llm}}
        for sd in seeds:
            test = splits[s][sd]["test"]
            rows = []
            for c in test:
                v = top[c][int(sd)]
                i = info[c]
                rows.append(record(c, rl.metrics_from_ranks(v["ranks"])["recall@20"],
                                   composition(v["top"], i["correct"], lambda k, io=i["io"]: covered(vars_of[k], io))))
            per["proposed"][sd] = rows
            per["baseline"][sd] = [record(c, rl.metrics_from_ranks(base[sd][c])["recall@20"]) for c in test]
            for name, by in llm.items():
                rows = []
                for c in test:
                    i = info[c]
                    runs = by[c]
                    comps = [composition(r["ranked_keys"] or [], i["correct"], lambda k, io=i["io"]: covered(vars_of[k], io))
                             for r in runs]
                    rows.append(record(c, float(np.mean([r["recall@20"] for r in runs])),
                                       {p: float(np.mean([x[p] for x in comps])) for p in PARTS},
                                       complete=float(np.mean([r["recall@20"] == 1.0 for r in runs]))))
                per[name][sd] = rows

        st["methods"] = {}
        for name, by_seed in per.items():
            m = {"recall": mean_sd(group_stats(r)["recall"] for r in by_seed.values()),
                 "complete": mean_sd(group_stats(r)["complete"] for r in by_seed.values()),
                 "by_cov": {}}
            for label, flag in (("within", False), ("over", True)):
                g = {sd: group_stats([x for x in r if x["over"] == flag]) for sd, r in by_seed.items()}
                m["by_cov"][label] = {"n_records": sum(v["n"] for v in g.values()),
                                      "recall": mean_sd(v["recall"] for v in g.values()),
                                      "complete": mean_sd(v["complete"] for v in g.values()),
                                      "missed": float(sum(v["missed"] for v in g.values()))}
            total = m["by_cov"]["within"]["missed"] + m["by_cov"]["over"]["missed"]
            m["missed"] = {"total": total, "share_over": m["by_cov"]["over"]["missed"] / total if total else None}
            if name != "baseline":
                m["composition"] = {p: mean_sd(float(np.mean([x["comp"][p] for x in r])) for r in by_seed.values())
                                    for p in PARTS}
            st["methods"][name] = m

        # 正解式の数ごと（10 通りの分割のテスト用のケースを延べで平均）
        st["by_k"] = {}
        for b in ("1", "2-3", "4-6", "7-10"):
            row = {}
            for name, by_seed in per.items():
                vals = [x["recall"] for r in by_seed.values() for x in r if k_bin(x["n_gt"]) == b]
                row[name] = float(np.mean(vals)) if vals else None
                row["n_records"] = len(vals)
            st["by_k"][b] = row
        out["settings"][s] = st
        c0 = st["cases"]
        print(f"setting {s}: {c0['n']} cases; equations fully covered by the query: median {c0['n_cov_median']:.0f}, "
              f"more than {TOP_N} in {c0['n_cov_over']} cases ({c0['share_cov_over']:.1%}); "
              f"ground truth fully covered in {c0['gt_all_covered']} cases")
        for name, m in st["methods"].items():
            comp = "  ".join(f"{p} {v['mean']:.2f}" for p, v in m.get("composition", {}).items())
            print(f"  {name:<9} recall {m['recall']['mean']:.3f}  complete {m['complete']['mean']:.3f}  "
                  f"within {m['by_cov']['within']['recall']['mean']:.3f} (n {m['by_cov']['within']['n_records']})  "
                  f"over {m['by_cov']['over']['recall']['mean']:.3f} (n {m['by_cov']['over']['n_records']})  "
                  f"missed in over {m['missed']['share_over']:.3f}  | {comp}")
        print("  by K:", {b: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for b, r in st["by_k"].items()})
        print("  GT Jaccard by inputs:", {k: (round(v["jaccard"], 3), round(v["n_vars"], 1), round(v["n_query_vars"], 1))
                                          for k, v in st["gt_jaccard_by_inputs"].items()})
    f = rl.OUT_DIR / OUT_NAME
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
