"""LLM との比較を、式の ID で判定した値（同義性の判定なし）でまとめる：報告書と修論の数値の出典.

判定：手法が答えた式が、正解集合の式のどれかと同じ ID を持つなら正解、そうでなければ不正解。DB にない ID は不正解で
その順位を占める。同じ ID の 2 回目以降は除く（run_llm_db_baseline.judge_ids）。上位 20 件。
2026-11-16 本人の決定：報告書と修論は、この判定の値だけを載せる（llm_recall_dedup.py の値は使わない）。

入力:
  - LLM: experiments/llm_db_baseline/results_claude-{opus,haiku}-5-5__eqd__ex-none__high__v4.json の per_case（ケース × 3 回）
  - 提案手法: experiments/kato1021/raw/select/{設定}_{w}_{neg}__{種}.json（検証で選んだ設定。reranker-10S）
  - baseline: experiments/kato1021/raw/baselines_percase_{設定}.json（重みは分割ごとに検証で選んだもの）
集計: ケースの値（LLM は 3 回の平均）→ 分割のテストケースの平均 → 10 通りの分割の平均 ± 標準偏差（ddof=1）。
提案手法と baseline の値が experiments/kato1021/main.json と、LLM の値が results_*__v4.json の summary と
一致することを確かめ、食い違えば止める。

出力: experiments/llm_db_baseline/id_summary.json。使い方: python3 llm_id_summary.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_llm_db_baseline as rl  # noqa: E402

OUT_JSON = rl.OUT_DIR / "id_summary.json"
LLM_RESULTS = {"Opus 5.5": "results_claude-opus-5-5__eqd__ex-none__high__v4.json",
               "Haiku 5.5": "results_claude-haiku-5-5__eqd__ex-none__high__v4.json"}
KINDS = ("single", "cross", "synth")
MAIN_KEYS = {"proposed": "proposed_static", "baseline": "baseline_tfidf"}


def split_means(per_seed_cases: dict, key: str, kind_of: dict | None = None, kind: str | None = None) -> dict:
    """{種: {case_id: 指標}} → {種: その分割のテストケースの key の平均}。kind を与えると、その種類のケースだけで平均する
    （その種類のケースがない分割は除く）."""
    out = {}
    for sd, per in per_seed_cases.items():
        vals = [m[key] for c, m in per.items() if kind is None or kind_of[c] == kind]
        if vals:
            out[sd] = float(np.mean(vals))
    return out


def runs_agreement(runs: dict) -> dict:
    """{case_id: [回ごとの Recall@20]} → 回の間の標準偏差（ddof=1）のケース平均と、全回で同じ値だったケースの割合."""
    multi = {c: v for c, v in runs.items() if len(v) > 1}
    return {"n_cases": len(multi),
            "recall@20_sd_mean": float(np.mean([np.std(v, ddof=1) for v in multi.values()])),
            "cases_all_runs_same": float(np.mean([len({round(x, 9) for x in v}) == 1 for v in multi.values()]))}


def case_kind(c: dict) -> str:
    v = c.get("variant_type", "")
    return "single" if v == "original" else "cross" if v.startswith("multisource_") else "synth" if v.startswith("dae_") else v


def mean_sd(per_seed: dict) -> dict:
    a = np.asarray(list(per_seed.values()), dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(a.size), "per_seed": per_seed}


def main() -> None:
    cases, splits = rl.load_case_index(), rl.load_splits()
    kind_of = {c: case_kind(v) for c, v in cases.items()}
    seeds = [str(s) for s in rl.ALL_SEEDS]
    main_json = json.loads((rl.REF_RUN / "main.json").read_text(encoding="utf-8"))
    llm_raw = {n: json.loads((rl.OUT_DIR / f).read_text(encoding="utf-8")) for n, f in LLM_RESULTS.items()}
    out = {"rule": __doc__.split("判定：")[1].split("入力:")[0].strip(), "settings": {}, "models": {}}

    for name, res in llm_raw.items():
        by = defaultdict(list)
        for r in res["per_case"]:
            by[r["case_id"]].append(r["recall@20"])
        out["models"][name] = {**runs_agreement(by),
                               "jaccard_mean": res["variability"]["jaccard_mean"],
                               "same_list_share": res["variability"]["same_list_share"]}

    for s in ("A", "B"):
        psc = {}
        # 提案手法
        psc["proposed"] = {}
        for sd in seeds:
            sel = rl._selected_ref(s, int(sd))
            f = rl.REF_RUN / "raw" / "select" / f"{s}_{sel['w']}_{sel['neg']}__{sd}.json"
            recs = rl.split_test_records(json.loads(f.read_text(encoding="utf-8"))["results"]["reranker-10S"]["per_case"], int(sd))
            psc["proposed"][sd] = {r["case_id"]: rl.metrics_from_ranks(r["ranks"]) for r in recs}
        # baseline
        psc["baseline"] = defaultdict(dict)
        pc = json.loads((rl.REF_RUN / "raw" / f"baselines_percase_{s}.json").read_text(encoding="utf-8"))
        for r in pc["results"]["baseline"]["per_case"]:
            psc["baseline"][str(r["seed"])][r["case_id"]] = rl.metrics_from_ranks(r["ranks"])
        psc["baseline"] = dict(psc["baseline"])
        # LLM（ケースの値は 3 回の平均）
        for name, res in llm_raw.items():
            by = defaultdict(list)
            for r in res["per_case"]:
                by[r["case_id"]].append(r)
            mean = {c: {"recall@20": float(np.mean([x["recall@20"] for x in xs])),
                        "AP@20": float(np.mean([x["AP@20"] for x in xs]))} for c, xs in by.items()}
            psc[name] = {sd: {c: mean[c] for c in splits[s][sd]["test"]} for sd in seeds}
        # テストケースの並びがそろっているか
        for name, per in psc.items():
            for sd in seeds:
                if set(per[sd]) != set(splits[s][sd]["test"]):
                    sys.exit(f"{name} {s} {sd}: テストケースが splits.json と食い違う")

        st = {"n_test_by_kind": {k: {sd: sum(1 for c in splits[s][sd]["test"] if kind_of[c] == k) for sd in seeds}
                                 for k in KINDS},
              "methods": {}, "compare": {}}
        for name, per in psc.items():
            m = {key: mean_sd(split_means(per, key)) for key in ("recall@20", "AP@20")}
            m["by_kind"] = {k: {key: mean_sd(v) for key in ("recall@20", "AP@20")
                                if (v := split_means(per, key, kind_of, k))} for k in KINDS}
            st["methods"][name] = m
        # 既存の値との照合
        for name, mk in MAIN_KEYS.items():
            for key, mkey in (("recall@20", "Recall@20"), ("AP@20", "MAP@20")):
                ref = main_json["table"][s][mk][mkey]["per_seed"]
                got = st["methods"][name][key]["per_seed"]
                if any(abs(got[sd] - ref[sd]) > 1e-9 for sd in seeds):
                    sys.exit(f"{name} {s} {key}: kato1021/main.json の値と食い違う")
        for name, res in llm_raw.items():
            ref = res["summary"][f"{s}_10splits"]["per_seed"]
            for key, rkey in (("recall@20", "recall@20"), ("AP@20", "MAP@20")):
                got = st["methods"][name][key]["per_seed"]
                if any(abs(got[sd] - ref[sd][rkey]) > 1e-9 for sd in seeds):
                    sys.exit(f"{name} {s} {key}: results の summary と食い違う")
        # 比較（a − b、a が上回った分割の数）
        for a in LLM_RESULTS:
            for b in ("proposed", "baseline"):
                st["compare"][f"{a} vs {b}"] = {
                    key: rl.compare_splits(st["methods"][a][key]["per_seed"], st["methods"][b][key]["per_seed"])
                    for key in ("recall@20", "AP@20")}
        st["compare"]["proposed vs baseline"] = {
            key: rl.compare_splits(st["methods"]["proposed"][key]["per_seed"], st["methods"]["baseline"][key]["per_seed"])
            for key in ("recall@20", "AP@20")}
        out["settings"][s] = st

    out["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    for s, st in out["settings"].items():
        print(f"### setting {s} (ID judgment; mean ± SD over 10 splits)")
        for name, m in st["methods"].items():
            print(f"  {name:9s} R@20 {m['recall@20']['mean']:.3f} ± {m['recall@20']['sd']:.3f}  "
                  f"MAP@20 {m['AP@20']['mean']:.3f} ± {m['AP@20']['sd']:.3f}")
        for k, v in st["compare"].items():
            print(f"    {k:22s} R@20 {v['recall@20']['mean_diff']:+.3f} ({v['recall@20']['wins']}/10)  "
                  f"MAP@20 {v['AP@20']['mean_diff']:+.3f} ({v['AP@20']['wins']}/10)")
        if s == "A":
            for k in KINDS:
                print(f"  {k:6s} " + " | ".join(
                    f"{n} R {m['by_kind'][k]['recall@20']['mean']:.3f}±{m['by_kind'][k]['recall@20']['sd']:.3f} "
                    f"MAP {m['by_kind'][k]['AP@20']['mean']:.3f}±{m['by_kind'][k]['AP@20']['sd']:.3f}"
                    for n, m in st["methods"].items()))
            print("  test cases per split by kind:", {k: (min(v.values()), max(v.values())) for k, v in st["n_test_by_kind"].items()})
    for name, v in out["models"].items():
        print(f"{name}: runs R@20 SD mean {v['recall@20_sd_mean']:.3f}, all runs same {v['cases_all_runs_same']:.3f}, "
              f"Jaccard {v['jaccard_mean']:.3f}, same list {v['same_list_share']:.4f}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
