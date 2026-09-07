#!/usr/bin/env python3
"""第 2 判定器と第 1 判定器の等価判定の一致度を測る（B1）.

入力: experiments/llm_<label>_equiv_results.json          （第 1 判定器 claude-opus-4-8）
      experiments/llm_<label>_equiv_results_<tag>.json    （第 2 判定器; 既定 tag = fable51）
出力: experiments/judge_agreement_stats.json（--tag ごとに別名も可）

ペア（正解式 1 本）単位の二値ラベル 2 種について一致率と Cohen's κ を出す:
  matched : 等価な生成式が生成リストのどこかにある（coverage の判定単位）
  topk    : 等価な生成式が上位 K 件（K = 正解式数）にある（Recall@K の判定単位）
さらに番号まで同じ割合（index_agree_rate）、両判定器の Recall@K_correct / coverage とその差、
matched の 2×2 分割表（両者あり / 第 1 のみ / 第 2 のみ / 両者なし）を保存する。
第 2 判定器が判定できなかったケース（match_index が null）は除外し件数を記録する。
仕様: docs/superpowers/specs/2026-09-07-b1-second-judge-design.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
LABELS = ["set_A", "set_B", "set_full"]
SETTING_NAME = {"set_A": "A", "set_B": "B", "set_full": "full"}


def cohen_kappa(a, b) -> float:
    """二値（または多値）ラベル列の Cohen's κ。偶然一致が 1 のとき（分散なし）は nan."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def pair_labels(case: dict) -> dict:
    """1 ケースの match_index から、正解式ごとの二値ラベルを作る."""
    k = int(case["n_correct"])
    idx = [int(g) for g in case["match_index"]]
    return {
        "matched": [1 if g >= 1 else 0 for g in idx],
        "topk": [1 if 1 <= g <= k else 0 for g in idx],
        "index": idx,
    }


def _case_metrics(idx, k):
    return (sum(1 for g in idx if 1 <= g <= k) / k, sum(1 for g in idx if g >= 1) / k)


def agreement(ref_cases: list[dict], new_cases: list[dict]) -> dict:
    """第 1 判定器（ref）と第 2 判定器（new）の per_case を case_id で整列して集計する."""
    new_by = {c["case_id"]: c for c in new_cases}
    lab = {"ref": {"matched": [], "topk": [], "index": []},
           "new": {"matched": [], "topk": [], "index": []}}
    rk = {"ref": [], "new": []}
    cov = {"ref": [], "new": []}
    n_cases = n_missing = 0
    for rc in ref_cases:
        nc = new_by.get(rc["case_id"])
        if nc is None or nc.get("match_index") is None:
            n_missing += 1
            continue
        if int(nc["n_correct"]) != int(rc["n_correct"]):
            raise ValueError(f"{rc['case_id']}: n_correct differs ({rc['n_correct']} vs {nc['n_correct']})")
        if len(rc["match_index"]) != len(nc["match_index"]):
            raise ValueError(f"{rc['case_id']}: match_index length differs")
        k = int(rc["n_correct"])
        for key, c in (("ref", rc), ("new", nc)):
            pl = pair_labels(c)
            for name in ("matched", "topk", "index"):
                lab[key][name].extend(pl[name])
            r, v = _case_metrics(pl["index"], k)
            rk[key].append(r)
            cov[key].append(v)
        n_cases += 1
    n_pairs = len(lab["ref"]["matched"])

    def binary_block(name):
        a, b = lab["ref"][name], lab["new"][name]
        both = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
        ref_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        new_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        neither = sum(1 for x, y in zip(a, b) if x == 0 and y == 0)
        return {
            "agree_rate": (both + neither) / n_pairs if n_pairs else float("nan"),
            "kappa": cohen_kappa(a, b),
            "ref_positive_rate": sum(a) / n_pairs if n_pairs else float("nan"),
            "new_positive_rate": sum(b) / n_pairs if n_pairs else float("nan"),
            "table": {"both": both, "ref_only": ref_only, "new_only": new_only, "neither": neither},
        }

    out = {
        "n_cases": n_cases, "n_missing": n_missing, "n_pairs": n_pairs,
        "matched": binary_block("matched"),
        "topk": binary_block("topk"),
        "index_agree_rate": (sum(1 for x, y in zip(lab["ref"]["index"], lab["new"]["index"]) if x == y)
                             / n_pairs) if n_pairs else float("nan"),
    }
    for key in ("ref", "new"):
        out[key] = {"Recall@K_correct": float(np.mean(rk[key])) if rk[key] else float("nan"),
                    "coverage": float(np.mean(cov[key])) if cov[key] else float("nan")}
    out["delta_new_minus_ref"] = {
        "Recall@K_correct": out["new"]["Recall@K_correct"] - out["ref"]["Recall@K_correct"],
        "coverage": out["new"]["coverage"] - out["ref"]["coverage"],
    }
    return out


def _clean(o):
    """nan を JSON の null にする."""
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fable51")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    out = {"tag": args.tag, "labels": {}}
    pooled_ref, pooled_new = [], []
    for lab in LABELS:
        ref_p, new_p = EXP / f"llm_{lab}_equiv_results.json", EXP / f"llm_{lab}_equiv_results_{args.tag}.json"
        if not new_p.exists():
            print(f"skip {lab}: {new_p.name} not found")
            continue
        ref, new = json.load(open(ref_p)), json.load(open(new_p))
        stats = agreement(ref["per_case"], new["per_case"])
        stats["judge_ref"] = ref.get("judge_model")
        stats["judge_new"] = new.get("judge_model")
        stats["setting"] = SETTING_NAME[lab]
        out["labels"][lab] = stats
        pooled_ref.extend(ref["per_case"])
        pooled_new.extend(new["per_case"])
    if pooled_ref:
        out["pooled"] = agreement(pooled_ref, pooled_new)
    dst = Path(args.output) if args.output else EXP / "judge_agreement_stats.json"
    json.dump(_clean(out), open(dst, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {dst}\n")
    rows = list(out["labels"].items()) + ([("pooled", out["pooled"])] if "pooled" in out else [])
    print(f"{'set':8s} {'cases':>5s} {'miss':>4s} {'pairs':>5s} | {'matched agree':>13s} {'kappa':>6s} | "
          f"{'topk agree':>10s} {'kappa':>6s} | {'idx=':>5s} | {'R@K ref':>7s} {'R@K new':>7s} | {'cov ref':>7s} {'cov new':>7s}")
    for lab, s in rows:
        print(f"{lab:8s} {s['n_cases']:5d} {s['n_missing']:4d} {s['n_pairs']:5d} | "
              f"{s['matched']['agree_rate']:13.3f} {s['matched']['kappa']:6.3f} | "
              f"{s['topk']['agree_rate']:10.3f} {s['topk']['kappa']:6.3f} | {s['index_agree_rate']:5.3f} | "
              f"{s['ref']['Recall@K_correct']:7.4f} {s['new']['Recall@K_correct']:7.4f} | "
              f"{s['ref']['coverage']:7.4f} {s['new']['coverage']:7.4f}")


if __name__ == "__main__":
    main()
