"""提案手法を、比べる相手の実行（experiments/kato1021/raw/select の選ばれた設定）と同じ設定・同じ乱数で学習し直し、
テストケースごとに上位 20 件の式を書き出す（LLM との比較で、どの式が正解式を上位 20 件から押し出したかを調べるため）.

既存の出力には正解式の順位しかないので、並べた式そのものを残す。学習し直した順位が既存の出力と
ケースごとに一致することを確かめ、食い違えば止める。

出力: experiments/llm_db_baseline/proposed_top20/{設定}__{種}.json
  {"setting", "seed", "config_from", "n_test", "rank_mismatch": 0,
   "cases": {case_id: {"top": [式の ID, ...（20 件）], "scores": [...], "ranks": [正解式の順位]}}}

例: python3 dump_proposed_top20.py --setting A --seeds 42,123
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import set_aware_reranker as sar  # noqa: E402
from two_stage_query_conditioned import (  # noqa: E402
    load_equations, load_cases, eq_key, eq_text, eq_vars, norm, case_src,
)
import run_llm_db_baseline as rl  # noqa: E402

OUT = ROOT / "experiments" / "llm_db_baseline" / "proposed_top20"
MODE = "reranker-10S"


def inputs(cfg: dict) -> tuple:
    """set_aware_reranker.main() と同じ手順で、ケースと式の列を作る."""
    eqs, cases = load_equations(), load_cases()
    wanted = [v.strip() for v in (cfg.get("variants") or "").split(",") if v.strip()]
    if wanted:
        cases = [c for c in cases if any(c.get("variant_type", "").startswith(w) if w.endswith("_")
                                         else c.get("variant_type", "") == w for w in wanted)]
    if cfg.get("exclude_cases"):
        cases = sar.filter_excluded(cases, ROOT / cfg["exclude_cases"])
    ek, et, ev, ed, es, ex = [], [], [], [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or "")); es.append(sar.get_src(e)); ex.append(sar.latex_key(e.get("equation")))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]
    return cases, ek, et, ev, ed, es, ex, cl, cs


def run_one(setting: str, seed: int, top: int) -> dict:
    sel = rl._selected_ref(setting, seed)
    ref_f = rl.REF_RUN / "raw" / "select" / f"{setting}_{sel['w']}_{sel['neg']}__{seed}.json"
    ref = json.loads(ref_f.read_text(encoding="utf-8"))
    cfg = ref["config"]
    cases, ek, et, ev, ed, es, ex, cl, cs = inputs(cfg)
    captured = {}

    def on_trained(info):
        model = info["model"]
        with torch.no_grad():
            for ci in info["te"]:
                cands = info["get_cands"](ci)
                feats = info["get_feats"](ci, cands)
                scores = model(torch.tensor(feats, dtype=torch.float32)).numpy().ravel()
                order = sorted(range(len(cands)), key=lambda k: -scores[k])   # 評価と同じ並べ方
                captured[cases[ci]["case_id"]] = {"top": [ek[cands[k]] for k in order[:top]],
                                                  "scores": [round(float(scores[k]), 6) for k in order[:top]]}

    res = sar.run_mode(
        MODE, [seed], cases, ek, et, ev, ed, es, cl, cs, cfg["top_k"], cfg["epochs"], cfg["lr"],
        save_per_case=True, hidden_dim=cfg["hidden_dim"], margin=cfg["margin"], batch_size=cfg["batch_size"],
        n_neg_samples=cfg["n_neg_samples"], weight_decay=cfg["weight_decay"], loss_type=cfg["loss"],
        hard_neg=cfg["hard_neg"], greedy=cfg["greedy"], split_mode=cfg["split"], train_greedy=cfg["train_greedy"],
        greedy_train_cap=cfg["greedy_train_cap"], stop_dof=cfg["stop_dof"], stage1_w_text=cfg["stage1_w_text"],
        stage1_w_var=cfg["stage1_w_var"], fast=cfg["fast"], neg_pool=cfg["neg_pool"], eval_val=cfg["eval_val"],
        equiv_ids=ex if cfg.get("equiv") else None, equiv_src=es if cfg.get("equiv") else None,
        on_trained=on_trained)
    new = {r["case_id"]: r["ranks"] for r in res["per_case"] if r.get("split", "test") == "test"}
    old = {r["case_id"]: r["ranks"] for r in ref["results"][MODE]["per_case"] if r.get("split", "test") == "test"}
    mismatch = sorted(c for c in new if new[c] != old.get(c))
    if set(new) != set(old) or mismatch:
        sys.exit(f"{setting} {seed}: 学習し直した順位が既存の出力と食い違う（{len(mismatch)} 件、例 {mismatch[:3]}）")
    out = {"setting": setting, "seed": seed, "mode": MODE, "config_from": str(ref_f.relative_to(ROOT)),
           "n_test": len(new), "rank_mismatch": 0,
           "cases": {c: {**captured[c], "ranks": new[c]} for c in new}}
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{setting}__{seed}.json"
    f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {f}: {len(new)} test cases, ranks identical to {ref_f.name}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="A", choices=["A", "B"])
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="出力先（既定 experiments/llm_db_baseline/proposed_top20）。--top を変えるときは別の場所にする")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    seeds = rl.ALL_SEEDS if args.seeds == "all" else [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        if (OUT / f"{args.setting}__{seed}.json").exists():
            print(f"skip {args.setting} {seed}（もうある）", flush=True)
            continue
        run_one(args.setting, seed, args.top)


if __name__ == "__main__":
    main()
