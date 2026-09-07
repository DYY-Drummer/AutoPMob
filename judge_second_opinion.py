#!/usr/bin/env python3
"""第 2 判定器で LLM 直接生成の等価判定をやり直す（B1: 判定の検証）.

第 1 判定器（experiments/llm_<label>_equiv_results.json, claude-opus-4-8）が判定したのと
同じケース・同じ順序・同じプロンプト（evaluate_llm_equiv.build_prompt）・同じ生成式リスト
（上位 12 件）で、別モデル（既定 claude-fable-5-1）に判定させ、同じ形式で
experiments/llm_<label>_equiv_results_<tag>.json に保存する。

方針:
  - fallbacks は使わない（拒否時に第 1 判定器へ落ちると「第 2 判定器」でなくなる）
  - stop_reason == "refusal" / 本文空 / JSON 不正 は再試行し、解消しなければ欠測
    （match_index = null）として記録。一致度分析（analyze_judge_agreement.py）は欠測を除外する
  - 思考は adaptive（Fable 5.1 は常時オン）。effort は --effort 指定時のみ output_config で渡す
  - ケースごとに JSON を書き出す（途中で落ちても部分結果が残る）
仕様: docs/superpowers/specs/2026-09-07-b1-second-judge-design.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
import anthropic  # noqa: E402

sys.path.insert(0, str(ROOT))
from set_aware_reranker import load_equations, load_cases, eq_key, norm  # noqa: E402
from evaluate_llm_equiv import build_prompt  # noqa: E402

EXP = ROOT / "experiments"
DEFAULT_MODEL = "claude-fable-5-1"
N_GEN = 12  # evaluate_llm_equiv.main と同じ: 生成式は上位 12 件
PATTERN = re.compile(r'\{[^{}]*"match_index"\s*:\s*\[[^\]]*\][^{}]*\}', re.S)


def default_tag(model: str) -> str:
    return "fable51" if model == DEFAULT_MODEL else re.sub(r"[^a-z0-9]+", "", model.lower())


def parse_match_index(text: str, k: int):
    m = PATTERN.search(text)
    if not m:
        return None
    arr = [int(x) for x in json.loads(m.group(0))["match_index"]]
    return (arr + [0] * k)[:k]


def judge_once(client, model, prompt, max_tokens, effort):
    kwargs = dict(model=model, max_tokens=max_tokens, thinking={"type": "adaptive"},
                  messages=[{"role": "user", "content": prompt}])
    if effort:
        kwargs["output_config"] = {"effort": effort}
    resp = client.messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return resp.stop_reason, resp.model, text, usage


def judge(client, model, prompt, k, max_tokens, effort, attempts=4, verbose=False):
    """判定を最大 attempts 回試み、(match_index or None, 試行ログ) を返す."""
    log = []
    for attempt in range(attempts):
        try:
            stop, served, text, usage = judge_once(client, model, prompt, max_tokens, effort)
        except anthropic.RateLimitError:
            time.sleep(20 * (attempt + 1))
            continue
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except anthropic.APIConnectionError:
            time.sleep(5 * (attempt + 1))
            continue
        entry = {"stop_reason": stop, "model": served, **usage}
        log.append(entry)
        if verbose:
            print(f"    judge raw ({stop}, {served}): {text[:100]}", flush=True)
        if stop == "refusal":
            continue
        idx = parse_match_index(text, k)
        if idx is not None:
            return idx, log
    return None, log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="set_A / set_B / set_full")
    ap.add_argument("--judge-model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default=None, help="出力ファイルの接尾辞（既定: モデルから決める）")
    ap.add_argument("--limit", type=int, default=None, help="先頭 N ケースだけ（smoke 用、接尾辞 _smoke）")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--effort", default=None, choices=[None, "low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    tag = args.tag or default_tag(args.judge_model)

    ref = json.load(open(EXP / f"llm_{args.label}_equiv_results.json"))
    preds = {p["case_id"]: p for p in json.load(open(EXP / f"llm_{args.label}_predictions.json"))}
    eqs, cases = load_equations(), load_cases()
    eqtext = {eq_key(e): (e.get("equation") or "") for e in eqs}
    by_id = {c.get("case_id"): c for c in cases}
    ref_cases = ref["per_case"][: args.limit] if args.limit else ref["per_case"]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    suffix = f"_{tag}" + ("_smoke" if args.limit else "")
    out_path = EXP / f"llm_{args.label}_equiv_results{suffix}.json"
    percase, rk_list, cov_list, byX = [], [], [], defaultdict(list)
    usage_total = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    n_failed = 0
    t0 = time.time()
    print(f"[{args.label}] second judge {args.judge_model} (tag={tag}) on {len(ref_cases)} cases "
          f"judged by {ref.get('judge_model')}", flush=True)

    def dump():
        out = {"method": "claude_direct_generation_equiv", "judge_model": args.judge_model,
               "reference_judge_model": ref.get("judge_model"),
               "reference_file": f"llm_{args.label}_equiv_results.json",
               "n_cases": len(rk_list), "n_failed": n_failed,
               "Recall@K_correct": float(np.mean(rk_list)) if rk_list else 0.0,
               "coverage": float(np.mean(cov_list)) if cov_list else 0.0,
               "by_n_correct": {str(k): float(np.mean(v)) for k, v in byX.items()},
               "config": {"thinking": "adaptive", "effort": args.effort or "default",
                          "max_tokens": args.max_tokens, "fallbacks": None, "n_gen": N_GEN,
                          "prompt": "evaluate_llm_equiv.JUDGE (identical to the first judge)"},
               "usage_total": usage_total, "elapsed_sec": round(time.time() - t0, 1),
               "per_case": percase}
        json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=2)
        return out

    for i, rc in enumerate(ref_cases):
        c, p = by_id.get(rc["case_id"]), preds.get(rc["case_id"])
        if c is None or p is None:
            raise RuntimeError(f"{rc['case_id']}: case or prediction not found")
        correct = [eqtext.get(norm(m), "") for m in (c.get("correct_model_ids") or [])]
        correct = [e for e in correct if e]
        k = len(correct)
        if k != int(rc["n_correct"]):
            raise RuntimeError(f"{rc['case_id']}: n_correct {k} != reference {rc['n_correct']}")
        gen = [g.get("equation", "") for g in p.get("predictions", [])][:N_GEN]
        prompt = build_prompt(c.get("input_variables"), c.get("output_variables"), correct, gen, k)
        idx, log = judge(client, args.judge_model, prompt, k, args.max_tokens, args.effort,
                         verbose=args.verbose)
        for e in log:
            usage_total["input_tokens"] += e["input_tokens"]
            usage_total["output_tokens"] += e["output_tokens"]
            usage_total["calls"] += 1
        rec = {"case_id": rc["case_id"], "n_correct": k, "variant": c.get("variant_type"),
               "match_index": idx, "match_index_ref": rc["match_index"],
               "attempts": log}
        if idx is None:
            n_failed += 1
            rec["Recall@K_equiv"] = None
            rec["coverage_equiv"] = None
        else:
            rk = sum(1 for g in idx if 1 <= g <= k) / k
            cov = sum(1 for g in idx if g >= 1) / k
            rk_list.append(rk); cov_list.append(cov); byX[k].append(rk)
            rec["Recall@K_equiv"] = rk
            rec["coverage_equiv"] = cov
        percase.append(rec)
        dump()
        same = "-" if idx is None else str(sum(1 for a, b in zip(idx, rc["match_index"]) if a == b)) + f"/{k}"
        print(f"  [{i + 1}/{len(ref_cases)}] {rc['case_id']:16s} K={k:2d} new={idx} ref={rc['match_index']} "
              f"same={same}  R@K so far={np.mean(rk_list) if rk_list else 0:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        time.sleep(args.sleep)

    out = dump()
    print(f"\n{args.label}: judge={args.judge_model}  Recall@K={out['Recall@K_correct']:.4f}  "
          f"coverage={out['coverage']:.4f}  n={out['n_cases']} failed={n_failed}  "
          f"tokens in/out={usage_total['input_tokens']}/{usage_total['output_tokens']}  "
          f"calls={usage_total['calls']}  {out['elapsed_sec']}s\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
