"""LLM 直接生成の「公平な」採点：char n-gram の表層一致ではなく、
正解 DB 式と LLM 生成式の【等価性】を審査者 LLM で判定して測り直す。

既存の予測 experiments/llm_<label>_predictions.json を再利用（再生成しない）。
審査者(Opus 4.8)が、各正解式に等価な生成式の番号(G#)を返す（0=なし）。
  - Recall@K（top-K）  : 等価式が LLM の上位 K 件に入っていた割合（検索の Recall@K と同条件）
  - coverage          : 等価式が生成リストのどこかにあった割合（順位を問わない・生成能力の上限）

出力: experiments/llm_<label>_equiv_results.json
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

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=True)
import anthropic

sys.path.insert(0, str(ROOT))
from set_aware_reranker import load_equations, load_cases, eq_key, norm

JUDGE_MODEL = "claude-opus-4-8"  # 審査者（生成手法の Sonnet 4.5 とは別の中立評価器）
EXP = ROOT / "experiments"

JUDGE = """You are a chemical/physical engineering expert. Judge whether an LLM produced the correct governing equations for a modeling task.

KNOWN (input) variables: {inp}
UNKNOWN (output) variables: {out}

CORRECT equations (ground truth), numbered:
{correct}

The LLM's GENERATED equations, numbered:
{gen}

For EACH correct equation C1..C{k} (in order), report the number of the GENERATED equation that is EQUIVALENT to it, or 0 if none is equivalent.
EQUIVALENT = expresses the SAME physical/mathematical relationship between the SAME quantities. Allow:
  - algebraic rearrangement  (e.g. "V dC/dt = F(C0 - C)" ≡ "dC/dt = (F/V)(C0 - C)")
  - partial vs total derivative (∂ vs d) for the same variable
  - different but standard notation for the SAME physical quantity
NOT equivalent =
  - a different physical relationship or different quantities
  - a governing/balance equation vs its explicit steady-state solution (these are DIFFERENT equations)
  - a generic placeholder (e.g. "r = k C^n") where the correct equation is a specific different expression
  - a specific empirical correlation (particular constants / functional form) vs a different form

Output ONLY a JSON object, no prose, no code fences:
{{"match_index": [g_for_C1, g_for_C2, ...]}}
with exactly {k} integers (each is a generated equation number, or 0)."""


def judge(client, model, inp, out, correct, gen, k, verbose=False):
    cstr = "\n".join(f"  C{i+1}: {e}" for i, e in enumerate(correct))
    gstr = "\n".join(f"  G{i+1}: {e}" for i, e in enumerate(gen)) or "  (none)"
    prompt = JUDGE.format(inp=inp, out=out, correct=cstr, gen=gstr, k=k)
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=model, max_tokens=6000,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if not text:  # thinking 消費で本文が空 → 再試行
                continue
            if verbose:
                print("    judge raw:", text[:120])
            m = re.search(r'\{[^{}]*"match_index"\s*:\s*\[[^\]]*\][^{}]*\}', text, re.S)
            arr = [int(x) for x in json.loads(m.group(0))["match_index"]]
            arr = (arr + [0] * k)[:k]
            return arr
        except anthropic.RateLimitError:
            time.sleep(20 * (attempt + 1))
        except Exception as e:
            print(f"  judge err {type(e).__name__}: {str(e)[:90]}")
            time.sleep(4)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    eqs = load_equations(); cases = load_cases()
    eqtext = {eq_key(e): (e.get("equation") or "") for e in eqs}
    by_id = {c.get("case_id"): c for c in cases}
    preds = json.load(open(EXP / f"llm_{args.label}_predictions.json"))
    if args.limit:
        preds = preds[:args.limit]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    rk_list, cov_list, byX, percase = [], [], defaultdict(list), []
    for i, p in enumerate(preds):
        c = by_id.get(p["case_id"])
        if not c:
            continue
        correct = [eqtext.get(norm(m), "") for m in (c.get("correct_model_ids") or [])]
        correct = [e for e in correct if e]
        K = len(correct)
        if K == 0:
            continue
        gen = [g.get("equation", "") for g in p.get("predictions", [])][:12]
        idx = judge(client, args.judge_model, c.get("input_variables"),
                    c.get("output_variables"), correct, gen, K, args.verbose)
        if idx is None:
            continue
        rk = sum(1 for g in idx if 1 <= g <= K) / K        # 上位K件
        cov = sum(1 for g in idx if g >= 1) / K             # 生成リスト全体
        rk_list.append(rk); cov_list.append(cov); byX[K].append(rk)
        percase.append({"case_id": p["case_id"], "n_correct": K, "variant": c.get("variant_type"),
                        "match_index": idx, "Recall@K_equiv": rk, "coverage_equiv": cov})
        if args.verbose:
            print(f"  {p['case_id']} (var={c.get('variant_type')}, K={K}) idx={idx} R@K={rk:.2f} cov={cov:.2f}")
        elif (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(preds)}] R@K={np.mean(rk_list):.3f} cov={np.mean(cov_list):.3f}")
        time.sleep(0.3)

    out = {"method": "claude_direct_generation_equiv", "judge_model": args.judge_model,
           "n_cases": len(rk_list),
           "Recall@K_correct": float(np.mean(rk_list)) if rk_list else 0.0,
           "coverage": float(np.mean(cov_list)) if cov_list else 0.0,
           "by_n_correct": {str(k): float(np.mean(v)) for k, v in byX.items()},
           "per_case": percase}
    suffix = "_smoke" if args.limit else ""
    json.dump(out, open(EXP / f"llm_{args.label}_equiv_results{suffix}.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"\n{args.label}: 等価採点  Recall@K={out['Recall@K_correct']:.4f}  "
          f"coverage={out['coverage']:.4f}  (n={out['n_cases']}, judge={args.judge_model})")


if __name__ == "__main__":
    main()
