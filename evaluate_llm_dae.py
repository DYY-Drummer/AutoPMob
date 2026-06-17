"""DAE 難ケースでの LLM 直接生成ベースライン（図B 用）.

LLM（Claude）を「問題を解く側」として用い，各 DAE ケースの context と
入出力変数のみから必要な数式を生成させる。生成式を現行 DB（11,146 式）と
char n-gram TF-IDF で照合してランキング化し，本手法と同じ Recall@K_correct
で評価する。X（正解式数）別に層化サンプリングして難易度勾配を見る。

出力:
  experiments/llm_dae_predictions.json
  experiments/llm_dae_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env", override=True)
import anthropic

sys.path.insert(0, str(ROOT))
from set_aware_reranker import load_equations, load_cases, eq_key, norm
from evaluate_multi_eq import compute_all_ranks, case_metrics, aggregate_metrics

MODEL = "claude-opus-4-8"  # Opus 4.8（API モデル ID は日付サフィックス不要）
RATE = 0.8
PRED = ROOT / "experiments" / "llm_dae_predictions.json"
OUT = ROOT / "experiments" / "llm_dae_results.json"

PROMPT = """You are an expert in physical and chemical engineering modeling.
A modeling task is described below, together with the KNOWN (input) variables
and the UNKNOWN (output) variables to be solved for.

List the governing equations required to solve this task. Output them in LaTeX,
ONE equation per line, ordered from most to least essential, up to {n} equations.
Output ONLY the equations (LaTeX), one per line — no numbering, no prose, no $ delimiters.

SCENARIO:
{ctx}

KNOWN (input) variables: {inp}
UNKNOWN (output) variables: {out}
"""


def gen_equations(client, case, n_eqs, model=MODEL):
    ctx = case.get("context", "") or ""
    inp = case.get("input_variables", [])
    out = case.get("output_variables", [])
    prompt = PROMPT.format(n=n_eqs, ctx=ctx[:1200], inp=inp, out=out)
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=model, max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            lines = [ln.strip().lstrip("0123456789.) ").strip(" $")
                     for ln in text.splitlines()]
            eqs = [ln for ln in lines if ln and any(c in ln for c in "=+-/^_\\")]
            return eqs[:n_eqs]
        except anthropic.RateLimitError:
            time.sleep(20 * (attempt + 1))
        except Exception as e:
            print(f"  err {type(e).__name__}: {str(e)[:80]}")
            time.sleep(4)
    return []


def _match_variant(vt, wanted):
    for w in wanted:
        if w.endswith("_") and vt.startswith(w):
            return True
        if vt == w:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-x", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regen", action="store_true", help="既存予測を無視して再生成")
    ap.add_argument("--label", default="dae", help="出力ファイルのラベル")
    ap.add_argument("--variants", default="dae_",
                    help="サンプル対象の variant（カンマ区切り・末尾 _ で前方一致）")
    ap.add_argument("--n-sample", type=int, default=None,
                    help="一様サンプル件数（指定時は X 層化でなく一様抽出）")
    ap.add_argument("--model", default=MODEL,
                    help="LLM のモデル ID（既定は MODEL 定数）")
    args = ap.parse_args()
    global PRED, OUT
    PRED = ROOT / "experiments" / f"llm_{args.label}_predictions.json"
    OUT = ROOT / "experiments" / f"llm_{args.label}_results.json"
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]

    eqs = load_equations()
    cases = load_cases()
    ek = [eq_key(e) for e in eqs]
    et = [norm(e.get("equation") or "") for e in eqs]
    ki = {k: i for i, k in enumerate(ek)}
    by_id = {c.get("case_id"): c for c in cases}

    # variant フィルタでサンプリング（一様 or X 層化）
    pool_all = [c for c in cases if _match_variant(c.get("variant_type", "") or "", wanted)]
    rng = random.Random(args.seed)
    if args.n_sample:
        rng.shuffle(pool_all)
        sample = pool_all[:args.n_sample]
        print(f"{args.label} sample: {len(sample)} 件（variants={wanted}, 一様抽出）")
    else:
        byX = defaultdict(list)
        for c in pool_all:
            byX[len(c.get("correct_model_ids") or [])].append(c)
        sample = []
        for X in sorted(byX):
            p = byX[X][:]
            rng.shuffle(p)
            sample.extend(p[:args.n_per_x])
        print(f"{args.label} sample: {len(sample)} 件（X={sorted(byX)} 各 {args.n_per_x}）")

    # 予測の生成（or 既存ロード）
    if PRED.exists() and not args.regen:
        preds = json.load(open(PRED))
        print(f"既存予測をロード: {len(preds)} 件")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (.env)")
        client = anthropic.Anthropic(api_key=api_key)
        preds = []
        for i, c in enumerate(sample):
            X = len(c.get("correct_model_ids") or [])
            eqs_gen = gen_equations(client, c, max(12, X + 4), model=args.model)
            preds.append({"case_id": c["case_id"], "n_correct": X,
                          "predictions": [{"equation": e} for e in eqs_gen]})
            print(f"  [{i+1}/{len(sample)}] {c['case_id']} X={X} -> {len(eqs_gen)} eqs")
            time.sleep(RATE)
        json.dump(preds, open(PRED, "w"), ensure_ascii=False, indent=2)
        print(f"Saved predictions -> {PRED}")

    # 照合（char n-gram TF-IDF）して採点
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True, min_df=1)
    tf.fit(et)
    db = tf.transform(et)

    results = []
    byX_metric = defaultdict(list)
    for p in preds:
        c = by_id.get(p["case_id"])
        if not c:
            continue
        corr = set(ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki)
        if not corr:
            continue
        llm = [norm(e.get("equation") or "") for e in p.get("predictions", [])]
        llm = [e for e in llm if e]
        if not llm:
            continue
        sim = cosine_similarity(tf.transform(llm), db)
        ranked, seen = [], set()
        for r in range(len(llm)):
            j = int(sim[r].argmax())
            if j not in seen:
                ranked.append(j); seen.add(j)
        for j in np.argsort(-sim.max(axis=0)).tolist():
            if j not in seen:
                ranked.append(j); seen.add(j)
        ranks = compute_all_ranks(ranked, corr, miss_rank=10_000)
        cm = case_metrics(ranks)
        cm["case_id"] = p["case_id"]
        cm["n_correct"] = p.get("n_correct", len(corr))
        results.append(cm)
        byX_metric[cm["n_correct"]].append(cm.get("Recall@K_correct", cm.get("Precision@C")))

    agg = aggregate_metrics(results)
    rkc = agg.get("Recall@K_correct", agg.get("Precision@C"))
    print("\n" + "=" * 56)
    print(f"LLM 直接生成 @ DAE サンプル（n={agg['n_cases']}）")
    print(f"  Recall@K_correct = {rkc:.4f}")
    print(f"  Recall@10        = {agg.get('Recall@10', 0):.4f}")
    print(f"  MAP              = {agg.get('MAP', 0):.4f}")
    print("  --- X 別 Recall@K_correct ---")
    for X in sorted(byX_metric):
        v = byX_metric[X]
        print(f"    X={X:2d} (n={len(v):2d}): {np.mean(v):.3f}")

    json.dump({"method": "claude_direct_generation_dae", "n_cases": agg["n_cases"],
               "Recall@K_correct": rkc, "summary": agg,
               "by_n_correct": {str(k): float(np.mean(v)) for k, v in byX_metric.items()}},
              open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
