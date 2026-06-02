"""LLM 直接生成ベースライン評価（マルチソース 20 件サブセット）.

evaluate_llm_baseline.py のマルチソース版。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from two_stage_query_conditioned import load_equations, load_cases, eq_key, norm
from evaluate_multi_eq import compute_all_ranks, case_metrics, aggregate_metrics


ROOT = Path(__file__).parent
PRED_JSON = ROOT / "experiments" / "llm_baseline_multisource_predictions.json"
OUT_JSON = ROOT / "experiments" / "llm_baseline_multisource_results.json"


def main():
    eqs = load_equations()
    cases = load_cases()
    ek = [eq_key(e) for e in eqs]
    et_eq_only = [norm(e.get("equation") or "") for e in eqs]
    ek_to_idx = {k: i for i, k in enumerate(ek)}
    case_by_id = {c.get("case_id"): c for c in cases}

    with open(PRED_JSON, encoding="utf-8") as f:
        preds = json.load(f)
    print(f"Loaded {len(preds)} multisource predictions")

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                            lowercase=True, min_df=1)
    tfidf.fit(et_eq_only)
    db_vecs = tfidf.transform(et_eq_only)

    case_results = []
    detailed = []
    for pred in preds:
        case_id = pred["case_id"]
        if case_id not in case_by_id:
            print(f"WARN: case_id {case_id} not in training_cases.json")
            continue
        case = case_by_id[case_id]
        correct_keys = set(norm(m) for m in (case.get("correct_model_ids") or []))
        correct_indices = set(ek_to_idx[k] for k in correct_keys if k in ek_to_idx)
        if not correct_indices:
            continue

        llm_eqs = [norm(p.get("equation") or "") for p in pred.get("predictions", [])]
        llm_eqs = [e for e in llm_eqs if e]
        if not llm_eqs:
            continue

        llm_vecs = tfidf.transform(llm_eqs)
        sim = cosine_similarity(llm_vecs, db_vecs)

        ranked_db = []
        seen = set()
        for i in range(len(llm_eqs)):
            top_idx = int(sim[i].argmax())
            if top_idx not in seen:
                ranked_db.append(top_idx)
                seen.add(top_idx)
        max_sim = sim.max(axis=0)
        remaining_order = np.argsort(-max_sim).tolist()
        for idx in remaining_order:
            if idx not in seen:
                ranked_db.append(idx)
                seen.add(idx)

        ranks = compute_all_ranks(ranked_db, correct_indices, miss_rank=10_000)
        cm = case_metrics(ranks)
        cm["case_id"] = case_id
        cm["variant"] = case.get("variant_type", "?")
        cm["n_correct"] = len(correct_keys)
        case_results.append(cm)

        top5_db = ranked_db[:5]
        top5_keys = [ek[i] for i in top5_db]
        detailed.append({
            "case_id": case_id,
            "variant": case.get("variant_type", "?"),
            "n_correct": len(correct_keys),
            "correct_keys": list(correct_keys),
            "top5_matched": top5_keys,
            "MRR_first": cm["MRR_first"],
            "Recall@3": cm["Recall"][3],
        })

    agg = aggregate_metrics(case_results)
    print()
    print("=" * 60)
    print("LLM 直接生成ベースライン（マルチソース 20 件）")
    print("=" * 60)
    print(f"  n_cases     = {agg['n_cases']}")
    print(f"  MRR_first   = {agg['MRR_first']:.4f}")
    print(f"  Recall@3    = {agg['Recall@3']:.4f}")
    print(f"  Recall@5    = {agg.get('Recall@5', 0):.4f}")
    print(f"  Recall@10   = {agg['Recall@10']:.4f}")
    print(f"  MAP         = {agg['MAP']:.4f}")
    print(f"  FullR@3     = {agg['FullRecall@3']:.4f}")

    # 正解数別
    print()
    print("=== 正解数別 ===")
    for n in sorted(set(cm["n_correct"] for cm in case_results)):
        sub = [cm for cm in case_results if cm["n_correct"] == n]
        a = aggregate_metrics(sub)
        print(f"  正解 {n} 個 ({len(sub)} 件): MRR_f={a['MRR_first']:.3f}, R@3={a['Recall@3']:.3f}, MAP={a['MAP']:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "method": "claude_direct_generation_multisource",
            "n_cases": agg["n_cases"],
            "summary": agg,
            "detailed": detailed,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
