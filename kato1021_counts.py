"""報告書の本文に出てくるデータの数を、外すケースの一覧ごとに数える（2026-10-21）.

2026-10-21：出典横断の 113 件の説明文を書き直した training_cases.json（e995ca0）と、外す一覧 v3（92 件＝v1 の 91 件＋
2 冊の教科書の式から成る単一出典の core_019_v1、b7f7cd2）。報告書 db21007 の値（v1、--check-v1）と並べて記録する。
分割は、c7c4feb の実行（experiments/kato1020/raw/select の per-case に記録された検証・テストのケース ID）とも比べる。

数えるもの（報告書 db21007 の値は 91 件を外した 1,732 件のもの）:
  - 族ごとの件数と、表 2 の行（件数、正解式の数の平均と最大、正解集合が含む出典の数の平均）
  - 要件 1〜3 を満たすケースと、要件 3 が自明に満たされる（入出力変数以外の変数がない）ケース（旧 1,725 件）
  - 2 本以上の正解式に現れる記号の定義の文章が式によって違うケース（旧 1,300 件＝定義の文章を完全一致で比べた数。
    大文字小文字と空白をそろえて比べると 1,293 件で、docs/appendix_materials/gt_same_symbol_different_definitions.csv はこちら）
  - LLM（Claude Opus 5.5）が、入出力変数の記号が式によって別の量を指すと判定したケース（旧 596 件。
    experiments/var_descriptions.json の meaning_conflict から数える。新しく LLM は呼ばない）
  - 設定 A の正解式の数の平均、正解式が 2 本以上のケースの割合
  - 正解式（異なる式）の数と、そのうち同じ文字列の式が別の出典にもある本数（旧 386/3,132 本）
  - 出典横断のうちプログラムで作った 496 件以外で、どの 2 本の正解式も変数記号を共有しないケース（旧 33 件中 24 件、111 件中 81 件）
出力: <out>/data_counts.json
使い方: python kato1021_counts.py --exclude experiments/kato1020/excluded_cases_v3.json --out experiments/kato1021
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent


def family(v):
    return ("単一出典" if v == "original" else "出典横断" if v.startswith("multisource")
            else "合成" if v.startswith("dae_") else None)


def counts(exclude_path):
    eqs = json.load(open(ROOT / "unified_equations.json", encoding="utf-8"))
    V = {f"{e['source_id']}__{e['eq_id']}": set(e["variables"].keys()) for e in eqs}
    SRC = {f"{e['source_id']}__{e['eq_id']}": e["source_id"] for e in eqs}
    ex = set(json.load(open(exclude_path, encoding="utf-8"))["excluded_case_ids"]) if exclude_path else set()
    P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s  # noqa: E731
    cases = [c for c in json.load(open(ROOT / "training_cases.json", encoding="utf-8"))
             if family(c["variant_type"]) and c["case_id"] not in ex]
    fam = defaultdict(list)
    req, trivial3 = Counter(), 0
    for c in cases:
        I = set(P(c["input_variables"])); O = set(P(c["output_variables"])); R = P(c["correct_model_ids"])
        allv = set().union(*[V[r] for r in R])
        occ = Counter(v for r in R for v in V[r])
        inter = allv - (I | O)
        req["req1"] += (I | O) <= allv
        req["req2"] += len(allv - I) == len(R)
        req["req3"] += all(occ[v] >= 2 for v in inter)
        trivial3 += not inter
        fam[family(c["variant_type"])].append((len(R), len({SRC[r] for r in R})))
    table2 = {f: {"n": len(v), "mean_K": round(sum(k for k, _ in v) / len(v), 2), "max_K": max(k for k, _ in v),
                  "mean_sources": round(sum(s for _, s in v) / len(v), 2)} for f, v in fam.items()}
    ids = {c["case_id"] for c in cases}
    DEF = {f"{e['source_id']}__{e['eq_id']}": e["variables"] for e in eqs}
    n_exact = n_norm = 0
    for c in cases:
        defs = defaultdict(list)
        for r in P(c["correct_model_ids"]):
            for v, d in DEF[r].items():
                defs[v].append(str(d))
        multi = [ds for ds in defs.values() if len(ds) >= 2]
        n_exact += any(len(set(ds)) > 1 for ds in multi)
        n_norm += any(len({" ".join(x.lower().split()) for x in ds}) > 1 for ds in multi)
    with open(ROOT / "docs" / "appendix_materials" / "gt_same_symbol_different_definitions.csv", encoding="utf-8-sig") as f:
        def_cases = {row["case_id"] for row in csv.DictReader(f)} & ids
    assert len(def_cases) == n_norm, (len(def_cases), n_norm)   # CSV（そろえて比べた表）と一致することの確認
    vd = json.load(open(ROOT / "experiments" / "var_descriptions.json", encoding="utf-8"))["cases"]
    conflict = {cid for cid, v in vd.items() if cid in ids and any(x.get("meaning_conflict") for x in v["variables"].values())}
    Ks = [k for v in fam.values() for k, _ in v]
    # 単一出典（original）の内訳（報告書 §3.2 の「92 件中 67 件は 1 式、61 件は基礎式集の式だけ」「92 件のうち 91 件は 1 つの出典」）。
    # 基礎式集は、LLM が分野ごとに生成した式集の 32 出典（source_id が handbook_ で始まる、670 式）
    singles = [P(c["correct_model_ids"]) for c in cases if c["variant_type"] == "original"]
    single = {"n": len(singles), "K1": sum(len(R) == 1 for R in singles),
              "handbook_only": sum(all(SRC[r].startswith("handbook_") for r in R) for R in singles),
              "one_source": sum(len({SRC[r] for r in R}) == 1 for R in singles)}
    # 出典横断のうちプログラムで作った 496 件（multisource_v3）以外で、どの 2 本の正解式も変数記号を 1 つも共有しないケース
    # （報告書 §3.2 の「144 件のうち 105 件（33 件のうち 24 件、111 件のうち 81 件）」）
    no_share = Counter()
    for c in cases:
        vt = c["variant_type"]
        if vt.startswith("multisource") and not vt.startswith("multisource_v3"):
            R = P(c["correct_model_ids"])
            no_share[vt] += all(not (V[a] & V[b]) for i, a in enumerate(R) for b in R[i + 1:])
    # 正解式（異なる式）のうち、同じ文字列の式が別の出典にもあるもの（報告書 §3.5・§8.2 の 386/3,132 本。
    # analyze_kato_0831_followup.gt_checks と同じ定義で、文字列は set_aware_reranker.latex_key でそろえる）
    from set_aware_reranker import latex_key
    LX = {f"{e['source_id']}__{e['eq_id']}": (latex_key(e["equation"]), e["source_id"]) for e in eqs}
    by_lx = defaultdict(list)
    for k, (s, _) in LX.items():
        by_lx[s].append(k)
    gt_all = set().union(*[set(P(c["correct_model_ids"])) for c in cases])
    dup = [g for g in gt_all if any(LX[o][1] != LX[g][1] for o in by_lx[LX[g][0]] if o != g)]
    return {"exclude": str(exclude_path), "n_cases": len(cases), "by_variant": dict(Counter(c["variant_type"].split("_X")[0] for c in cases)),
            "gt_distinct_equations": len(gt_all), "gt_identical_latex_in_other_source": len(dup),
            "cross_source_no_shared_symbol_by_variant": dict(no_share),
            "single_source_breakdown": single,
            "table2": table2, "requirements_1_3": dict(req), "req3_trivial_no_intermediate": trivial3,
            "req3_nontrivial": len(cases) - trivial3,
            "same_symbol_different_definition_cases": n_exact,
            "same_symbol_different_definition_cases_normalized": n_norm,
            "llm_meaning_conflict_cases": len(conflict),
            "setting_A_mean_K": round(sum(Ks) / len(Ks), 2),
            "setting_A_share_K_ge_2": round(sum(1 for k in Ks if k >= 2) / len(Ks), 3)}


def split_ids(exclude_path, setting):
    """10 通りの分割の訓練・検証・テストのケース ID（set_aware_reranker の層化分割と同じ。analyze_stage1_n1.split_for）."""
    from analyze_stage1_n1 import SEEDS, load_setting_A, split_for
    D = load_setting_A(exclude_path)
    cases, ek, cl, cs = D["cases"], D["ek"], D["cl"], D["cs"]
    idx = (list(range(len(cases))) if setting == "A"
           else [i for i, c in enumerate(cases) if str(c.get("variant_type", "")).startswith("dae_")])
    sub = [cases[i] for i in idx]
    out = {}
    for seed in SEEDS:
        tr, trv, te = split_for(sub, [cl[i] for i in idx], ek, [cs[i] for i in idx], seed)
        ids = lambda I: sorted(sub[i]["case_id"] for i in I)  # noqa: E731
        out[str(seed)] = {"train": ids(tr), "val": ids(sorted(set(trv) - set(tr))), "test": ids(te)}
    return out


def context_changes(exclude_path, rev="db21007"):
    """報告書 db21007 の時点の training_cases.json と比べて、変わったケースとフィールド（2026-10-21 の説明文の書き直しの確認）."""
    import subprocess
    old = json.loads(subprocess.run(["git", "show", f"{rev}:training_cases.json"], cwd=ROOT, capture_output=True,
                                    text=True, check=True).stdout)
    new = json.load(open(ROOT / "training_cases.json", encoding="utf-8"))
    assert [c["case_id"] for c in old] == [c["case_id"] for c in new]
    ex = set(json.load(open(exclude_path, encoding="utf-8"))["excluded_case_ids"])
    ch = [(a, b) for a, b in zip(old, new) if a != b]
    return {"rev": rev, "n_changed": len(ch),
            "fields": dict(Counter(k for a, b in ch for k in set(a) | set(b) if a.get(k) != b.get(k))),
            "by_variant": dict(Counter(b["variant_type"] for _, b in ch)),
            "case_ids_not_random_io": sorted(b["case_id"] for _, b in ch if b["variant_type"] != "multisource_random_io"),
            "n_changed_excluded": sum(b["case_id"] in ex for _, b in ch),
            "n_changed_setting_B": sum(b["variant_type"].startswith("dae_") for _, b in ch)}


def splits_from_runs(raw_dir, setting):
    """実行の per-case に記録された分割（種ごとの検証・テストのケース ID）。重み (0.3, 0.7)・hard negative の select の出力から読む."""
    import glob
    out = {}
    for f in glob.glob(str(Path(raw_dir) / "select" / f"{setting}_w30-70_hard__*.json")):
        r = json.load(open(f, encoding="utf-8"))["results"]
        pc = r[list(r)[0]]["per_case"]
        out[Path(f).stem.split("__")[1]] = {k: sorted(x["case_id"] for x in pc if x.get("split") == k) for k in ("val", "test")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-v1", default=str(ROOT / "experiments" / "kato1020" / "excluded_cases.json"),
                    help="旧（91 件を外した）値の再現の確認に使う一覧")
    a = ap.parse_args()
    old = counts(a.check_v1)
    new = counts(a.exclude)
    doc = {"note": "報告書の本文に出てくるデータの数。v1 は 91 件を外した 1,732 件（報告書 db21007 の値の確認）。",
           "v1_check": old, "new": new, "context_changes_vs_db21007": context_changes(a.exclude)}
    cc = doc["context_changes_vs_db21007"]
    print(f"db21007 から変わったケース {cc['n_changed']} 件、フィールド {cc['fields']}、{cc['by_variant']}、"
          f"除外の中 {cc['n_changed_excluded']}、設定 B {cc['n_changed_setting_B']}")
    # 設定 B（合成のみ）の分割が、外す一覧を変えても同じか（ケース ID で比べる）と、設定 A の分割ごとのケース数
    sB_old, sB_new = split_ids(a.check_v1, "B"), split_ids(a.exclude, "B")
    doc["setting_B_splits_identical"] = sB_old == sB_new
    doc["split_sizes"] = {nm: {s: {k: len(v) for k, v in d.items()} for s, d in sp.items()}
                          for nm, sp in (("A_v1", split_ids(a.check_v1, "A")), ("A_new", split_ids(a.exclude, "A")), ("B", sB_new))}
    # c7c4feb の実行に記録された分割（検証・テストのケース ID）と、いまのデータで作る分割を比べる
    prev = ROOT / "experiments" / "kato1020" / "raw"
    if (prev / "select").exists():
        sA_new = split_ids(a.exclude, "A")
        ex_now = set(json.load(open(a.exclude, encoding="utf-8"))["excluded_case_ids"])

        def same(run, now):
            """c7c4feb の実行の検証・テストから、いま外すケースを除くと、いまの分割と同じか."""
            return len(run) == 10 and all({k: [c for c in run[s][k] if c not in ex_now] for k in ("val", "test")}
                                          == {k: now[s][k] for k in ("val", "test")} for s in run)
        runA = splits_from_runs(prev, "A")
        doc["splits_match_kato1020_runs"] = {"A": same(runA, sA_new), "B": same(splits_from_runs(prev, "B"), sB_new)}
        newly = sorted(ex_now - set(json.load(open(a.check_v1, encoding="utf-8"))["excluded_case_ids"]))
        doc["newly_excluded_in_kato1020_runs"] = {
            cid: {s: ("test" if cid in runA[s]["test"] else "val" if cid in runA[s]["val"] else "train") for s in sorted(runA, key=int)}
            for cid in newly}
        print(f"c7c4feb の実行の分割（検証・テストのケース ID。いま外すケースを除く）と一致：{doc['splits_match_kato1020_runs']}")
        for cid, d in doc["newly_excluded_in_kato1020_runs"].items():
            print(f"  新しく外した {cid} の c7c4feb での位置：{d}")
    print(f"設定 B の分割が一致：{doc['setting_B_splits_identical']}")
    for nm in ("A_v1", "A_new"):
        z = doc["split_sizes"][nm].values()
        print(f"  {nm}: テスト {min(x['test'] for x in z)}〜{max(x['test'] for x in z)}、検証 {min(x['val'] for x in z)}〜{max(x['val'] for x in z)}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(out / "data_counts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {out / 'data_counts.json'}")
    for nm, d in (("v1（1,732 件の確認）", old), ("新", new)):
        print(f"[{nm}] n={d['n_cases']} 要件3自明 {d['req3_trivial_no_intermediate']} 定義が違う {d['same_symbol_different_definition_cases']} "
              f"LLM で別の量 {d['llm_meaning_conflict_cases']} 平均K {d['setting_A_mean_K']} K≥2 {d['setting_A_share_K_ge_2']} 要件 {d['requirements_1_3']}")
        for f, t in d["table2"].items():
            print(f"    {f}: {t}")


if __name__ == "__main__":
    main()
