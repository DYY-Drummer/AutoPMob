"""人手の確認の結果を読み、基準ごとに数え、2 人の判定の一致を出す.

入力（どちらも列は case_id, C1, C2, C3, C4, note。標本は data/manual_check_sample.json）:
  data/manual_check_results.csv          著者がチェックシートで付けた判定（論文に書く値のもと）
  data/manual_check_results_claude.csv   同じ 50 件を言語モデル（Claude Fable 5.2）が判定したもの（照合の相手。人手の確認ではない）
出力: data/manual_check_summary.json（著者の結果があるときだけ）。使い方: python3 paper/cace/manual_check.py

2026-11-17 の本人の選択：論文には、人手の確認ではなく「言語モデルが判定した」結果として書く。論文の数値は
make_numbers._sample_check が data/manual_check_results_claude.csv から作る。著者の結果のファイルはない。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_manual_check_sheet as mc  # noqa: E402

RESULTS_AUTHOR = mc.RESULTS_CSV
RESULTS_CLAUDE = HERE / "data" / "manual_check_results_claude.csv"
SUMMARY_JSON = HERE / "data" / "manual_check_summary.json"
CRITERIA = [c[0] for c in mc.CRITERIA]
ALLOWED = {c: ("Y", "N", "NA") if c == "C3" else ("Y", "N") for c in CRITERIA}   # 「対象外」は C3 だけ


def read_results(path: Path, sample_cases: list) -> dict:
    """結果の CSV を {case_id: {C1, C2, C3, C4, note}} にする（標本の並び順）。欠け、余り、空欄、形の違う値があれば止まる."""
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != mc.RESULT_COLUMNS:
            raise ValueError(f"{path}: columns are {reader.fieldnames}, expected {mc.RESULT_COLUMNS}")
        rows = list(reader)
    out = {}
    for r in rows:
        cid = (r["case_id"] or "").strip()
        if cid in out:
            raise ValueError(f"{path}: case {cid} appears twice")
        if cid not in sample_cases:
            raise ValueError(f"{path}: case {cid} is not in the sample")
        for c in CRITERIA:
            v = (r[c] or "").strip()
            if v not in ALLOWED[c]:
                raise ValueError(f"{path}: case {cid} has {c} = {v!r}, expected one of {ALLOWED[c]}")
            r[c] = v
        out[cid] = {**{c: r[c] for c in CRITERIA}, "note": (r["note"] or "").strip()}
    missing = [c for c in sample_cases if c not in out]
    if missing:
        raise ValueError(f"{path}: {len(missing)} cases of the sample are missing: {missing[:5]}")
    return {c: out[c] for c in sample_cases}


def summarize(results: dict, kind_of: dict) -> dict:
    """基準ごとの「はい／いいえ／対象外」の件数（全体と種類別）と、4 つとも満たした件数（対象外は満たした側に数える）."""
    def counts(cases):
        return {c: {v: sum(1 for k in cases if results[k][c] == v) for v in ("Y", "N", "NA")} for c in CRITERIA}
    kinds = list(mc.KINDS)
    met = [k for k, r in results.items() if all(r[c] in ("Y", "NA") for c in CRITERIA)]
    return {"n": len(results),
            "n_by_kind": {kd: sum(1 for k in results if kind_of[k] == kd) for kd in kinds},
            "criteria": counts(results),
            "by_kind": {kd: counts([k for k in results if kind_of[k] == kd]) for kd in kinds},
            "all_met": {"n": len(met), "by_kind": {kd: sum(1 for k in met if kind_of[k] == kd) for kd in kinds}, "cases": met}}


def cohen_kappa(pairs: list):
    """2 人の判定の組 [(a, b)] の Cohen の κ。偶然の一致が 1（どちらも 1 種類の答えだけ）のときは定義できないので None."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[v] * cb[v] for v in set(ca) | set(cb)) / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(a: dict, b: dict) -> dict:
    """同じケースへの 2 人の判定を比べる：基準ごとの一致の数と κ、4 つとも一致した件数、食い違った判定の一覧."""
    if set(a) != set(b):
        raise ValueError("the two results do not cover the same cases")
    crit, diff = {}, []
    for c in CRITERIA:
        pairs = [(a[k][c], b[k][c]) for k in a]
        crit[c] = {"n": len(pairs), "agree": sum(x == y for x, y in pairs), "kappa": cohen_kappa(pairs)}
    for k in a:
        for c in CRITERIA:
            if a[k][c] != b[k][c]:
                diff.append({"case_id": k, "criterion": c, "a": a[k][c], "b": b[k][c],
                             "note_a": a[k]["note"], "note_b": b[k]["note"]})
    return {"criteria": crit,
            "cases_all_four_agree": sum(all(a[k][c] == b[k][c] for c in CRITERIA) for k in a),
            "judgments": {"n": len(a) * len(CRITERIA), "agree": sum(v["agree"] for v in crit.values())},
            "differences": diff}


def _line(s: dict) -> str:
    return "  ".join(f"{c} Y{v['Y']}/N{v['N']}" + (f"/NA{v['NA']}" if v["NA"] else "") for c, v in s["criteria"].items()) \
        + f"  all met {s['all_met']['n']} {s['all_met']['by_kind']}"


def main() -> None:
    sample = json.loads(mc.SAMPLE_JSON.read_text(encoding="utf-8"))
    cases = mc.load_cases()
    kind_of = {c: mc.so.case_kind(cases[c]) for c in sample["cases"]}
    claude = read_results(RESULTS_CLAUDE, sample["cases"])
    print("language model:", _line(summarize(claude, kind_of)))
    if not RESULTS_AUTHOR.exists():
        print(f"no results of the author yet: put the export of the check sheet at {RESULTS_AUTHOR}")
        return
    author = read_results(RESULTS_AUTHOR, sample["cases"])
    s, ag = summarize(author, kind_of), agreement(author, claude)
    print("author:        ", _line(s))
    for c, v in ag["criteria"].items():
        k = "n/a" if v["kappa"] is None else f"{v['kappa']:.2f}"
        print(f"  {c}: agree {v['agree']}/{v['n']}, kappa {k}")
    print(f"  all four agree in {ag['cases_all_four_agree']}/{len(author)} cases; "
          f"{ag['judgments']['agree']}/{ag['judgments']['n']} judgments agree")
    for d in ag["differences"]:
        print(f"  {d['case_id']} {d['criterion']}: author {d['a']} / model {d['b']}  | author: {d['note_a'][:60]} | model: {d['note_b'][:80]}")
    SUMMARY_JSON.write_text(json.dumps({"sample": {k: sample[k] for k in ("seed", "n", "population", "allocation")},
                                        "author": s, "agreement_with_language_model": ag}, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    print(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
