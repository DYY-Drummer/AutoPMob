"""2026-10-20（加藤先生の 2 回目のコメントと本人の決定）：未知の変数が決まらないケースを評価と訓練から外す.

先生のコメント（報告書 §2.1 の 91 件の記述に）：「これはそのままで問題ないか？モデルと呼べるのか？」。
本人の決定：入力変数を与えても一部の未知の変数が決まらない出典横断の 91 件を、評価からも訓練からも外す。

判定は analyze_kato_0831_followup.gt_checks（報告書 §2.1 の 91 件の出典）と同じ：
  正解式 R と未知の変数 U = V(R) − 入力変数 の二部グラフ（式 r と、r に現れる未知の変数を結ぶ）で、
  式の数 = 未知の変数の数 かつ 完全マッチングがあれば「未知の変数のそれぞれに、それを決める式を 1 本ずつ割り当てられる」。
  変数は unified_equations.json の variables のキー、入出力は training_cases.json（文字列なら literal_eval）。
あわせて「どれかの正解式の変数がすべて入力変数になっている」（報告書の説明）も数え、両者の関係を記録する。

出力: experiments/kato1020/excluded_cases.json（case_id の一覧、理由、判定の定義、数の確認）
"""
from __future__ import annotations

import ast
import collections
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).parent
OUT = ROOT / "experiments" / "kato1020" / "excluded_cases.json"


def solvable(eq_var_sets, inputs) -> bool:
    """式の数 = 未知の変数の数 で、式と未知の変数の完全マッチングがあるか（gt_checks と同じ）."""
    allv = set().union(*eq_var_sets) if eq_var_sets else set()
    unk = sorted(allv - set(inputs))
    if len(unk) != len(eq_var_sets):
        return False
    idx = {u: i for i, u in enumerate(unk)}
    rows, cols = [], []
    for i, vs in enumerate(eq_var_sets):
        for v in vs - set(inputs):
            rows.append(i); cols.append(idx[v])
    if not rows:
        return False
    m = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(eq_var_sets), len(unk)))
    return int((maximum_bipartite_matching(m, perm_type="column") >= 0).sum()) == len(eq_var_sets)


def all_vars_are_inputs(eq_var_sets, inputs) -> bool:
    """どれかの正解式の変数がすべて入力変数か（その式が決める未知の変数がない）."""
    return any(vs <= set(inputs) for vs in eq_var_sets)


def family(v: str):
    return ("single_source" if v == "original" else "cross_source" if v.startswith("multisource")
            else "synthesized" if v.startswith("dae_") else None)


def main():
    eqs = json.load(open(ROOT / "unified_equations.json", encoding="utf-8"))
    V = {f"{e['source_id']}__{e['eq_id']}": set(e["variables"].keys()) for e in eqs}
    cases = json.load(open(ROOT / "training_cases.json", encoding="utf-8"))
    P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s  # noqa: E731
    out_cases, counts = [], collections.Counter()
    for c in cases:
        fam = family(c["variant_type"])
        if fam is None:
            continue
        I = set(P(c["input_variables"])); R = P(c["correct_model_ids"])
        sets = [V[r] for r in R]
        ok = solvable(sets, I)
        allin = all_vars_are_inputs(sets, I)
        counts[(fam, ok, allin)] += 1
        if not ok:
            out_cases.append({"case_id": c["case_id"], "variant_type": c["variant_type"], "family": fam,
                              "n_correct": len(R), "some_equation_all_inputs": allin})
    by_fam = collections.Counter(x["family"] for x in out_cases)
    doc = {
        "description": "未知の変数のそれぞれに、それを決める式を 1 本ずつ割り当てられない（完全マッチングがない）ケース。評価と訓練から外す（2026-10-20 本人の決定）。",
        "criterion": "analyze_kato_0831_followup.gt_checks と同じ：|V(R) − 入力| = |R| かつ 式と未知の変数の完全マッチングがある、を満たさない",
        "n_excluded": len(out_cases), "by_family": dict(by_fam),
        "n_with_some_equation_all_inputs": sum(1 for x in out_cases if x["some_equation_all_inputs"]),
        "n_derived_io_variants": sum(1 for x in out_cases if x["variant_type"] == "multisource_random_io"),
        "counts_by_family_solvable_allinputs": {f"{f}|solvable={s}|some_eq_all_inputs={a}": n
                                                for (f, s, a), n in sorted(counts.items())},
        "setting_A_after": sum(n for (f, s, a), n in counts.items()) - len(out_cases),
        "excluded_case_ids": [x["case_id"] for x in out_cases],
        "cases": out_cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {OUT}")
    print(f"外すケース {doc['n_excluded']} 件 {dict(by_fam)}、うち正解式の変数がすべて入力の式を含む {doc['n_with_some_equation_all_inputs']} 件、"
          f"入出力を選び直した派生 {doc['n_derived_io_variants']} 件。設定 A は {doc['setting_A_after']} 件になる。")
    for k, v in doc["counts_by_family_solvable_allinputs"].items():
        print("  ", k, v)


if __name__ == "__main__":
    main()
