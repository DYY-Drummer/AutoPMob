"""2026-10-20 の再実験の実行（91 件を外し、定数を検証用のケースで選ぶ）.

段階:
  select : 提案手法（一括の並べ直し）を、第 1 段の重み 4 通り × 不正解の選び方 2 通り × 設定 A・B × 分割 10 通りで訓練し、
           検証用のケースとテストのケースを評価する（--eval-val --equiv）。N1 は重みと分割ごとに n1_rule.json の値。
  final  : selected.json（kato1020_select.py）の設定で、逐次選択（推論のみ・訓練も逐次）、§5・§8 のアブレーション、
           PFI、分野ラベルの合併、誤りの位置（種 42）、話題の特徴量の AUC（種 42 の設定）を実行する。
  sweep  : 付録用に、選んだ重みと不正解の選び方のまま N1 = 50, 200, 500, 1,000, 2,000 で一括の並べ直しを実行する。
共通: --split stratified、15 エポック、m = 5、γ = 0.1、n_neg = 8、隠れ層 64（set_aware_reranker.py の既定）、--fast、
      --exclude-cases experiments/kato1020/excluded_cases.json。出力が既にあれば飛ばす。
使い方: python kato1020_run.py select [-P 6]
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "experiments" / "kato1020"
RAW = OUT / "raw"
EX = OUT / "excluded_cases.json"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
WEIGHTS = {"w70-30": (0.7, 0.3), "w50-50": (0.5, 0.5), "w30-70": (0.3, 0.7), "w00-100": (0.0, 1.0)}
NEGS = {"random": [], "hard": ["--hard-neg"]}
VARIANTS = {"A": "original,multisource_,dae_", "B": "dae_"}
SWEEP_N1 = [50, 200, 500, 1000, 2000]
ABLATIONS = ["reranker-2", "reranker-7", "reranker-2+Spec", "reranker-2+Svd", "reranker-2+Dom",
             "reranker-7+Comp", "reranker-7+Coh", "reranker-7+Dom"]


def sar(out, setting, seed, top_k, w, neg, extra=()):
    wt, wv = WEIGHTS[w]
    return (["python3", "set_aware_reranker.py", "--seed-list", str(seed), "--split", "stratified",
             "--variants", VARIANTS[setting], "--top-k", str(top_k), "--stage1-w-text", str(wt),
             "--stage1-w-var", str(wv), "--fast", "--exclude-cases", str(EX), "--save-per-case",
             "--output", str(out)] + NEGS[neg] + list(extra))


def jobs_select():
    rule = json.load(open(OUT / "n1_rule.json", encoding="utf-8"))["settings"]
    J = []
    for S in ("A", "B"):
        for seed in SEEDS:
            for w in WEIGHTS:
                for neg in NEGS:
                    n1 = rule[S][str(seed)][w]["N1"]
                    out = RAW / "select" / f"{S}_{w}_{neg}__{seed}.json"
                    J.append((out, sar(out, S, seed, n1, w, neg, ["--modes", "reranker-10S", "--eval-val", "--equiv"])))
    return J


def jobs_final(selected="selected.json", tag=""):
    sel = json.load(open(OUT / selected, encoding="utf-8"))["selected"]
    FIN = RAW / f"final{tag}"
    J = []
    for S in ("A", "B"):
        for seed in SEEDS:
            c = sel[S][str(seed)]
            for kind, flag in (("infer", "--greedy"), ("train", "--train-greedy")):
                out = FIN / f"{kind}_{S}__{seed}.json"
                J.append((out, sar(out, S, seed, c["N1"], c["w"], c["neg"],
                                   ["--modes", "reranker-10S", flag, "--eval-val", "--equiv"])))
    for seed in SEEDS:
        c = sel["A"][str(seed)]
        wt, wv = WEIGHTS[c["w"]]
        for m in ABLATIONS:
            out = FIN / f"{m.replace('+', '_')}_A__{seed}.json"
            J.append((out, sar(out, "A", seed, c["N1"], c["w"], c["neg"], ["--modes", m])))
        out = FIN / "pfi" / f"pfi_A__{seed}.json"
        J.append((out, ["python3", "kato1014_pfi.py", "--seed", str(seed), "--top-k", str(c["N1"]),
                        "--stage1-w-text", str(wt), "--stage1-w-var", str(wv), "--exclude", str(EX),
                        "--output", str(out)] + NEGS[c["neg"]]))
        for mg in ("orig", "L1", "K100", "K40"):
            out = FIN / "domain_merge" / f"{mg}__{seed}.json"
            J.append((out, ["python3", "kato1014_domain_merge.py", "--merge", mg, "--seed", str(seed),
                            "--top-k", str(c["N1"]), "--stage1-w-text", str(wt), "--stage1-w-var", str(wv),
                            "--exclude", str(EX), "--output", str(out)] + NEGS[c["neg"]]))
    c = sel["A"]["42"]
    wt, wv = WEIGHTS[c["w"]]
    out = OUT / f"case_outcomes_final{tag}.json"
    J.append((out, ["python3", "kato1014_case_outcomes.py", "--top-k", str(c["N1"]), "--tag", f"final{tag}",
                    "--stage1-w-text", str(wt), "--stage1-w-var", str(wv), "--exclude", str(EX),
                    "--out-dir", str(OUT)] + NEGS[c["neg"]]))
    out = OUT / f"topic_diag_final{tag}.json"
    J.append((out, ["python3", "kato1014_topic_diag.py", "--top-k", str(c["N1"]), "--tag", f"final{tag}",
                    "--w-text", str(wt), "--w-var", str(wv), "--exclude", str(EX), "--out-dir", str(OUT)]))
    return J


def jobs_sweep(selected="selected.json", tag=""):
    sel = json.load(open(OUT / selected, encoding="utf-8"))["selected"]
    J = []
    for S in ("A", "B"):
        for seed in SEEDS:
            c = sel[S][str(seed)]
            for n1 in SWEEP_N1:
                out = RAW / f"sweep{tag}" / f"static_{S}_N{n1}__{seed}.json"
                J.append((out, sar(out, S, seed, n1, c["w"], c["neg"], ["--modes", "reranker-10S", "--equiv"])))
    return J


def run(jobs, P):
    todo = [(o, c) for o, c in jobs if not (Path(o).exists() and Path(o).stat().st_size > 0)]
    print(f"jobs {len(jobs)}、実行 {len(todo)}（{P} 並列）", flush=True)

    def one(o, c):
        Path(o).parent.mkdir(parents=True, exist_ok=True)
        log = Path(str(o).rsplit(".", 1)[0] + ".log")
        t0 = time.time()
        with open(log, "w") as f:
            f.write("$ " + " ".join(shlex.quote(x) for x in c) + "\n"); f.flush()
            rc = subprocess.run(c, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        return o, rc, time.time() - t0

    fails = 0
    with ThreadPoolExecutor(max_workers=P) as ex:
        for fut in as_completed([ex.submit(one, o, c) for o, c in todo]):
            o, rc, dt = fut.result()
            fails += rc != 0
            print(f"{'done' if rc == 0 else 'FAIL'} {Path(o).relative_to(ROOT)} {dt:.0f}s", flush=True)
    print(f"ALL JOBS FINISHED（失敗 {fails}）", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "final", "sweep"])
    ap.add_argument("-P", type=int, default=6)
    ap.add_argument("--selected", default="selected.json", help="kato1020_select.py の出力（案 a/b/c）")
    ap.add_argument("--tag", default="", help="final・sweep の出力先の接尾辞（例 _w30-70）")
    a = ap.parse_args()
    jobs = jobs_select() if a.stage == "select" else {"final": jobs_final, "sweep": jobs_sweep}[a.stage](a.selected, a.tag)
    run(jobs, a.P)
