#!/usr/bin/env python3
"""変数の説明による照合の追加実験: 検証用ケースで τ と第 1 段の重みを選び、選んだ設定の test の値を報告する.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §7
  python3 tune_var_description.py run --setting A [--seed-list 42,123] [--tau-deltas 0,0.02,0.04,0.06,0.08]
      [--weights 0.7:0.3,0.5:0.5,0.3:0.7] [--renames 0.5,1.0] [--save-per-case] --output PATH
  python3 tune_var_description.py merge --setting A --inputs PART1 PART2 ... --output PATH
seed ごとに、層化分割の train で学習し、val の R@K（Recall@K_correct）が最大の設定を選ぶ。選んだ設定の test の値を報告する（test は選択に使わない）。
  S1: 第 1 段の重み（文章:変数）3 通り（τ = 1 のまま）
  D1: Δτ（その seed の Youden の τ に足す値）5 通り × 重み 3 通り = 15 通り
ラベル: S1-tuned, S1-tuned-r50, S1-tuned-r100（選んだ S1 の模型で、テストの記号を替える）, D1-tuned。
格子の (Δτ=0, 重み (0.3, 0.7)) は本実験（run_var_description.py）の S1・D1 と同じ設定なので、test の値が厳密に一致する。
merge は、seed を分けて並列に走らせた run の出力（部分ファイル）を 1 つにする。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate_multi_eq import aggregate_metrics  # noqa: E402
from run_var_description import (  # noqa: E402
    SEEDS, HP, SPEC, prepare, train, evaluate, tau_for, summarize, per_case_record, save_json,
    make_tfidf, seed_svd_sim, split_cases,
)
from var_conditions import load_data, load_var_descriptions, build_views, make_matcher  # noqa: E402
from var_matching import SYMBOL_TAU, occurrence_incidence  # noqa: E402

TAU_DELTAS = "0,0.02,0.04,0.06,0.08"  # --tau-deltas の既定（D1 の τ = その seed の Youden の τ + Δτ）
WEIGHTS = "0.7:0.3,0.5:0.5,0.3:0.7"   # --weights の既定（第 1 段の重み 文章:変数）
RENAMES = "0.5,1.0"                   # --renames の既定（run_var_description.py と同じ）
DEFAULT_W_TEXT = 0.3                  # 本実験の既定の重みは (0.3, 0.7)。同点のときは文章の重みがこれに近い方を選ぶ
TIE_TOL = 1e-12                       # val の R@K の差がこれ以下なら同点（浮動小数点の誤差 ~1e-16 より大きく、R@K の本物の差より十分小さい）
SCORE_KEYS = ("Recall@K_correct", "MAP", "Recall@20", "n_cases")  # 格子の点ごとに残す val・test の値
CONDS = ("S1", "D1")


# ---- 引数の解析（argparse の type に渡す。不正なら ArgumentTypeError） ----

def parse_floats(s: str, what: str) -> list:
    try:
        vals = [float(p) for p in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what} は数のカンマ区切り（例: 0,0.02）: {s!r}") from None
    if not all(math.isfinite(v) for v in vals):
        raise argparse.ArgumentTypeError(f"{what} に有限でない数がある: {s!r}")
    return vals


def parse_tau_deltas(s: str) -> list:
    vals = parse_floats(s, "--tau-deltas")
    if len(set(vals)) != len(vals):
        raise argparse.ArgumentTypeError(f"--tau-deltas に同じ値がある: {s!r}")
    return vals


def parse_weights(s: str) -> list:
    """'0.7:0.3,0.5:0.5' → [(0.7, 0.3), (0.5, 0.5)]（文章の重み, 変数の重み）."""
    out = []
    for part in s.split(","):
        pair = part.split(":")
        try:
            if len(pair) != 2:
                raise ValueError
            w_text, w_var = float(pair[0]), float(pair[1])
        except ValueError:
            raise argparse.ArgumentTypeError(f"--weights は 文章:変数 のカンマ区切り（例: 0.7:0.3,0.5:0.5）: {s!r}") from None
        if not (math.isfinite(w_text) and math.isfinite(w_var) and w_text >= 0 and w_var >= 0 and w_text + w_var > 0):
            raise argparse.ArgumentTypeError(f"--weights の重みは 0 以上の有限な数で、2 つとも 0 ではない: {part!r}")
        out.append((w_text, w_var))
    if len(set(out)) != len(out):
        raise argparse.ArgumentTypeError(f"--weights に同じ組がある: {s!r}")
    return out


def parse_seed_list(s: str) -> list:
    try:
        seeds = [int(p) for p in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--seed-list は整数のカンマ区切り（例: 42,123）: {s!r}") from None
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"--seed-list に同じ seed がある: {s!r}")
    return seeds


def rename_suffix(p: float) -> str:
    return f"r{int(round(p * 100))}"


def parse_renames(s: str) -> list:
    """付け替えの割合。空文字は「付け替えの評価を省く」（run_var_description.py の --renames と同じ）."""
    if not s.strip():
        return []
    ps = parse_floats(s, "--renames")
    if not all(0 < p <= 1 for p in ps):
        raise argparse.ArgumentTypeError(f"--renames の割合は 0 より大きく 1 以下: {s!r}")
    if len({rename_suffix(p) for p in ps}) != len(ps):
        raise argparse.ArgumentTypeError(f"--renames のラベル（S1-tuned-r..）が重なる: {s!r}")
    return ps


# ---- 選択（val の R@K が最大。同点は本実験の既定に近い方） ----

def tie_key(e: dict) -> tuple:
    """同点のときの優先順位（小さいほど本実験の既定 Δτ=0・重み (0.3, 0.7) に近い）.

    Δτ の小さい方 → 文章の重みが既定の 0.3 に近い方（既定の 3 組では 0.3 → 0.5 → 0.7 の順）。
    等距離は値の小さい方。距離は丸めてから比べる（0.2 と 0.4 は 0.3 から等距離だが、浮動小数点では 1e-17 ずれる）."""
    return (abs(e["tau_delta"]), e["tau_delta"], round(abs(e["w_text"] - DEFAULT_W_TEXT), 9), e["w_text"])


def select_entry(entries: list) -> dict:
    """val の R@K（Recall@K_correct）が最大の格子点を返す（entries の要素そのもの）。test は見ない.

    最大が複数（TIE_TOL 以内）なら tie_key が最小のもの。val の R@K が有限でなければ ValueError."""
    if not entries:
        raise ValueError("格子点がない")
    vals = [e["val"]["Recall@K_correct"] for e in entries]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"val の R@K が有限でない: {vals}")
    best = max(vals)
    return min((e for e, v in zip(entries, vals) if v >= best - TIE_TOL), key=tie_key)


def grid_points(tau0: float, tau_deltas: list, weights: list) -> list:
    """格子の点 (Δτ, τ, 文章の重み, 変数の重み)。Δτ が外側・重みが内側の順に評価する."""
    return [(d, tau0 + d, w_text, w_var) for d in tau_deltas for w_text, w_var in weights]


def make_entry(dtau, tau, w_text, w_var, agg_va: dict, agg_te: dict) -> dict:
    return {"tau_delta": dtau, "tau": tau, "w_text": w_text, "w_var": w_var,
            "val": {k: agg_va[k] for k in SCORE_KEYS}, "test": {k: agg_te[k] for k in SCORE_KEYS}}


# ---- run ----

def evaluate_grid(seed, cond, env, points, idx, svd_sim) -> list:
    """条件 cond の格子 points を評価する。各点で train を学習し、val と test を評価する.

    返り値は [(entry, 学習した模型, test のケース別結果)]。特徴量の準備は train + val + test の全ケース."""
    views, matcher, X_ctx = env.conds[cond]
    tr, va, te = idx["train"], idx["val"], idx["test"]
    rows = []
    for dtau, tau, w_text, w_var in points:
        t0 = time.time()
        prep, _ = prepare(tr + va + te, views, matcher, tau, X_ctx, env.X_eq, svd_sim, env.data, env.inc,
                          w_text, w_var)
        model = train(seed, tr, prep, env.cl)
        res_va, res_te = evaluate(model, va, prep, env.data), evaluate(model, te, prep, env.data)
        agg_va, agg_te = aggregate_metrics(res_va), aggregate_metrics(res_te)
        rows.append((make_entry(dtau, tau, w_text, w_var, agg_va, agg_te), model, res_te))
        print(f"  [seed={seed}] {cond} dtau={dtau:+.2f} w=({w_text:g},{w_var:g}) tau={tau:.4f} "
              f"val R@K={agg_va['Recall@K_correct']:.4f} test R@K={agg_te['Recall@K_correct']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return rows


def tune_seed(seed, env, args) -> SimpleNamespace:
    """1 seed 分: S1 と D1 の格子を評価し、val で選んだ設定の test の結果を返す.

    返り値: tau0（D1 の Youden の τ）, grid[条件]（entry のリスト）, selected[条件]（entry）, evals[ラベル]（test のケース別結果）。
    S1-tuned-r* は、選んだ S1 の模型を、選んだ重みで準備した付け替え済みのテスト用ケースで評価する（run_condition の S1-r と同じ）."""
    idx = split_cases(env.data, seed)
    if not idx["val"] or not idx["test"]:
        raise ValueError(f"seed {seed}: val か test に正解式のあるケースがない（val で選べない・test を報告できない）")
    svd_sim_of = seed_svd_sim(seed, env.tfidf, env.X_eq)
    out = SimpleNamespace(tau0=None, grid={}, selected={}, evals={})
    for cond in CONDS:
        views, matcher, X_ctx = env.conds[cond]
        tau0 = tau_for(matcher, views, env.data, idx["train"])  # 記号版は SYMBOL_TAU、説明版は train の Youden の J が最大の τ
        deltas = [0.0] if cond == "S1" else args.tau_deltas  # S1 は τ = 1 のまま
        rows = evaluate_grid(seed, cond, env, grid_points(tau0, deltas, args.weights), idx, svd_sim_of(views))
        best_entry = select_entry([r[0] for r in rows])
        _, model, res_te = next(r for r in rows if r[0] is best_entry)
        out.grid[cond], out.selected[cond] = [r[0] for r in rows], best_entry
        out.evals[f"{cond}-tuned"] = res_te
        if cond == "D1":
            out.tau0 = tau0
        else:
            for suffix, rv in env.rviews.items():
                rprep, _ = prepare(idx["test"], rv, env.rmatchers[suffix], SYMBOL_TAU, X_ctx, env.X_eq,
                                   svd_sim_of(rv), env.data, env.inc, best_entry["w_text"], best_entry["w_var"])
                out.evals[f"S1-tuned-{suffix}"] = evaluate(model, idx["test"], rprep, env.data)
        print(f"  [seed={seed}] {cond} selected: dtau={best_entry['tau_delta']:+.2f} "
              f"w=({best_entry['w_text']:g},{best_entry['w_var']:g}) tau={best_entry['tau']:.4f} "
              f"val R@K={best_entry['val']['Recall@K_correct']:.4f} "
              f"test R@K={best_entry['test']['Recall@K_correct']:.4f}", flush=True)
    return out


def cmd_run(args) -> None:
    seeds = args.seed_list or SEEDS
    data = load_data(args.setting)
    vd = load_var_descriptions()
    tfidf, X_eq = make_tfidf(data["eq_texts"])
    env = SimpleNamespace(data=data, cl=data["correct_lists"], tfidf=tfidf, X_eq=X_eq,
                          inc=occurrence_incidence(data["eq_ptr"]), conds={}, rviews={}, rmatchers={})
    for cond in CONDS:  # seed によらない準備は 1 回だけ（run_condition と同じ作り方）
        views = build_views(cond, data, vd)
        env.conds[cond] = (views, make_matcher(cond, data, views), tfidf.transform([v.ctx for v in views]))
    for p in args.renames:
        env.rviews[rename_suffix(p)] = rv = build_views("S1", data, vd, p)
        env.rmatchers[rename_suffix(p)] = make_matcher("S1", data, rv)
    config = {**{k: v for k, v in vars(args).items() if k != "cmd"}, "seed_list": list(seeds),
              **HP, "spec": SPEC, "n_cases": len(data["cases"])}
    labels = ["S1-tuned", *[f"S1-tuned-{s}" for s in env.rviews], "D1-tuned"]
    tau, grid, selected = {"D1": {}}, {c: {} for c in CONDS}, {c: {} for c in CONDS}
    aggs = {lab: [] for lab in labels}
    pcs = {lab: [] for lab in labels} if args.save_per_case else None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"=== tuning setting {args.setting}: seeds {list(seeds)}, tau deltas {args.tau_deltas}, "
          f"weights {args.weights} ===", flush=True)
    for seed in seeds:
        t0 = time.time()
        r = tune_seed(seed, env, args)
        tau["D1"][str(seed)] = r.tau0
        for cond in CONDS:
            grid[cond][str(seed)], selected[cond][str(seed)] = r.grid[cond], r.selected[cond]
        for lab in labels:
            agg = aggregate_metrics(r.evals[lab])
            agg["seed"] = seed
            aggs[lab].append(agg)
            if pcs is not None:
                pcs[lab] += [per_case_record(seed, lab, cm) for cm in r.evals[lab]]
        # seed ごとに中間結果を保存（失敗時に済んだ seed を残す）
        save_json({"config": config, "tau": tau, "grid": grid, "selected": selected,
                   "results": {lab: summarize(lab, aggs[lab], pcs[lab] if pcs is not None else None)
                               for lab in labels}}, args.output)
        print(f"  [seed={seed}] done ({time.time() - t0:.0f}s)", flush=True)
    print(f"Saved: {args.output}")


# ---- merge ----

class MergeError(ValueError):
    """部分ファイルを結合できない（メッセージに原因を書く）."""


def part_seeds(name: str, doc: dict) -> list:
    """部分ファイルの seed（昇順）。tau・grid・selected・results が同じ seed の集合を持たなければ MergeError."""
    missing = [k for k in ("config", "tau", "grid", "selected", "results") if k not in doc]
    if missing or not isinstance(doc["config"], dict):
        raise MergeError(f"{name}: tune_var_description.py run の出力の形式ではない"
                         f"（{', '.join(missing) + ' がない' if missing else 'config が辞書でない'}）")
    try:
        sets = {"tau.D1": {int(s) for s in doc["tau"]["D1"]}}
        for key in ("grid", "selected"):
            for cond in CONDS:
                sets[f"{key}.{cond}"] = {int(s) for s in doc[key][cond]}
        rows_of = {label: res["per_seed"] for label, res in doc["results"].items()}
        for label, rows in rows_of.items():
            sets[f"results.{label}"] = {r["seed"] for r in rows}
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        raise MergeError(f"{name}: tune_var_description.py run の出力の形式ではない（{type(e).__name__}: {e}）") from None
    for label, rows in rows_of.items():
        if len(rows) != len(sets[f"results.{label}"]):
            raise MergeError(f"{name}: results.{label} の per_seed に seed の重複がある")
    union = set().union(*sets.values())
    for what, s in sets.items():
        if s != union:
            raise MergeError(f"{name}: {what} の seed が他の構造と違う（{what} にない seed: {sorted(union - s)}）")
    return sorted(union)


def merge_parts(parts: dict, setting: str, output: str) -> dict:
    """部分ファイル（名前 → 内容）を 1 つにする.

    条件: config が seed_list・output を除いて同一／seed が互いに重ならない／seed の和集合が正典の SEEDS と一致する。
    tau・grid・selected は seed ごとにまとめ、results はラベルごとに per_seed を seed の昇順に並べ、要約を summarize で作り直し、
    per_case は連結する（seed の昇順、同じ seed の中は元の順）。満たさなければ MergeError."""
    if not parts:
        raise MergeError("入力の部分ファイルがない")
    names = list(parts)
    seeds_of = {n: part_seeds(n, parts[n]) for n in names}
    for n in names:
        got = parts[n]["config"].get("setting")
        if got != setting:
            raise MergeError(f"{n}: config の setting が {got!r}（--setting {setting}）")
    first = names[0]

    def comparable(cfg):
        return {k: v for k, v in cfg.items() if k not in ("seed_list", "output")}

    for n in names[1:]:
        a, b = comparable(parts[first]["config"]), comparable(parts[n]["config"])
        if a != b:
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise MergeError(f"config が一致しない（{first} と {n}）: {', '.join(diff)}（seed_list と output だけは違ってよい）")
    for n in names:  # run が途中で止まった部分ファイルを、seed が足りないという一般の誤りより先に名指しする
        want = parts[n]["config"].get("seed_list")
        if isinstance(want, list) and set(want) - set(seeds_of[n]):
            raise MergeError(f"{n}: run が途中で止まっている（config の seed_list にあって、結果にない seed: "
                             f"{sorted(set(want) - set(seeds_of[n]))}）")
    labels = list(parts[first]["results"])
    for n in names[1:]:
        if set(parts[n]["results"]) != set(labels):
            raise MergeError(f"結果のラベルが一致しない（{first} と {n}）: "
                             f"{', '.join(sorted(set(parts[n]['results']) ^ set(labels)))}")
    for lab in labels:
        if len({"per_case" in parts[n]["results"][lab] for n in names}) > 1:
            raise MergeError(f"per_case がある部分ファイルとない部分ファイルが混じっている（{lab}）")
    owner, overlap = {}, {}
    for n in names:
        for s in seeds_of[n]:
            if s in owner:
                overlap.setdefault(s, [owner[s]]).append(n)
            else:
                owner[s] = n
    if overlap:
        raise MergeError("seed が重なっている: " + "; ".join(f"{s}（{' と '.join(ns)}）" for s, ns in sorted(overlap.items())))
    missing, extra = sorted(set(SEEDS) - set(owner)), sorted(set(owner) - set(SEEDS))
    if missing:
        raise MergeError(f"seed が足りない: {missing}（正典の {len(SEEDS)} seed がそろうこと）")
    if extra:
        raise MergeError(f"正典の seed にない seed がある: {extra}")
    ordered = sorted(owner)
    results = {}
    for lab in labels:
        rows = sorted((r for n in names for r in parts[n]["results"][lab]["per_seed"]), key=lambda r: r["seed"])
        pcs = (sorted((r for n in names for r in parts[n]["results"][lab]["per_case"]), key=lambda r: r["seed"])
               if "per_case" in parts[first]["results"][lab] else None)
        results[lab] = summarize(lab, rows, pcs)
    config = dict(parts[first]["config"])
    config["seed_list"], config["output"] = ordered, output
    return {"config": config,
            "tau": {"D1": {str(s): parts[owner[s]]["tau"]["D1"][str(s)] for s in ordered}},
            "grid": {c: {str(s): parts[owner[s]]["grid"][c][str(s)] for s in ordered} for c in CONDS},
            "selected": {c: {str(s): parts[owner[s]]["selected"][c][str(s)] for s in ordered} for c in CONDS},
            "results": results}


def cmd_merge(args) -> None:
    out_path = Path(args.output).resolve()
    if len({str(Path(p).resolve()) for p in args.inputs}) != len(args.inputs):
        sys.exit("merge: 同じ入力が 2 回ある")
    parts = {}
    for p in args.inputs:
        if Path(p).resolve() == out_path:
            sys.exit(f"merge: 出力 {args.output} が入力のファイルと同じ（入力を上書きしない）")
        try:
            with open(p, encoding="utf-8") as f:
                parts[p] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            sys.exit(f"merge: 入力 {p} を読めない: {e}")
    try:
        merged = merge_parts(parts, args.setting, args.output)
    except MergeError as e:
        sys.exit(f"merge: {e}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(merged, args.output)
    for lab, res in merged["results"].items():
        m = res["Recall@K_correct"]
        print(f"  {lab:14s} R@K={m['mean']:.4f} ± {m['std']:.4f} ({len(res['per_seed'])} seeds)")
    print(f"Saved: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="格子を評価し、val で選んだ設定の test の値を保存する")
    run.add_argument("--setting", choices=["A", "B"], required=True)
    run.add_argument("--seed-list", type=parse_seed_list, default=None, help="カンマ区切り（既定: 正典の 10 seed）")
    run.add_argument("--tau-deltas", type=parse_tau_deltas, default=TAU_DELTAS,
                     help=f"D1 の τ に足す値（既定: {TAU_DELTAS}）")
    run.add_argument("--weights", type=parse_weights, default=WEIGHTS,
                     help=f"第 1 段の重み 文章:変数（既定: {WEIGHTS}）")
    run.add_argument("--renames", type=parse_renames, default=RENAMES,
                     help=f"選んだ S1 の模型で評価する、記号を替える割合（既定: {RENAMES}）")
    run.add_argument("--save-per-case", action="store_true")
    run.add_argument("--output", required=True)
    merge = sub.add_parser("merge", help="seed を分けた run の出力（部分ファイル）を 1 つにする")
    merge.add_argument("--setting", choices=["A", "B"], required=True)
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    {"run": cmd_run, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
