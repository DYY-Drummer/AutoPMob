#!/usr/bin/env python3
"""変数の説明による照合の分析: 本実験の前の確認（precheck）、検定（stats）、図（figure）.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.3, §3.6, §5
  python3 analyze_var_description.py precheck   # E5 が同じ変数と同じ式の中の別の変数を分ける AUC（< 0.80 なら終了コード 2）
  python3 analyze_var_description.py stats      # 対応あり検定・95% 信頼区間・判定・族別の内訳・QC を通らなかったケースを除いた比較 → var_desc_stats.json
  python3 analyze_var_description.py figure     # figures/fig_var_description.pdf / .png
出力先は各コマンドの中で、モジュールの EXP / FIG_DIR から決める（テストでは両方を差し替える）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
SPEC = "docs/superpowers/specs/2026-10-01-variable-description-matching-design.md"
FIG_DIR = ROOT / "figures"
AUC_GATE = 0.80
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]  # 正典の 10 seed。対にする seed は結果ファイルの値から取る
SENS_SEED = 42  # τ の感度を測る run の seed（--seed-list 42）
ZERO_VAR_TOL = 1e-12  # 差の幅（最大 − 最小）がこれ以下なら「すべて同じ」。浮動小数点の丸め（~1e-16）より十分大きく、指標の差より十分小さい
METRICS = ["Recall@K_correct", "MAP", "Recall@20"]
COMPARISONS = [  # (a, b, 目的): a − b を検定する
    ("D1", "S1", "置き換え"),
    ("S1-r50", "S1", "頑健性（半分を替える）"),
    ("S1-r100", "S1", "頑健性（全部替える）"),
    ("D1", "S1-r100", "全部替えた記号版と説明版"),
    ("D0", "D1", "漏れの大きさ"),
    ("S1", "S0", "説明文から記号を消した影響"),
]
QC_COMPARISONS = [  # (a, b): QC を通らなかったケース（qc_failed）を除いて a − b を検定し直す対。D1・S1-r の結果に効く 4 つ
    ("D1", "S1"), ("S1-r50", "S1"), ("S1-r100", "S1"), ("D1", "S1-r100"),
]
C_DESC, C_SYM, C_COPY = "#2a78d6", "#eb6834", "#1baf7a"  # dataviz の検証済み 3 色
INK = "#222222"


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    """厳密な JSON（NaN・inf があれば ValueError）で書く。先に文字列にするので、失敗しても出力先を壊さない."""
    text = json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def quantiles(a) -> dict:
    x = np.asarray(a, dtype=float)
    return {"mean": round(float(x.mean()), 4),
            **{f"p{q}": round(float(np.percentile(x, q)), 4) for q in (5, 25, 50, 75, 95)}}


def family(variant: str) -> str:
    if variant == "original":
        return "single"
    if variant.startswith("multisource_"):
        return "cross"
    if variant.startswith("dae_"):
        return "synth"
    return "other"


def per_seed(res: dict, label: str, metric: str) -> tuple:
    """(seed の昇順, その seed の指標値). seed は結果ファイルの値から取る。seed が重複していれば ValueError."""
    rows = sorted(res[label]["per_seed"], key=lambda p: p["seed"])
    seeds = [p["seed"] for p in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{label}: per_seed に seed の重複がある: {seeds}")
    return seeds, [p[metric] for p in rows]


def pair_by_seed(ma: dict, mb: dict, what: str) -> tuple:
    """seed → 値 の 2 つの辞書を seed の昇順で対にする。seed の集合が違えば ValueError（共通部分だけを黙って使わない）."""
    if set(ma) != set(mb):
        raise ValueError(f"{what}: seed の集合が一致しない: {sorted(ma)} vs {sorted(mb)}")
    seeds = sorted(ma)
    return seeds, [ma[s] for s in seeds], [mb[s] for s in seeds]


def ci95(d, digits: int | None = 4) -> list:
    """差の平均の 95% 信頼区間（t 分布）。既定は 4 桁に丸める（JSON・表示用）.

    digits=None なら丸めない（判定用。丸めた値で ±0.02 と比べると、厳密な上端 0.02003 が 0.0200 になって同程度に入る）。
    2 件未満では定義できないので [None, None]."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return [None, None]
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = float(d.mean() - h), float(d.mean() + h)
    return [lo, hi] if digits is None else [round(lo, digits), round(hi, digits)]


def is_significant(st: dict):
    """両検定で p < 0.05 なら True、そうでなければ False。p が定義できない（None）なら None."""
    p_t, p_w = st.get("p_ttest"), st.get("p_wilcoxon")
    if p_t is None or p_w is None:
        return None
    return bool(p_t < 0.05 and p_w < 0.05)


def verdict(st: dict, ci: list, unrounded_mean_diff: float | None = None) -> str:
    """結果の読み方（仕様 §5）。信頼区間が ±0.02 に収まれば同程度、両検定で p<0.05 なら有意.

    ci は丸める前の区間を渡す（ちょうど ±0.02 は収まる側）。
    unrounded_mean_diff は差の未丸め平均（向き判定用）。None なら st["mean_delta"] を使う。
    同程度でも有意なら "; significant (a higher|a lower)" を足す（差が小さいことと、差が 0 でないことは両立する）。
    検定が定義できず（p が None）、区間も ±0.02 に収まらなければ "inconclusive (tests undefined)".
    """
    sig = is_significant(st)
    # 向きは未丸めの差か、丸めた mean_delta から決める（未丸めが優先）
    mean_for_direction = unrounded_mean_diff if unrounded_mean_diff is not None else (st.get("mean_delta") or 0)
    direction = "a higher" if mean_for_direction > 0 else "a lower"
    if ci[0] is not None and -0.02 <= ci[0] and ci[1] <= 0.02:
        return "equivalent: 95% CI within ±0.02" + (f"; significant ({direction})" if sig else "")
    if sig:
        return f"{direction} (significant)"
    return "inconclusive (tests undefined)" if sig is None else "inconclusive"


def safe_paired(base, rer) -> dict:
    """analyze_significance.paired_stats の安全版（差 = rer − base）.

    差がすべて同じ（分散 0。全部 0 を含む）か、1 組しかないときは、t 検定・Wilcoxon・Cohen の d_z が定義できない。
    paired_stats は NaN や inf を返すので、それらを None にして note を付ける（std_delta は、2 組以上なら 0.0）。
    ほかに残った NaN・inf も None にする。入力が壊れている（組の数が違う・空・NaN や inf を含む）なら ValueError.
    """
    from analyze_significance import paired_stats
    b = np.asarray(base, dtype=float)
    r = np.asarray(rer, dtype=float)
    if b.ndim != 1 or b.shape != r.shape or b.size == 0:
        raise ValueError(f"対応あり検定の入力が不正: base {b.shape}, rer {r.shape}")
    if not (np.isfinite(b).all() and np.isfinite(r).all()):
        raise ValueError("対応あり検定の入力に NaN か inf がある")
    d = r - b
    if d.size < 2 or np.ptp(d) <= ZERO_VAR_TOL:
        why = "only one pair" if d.size < 2 else "all paired differences are identical (zero variance)"
        st = {"n": int(d.size), "mean_base": round(float(b.mean()), 4), "mean_rer": round(float(r.mean()), 4),
              "mean_delta": round(float(d.mean()), 4), "std_delta": 0.0 if d.size > 1 else None,
              "p_ttest": None, "p_wilcoxon": None, "cohen_dz": None, "note": f"tests undefined: {why}"}
    else:
        st = paired_stats(b, r)
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in st.items()}


def paired_with_ci(va, vb) -> dict:
    """a（va）と b（vb）の対応あり検定（差 = a − b）に、95% 信頼区間・有意か・判定を足す.

    ci95 には丸めた区間（JSON・表示用）を入れ、判定（verdict）は丸める前の区間と未丸めの平均差で行う。"""
    st = safe_paired(base=vb, rer=va)
    diff = np.asarray(va, dtype=float) - np.asarray(vb, dtype=float)
    st["ci95"] = ci95(diff)
    st["significant"] = is_significant(st)
    # verdict は未丸めの平均差で向きを決める（st["mean_delta"] は丸めてある）
    unrounded_mean_diff = float(diff.mean()) if len(diff) > 0 else None
    st["verdict"] = verdict(st, ci95(diff, digits=None), unrounded_mean_diff=unrounded_mean_diff)
    return st


def compare(res: dict, a: str, b: str, purpose: str) -> dict:
    entry = {"purpose": purpose, "a": a, "b": b}
    for metric in METRICS:
        ma = dict(zip(*per_seed(res, a, metric)))
        mb = dict(zip(*per_seed(res, b, metric)))
        _, va, vb = pair_by_seed(ma, mb, f"{a} vs {b} ({metric})")
        entry[metric] = paired_with_ci(va, vb)
    return entry


def per_case(res: dict, label: str) -> list:
    recs = res[label].get("per_case")
    if not recs:
        raise ValueError(f"{label}: per_case がない（run_var_description.py を --save-per-case で実行したか）")
    return recs


def case_seed_means(res: dict, label: str, exclude=frozenset(), fam: str | None = None) -> dict:
    """seed → そのテストケースの R@K の平均。exclude の case_id は除き、fam を与えればその族だけ."""
    by = {}
    for r in per_case(res, label):
        if r["case_id"] in exclude or (fam is not None and family(r["variant"]) != fam):
            continue
        by.setdefault(r["seed"], []).append(r["Recall@K_correct"])
    return {s: float(np.mean(v)) for s, v in by.items()}


def family_means(res: dict, label: str, exclude=frozenset()) -> dict:
    """族 → seed → その族のテストケースの R@K の平均."""
    fams = sorted({family(r["variant"]) for r in per_case(res, label)})
    out = {f: case_seed_means(res, label, exclude, f) for f in fams}
    return {f: m for f, m in out.items() if m}


def family_table(res: dict) -> dict:
    tab = {}
    for label in res:
        tab[label] = {}
        for f, d in family_means(res, label).items():
            v = list(d.values())
            tab[label][f] = {"mean": round(float(np.mean(v)), 4),
                             "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0,
                             "n_seeds": len(v)}
    return tab


def family_tests(res: dict, a: str = "D1", b: str = "S1") -> dict:
    """族ごとの a − b。その族のテストケースがある seed だけを使うが、a と b で seed の集合が違えば ValueError."""
    fa, fb = family_means(res, a), family_means(res, b)
    out = {}
    for f in sorted(set(fa) | set(fb)):
        _, va, vb = pair_by_seed(fa.get(f, {}), fb.get(f, {}), f"{a} vs {b} / 族 {f}")
        out[f] = paired_with_ci(va, vb)
    return {f"{a}_vs_{b}": out}


def excluding_conflict(res: dict, conflict: set, a: str = "D1", b: str = "S1",
                       reason: str = "meaning_conflict") -> dict:
    """conflict の case_id（meaning_conflict か qc_failed のケース）を除いた a − b。

    除いた数は、比較する 2 ラベルの per_case から実際に落ちた case_id の数（seed ごとと、全体の重複なしの数）。
    conflict には学習側にしかないケースも含まれるので、len(conflict) とは違う。
    reason は除く理由の名前で、seed の集合が合わないときのエラーの文言にだけ使う（値は変わらない）."""
    ma, mb = case_seed_means(res, a, conflict), case_seed_means(res, b, conflict)
    _, va, vb = pair_by_seed(ma, mb, f"{a} vs {b} ({reason} を除く)")
    seeds, dropped = set(), {}
    for label in (a, b):
        for r in per_case(res, label):
            seeds.add(r["seed"])
            if r["case_id"] in conflict:
                dropped.setdefault(r["seed"], set()).add(r["case_id"])
    return {f"{a}_vs_{b}": paired_with_ci(va, vb),
            "n_excluded_cases": len(set().union(*dropped.values())),
            "n_excluded_cases_per_seed": {str(s): len(dropped.get(s, ())) for s in sorted(seeds)}}


def cmd_precheck(args) -> None:
    from sklearn.metrics import roc_auc_score
    from var_matching import youden_tau
    from var_conditions import (load_data, load_var_descriptions, build_views, make_matcher,
                                calibration_pairs, pair_sims)
    data = load_data("A")
    vd = load_var_descriptions()
    idx = [i for i in range(len(data["cases"])) if data["correct_lists"][i]]
    out = {"spec": SPEC, "n_cases": len(idx)}
    aucs = {}
    for cond in ("D1", "D0"):
        views = build_views(cond, data, vd)
        matcher = make_matcher(cond, data, views)
        pr = calibration_pairs(idx, views, data)
        sims = pair_sims(pr, matcher)
        y = pr["label"]

        def row(t):
            ci, k, o = int(pr["case"][t]), int(pr["item"][t]), int(pr["occ"][t])
            return {"case_id": data["cases"][ci]["case_id"], "query_symbol": views[ci].orig_syms[k],
                    "query_text": views[ci].texts[k], "db_symbol": data["occ_sym"][o],
                    "db_description": data["occ_desc"][o],
                    "eq_key": data["eq_keys"][int(data["occ_eq"][o])], "sim": round(float(sims[t]), 4)}

        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        aucs[cond] = float(roc_auc_score(y, sims))
        out[cond] = {
            "n_pos": int(len(pos)), "n_neg": int(len(neg)),
            "auc": round(aucs[cond], 4),
            "tau_all_cases": round(youden_tau(sims, y), 4),
            "sim_pos": quantiles(sims[pos]), "sim_neg": quantiles(sims[neg]),
            "hardest_negatives": [row(t) for t in neg[np.argsort(-sims[neg])][:20]],
            "hardest_positives": [row(t) for t in pos[np.argsort(sims[pos])][:20]],
        }
        print(f"{cond}: AUC={out[cond]['auc']:.4f} pos={len(pos)} neg={len(neg)} "
              f"tau(all)={out[cond]['tau_all_cases']:.4f}", flush=True)
    # JSON には丸めた AUC を入れるが、ゲートは丸める前の値で判定する（0.79996 は 0.8000 に丸まる）
    out["gate"] = {"metric": "D1 AUC", "threshold": AUC_GATE, "pass": bool(aucs["D1"] >= AUC_GATE)}
    precheck_json = EXP / "var_desc_precheck.json"
    write_json(precheck_json, out)
    print(f"Saved: {precheck_json}")
    if not out["gate"]["pass"]:
        print(f"STOP: D1 の AUC {aucs['D1']:.6f} < {AUC_GATE}。本実験の前に本人に報告して相談する（仕様 §3.3）")
        sys.exit(2)


def result_line(tag: str, st: dict) -> str:
    def p(x):
        return "n/a" if x is None else x
    return (f"{tag}: Δ={st['mean_delta']:+.4f} CI={st['ci95']} p_t={p(st['p_ttest'])} "
            f"p_w={p(st['p_wilcoxon'])} → {st['verdict']}")


def seed_row(res: dict, label: str, seed: int) -> dict:
    rows = [p for p in res[label]["per_seed"] if p["seed"] == seed]
    if len(rows) != 1:
        raise ValueError(f"{label}: seed {seed} の per_seed が {len(rows)} 件ある（1 件のはず）")
    return rows[0]


def excluding_qc_failed(setting: str, res: dict, qc_failed: set) -> dict:
    """QC を通らなかったケース（qc_failed）を除いた比較（QC_COMPARISONS の対ごと）。対ごとに 1 行を出力する.

    qc_failed のケースには、D1 の問い合わせ文に記号が残るもの（para-leak）や、S1-r の付け替えが弱くなるもの
    （alt-same・alt-collides）が入りうる。それらを除いても結論が変わらないかを見る感度の確認で、
    meaning_conflict と違って比率のゲートはなく、いつも出す。ラベルが足りない対は、黙って落とさず error にする."""
    out = {}
    for a, b in QC_COMPARISONS:
        key, tag = f"{a}_vs_{b}", f"[{setting}] {a} − {b} excluding qc_failed"
        missing = [lab for lab in (a, b) if lab not in res]
        if missing:
            out[key] = {"error": f"missing label: {', '.join(missing)}"}
            print(f"{tag}: missing label ({', '.join(missing)})")
            continue
        ex = out[key] = excluding_conflict(res, qc_failed, a, b, reason="qc_failed")
        print(result_line(f"{tag} (n_excluded_cases={ex['n_excluded_cases']})", ex[key]))
    return out


def setting_stats(setting: str, doc: dict, conflict: set, exclusion: bool, qc_failed: set) -> dict:
    """設定 1 つ分: 比較（指標ごと）・族別・meaning_conflict を除いた比較・qc_failed を除いた比較。比較ごとに 1 行を出力する."""
    res = doc["results"]
    comps = {}
    for a, b, purpose in COMPARISONS:
        key, tag = f"{a}_vs_{b}", f"[{setting}] {a} − {b}"
        missing = [lab for lab in (a, b) if lab not in res]
        if missing:  # 黙って落とさず、何が足りないかを出す
            comps[key] = {"purpose": purpose, "a": a, "b": b, "error": f"missing label: {', '.join(missing)}"}
            print(f"{tag}: missing label ({', '.join(missing)})")
            continue
        comps[key] = compare(res, a, b, purpose)  # seed が合わなければ ValueError（握りつぶさない）
        print(result_line(tag, comps[key]["Recall@K_correct"]))
    out = {"comparisons": comps, "family": family_table(res), "family_tests": {},
           "tau": doc["tau"], "approx_mismatch": doc["approx_mismatch"]}
    if "D1" not in res or "S1" not in res:
        print(f"[{setting}] D1 − S1 の族別・meaning_conflict を除いた比較: missing label（D1 か S1）")
    else:
        out["family_tests"] = family_tests(res)
        for f, st in out["family_tests"]["D1_vs_S1"].items():
            print(result_line(f"[{setting}] D1 − S1 (family {f})", st))
        if exclusion:
            ex = out["excluding_meaning_conflict"] = excluding_conflict(res, conflict)
            print(result_line(f"[{setting}] D1 − S1 excluding meaning_conflict "
                              f"(n_excluded_cases={ex['n_excluded_cases']})", ex["D1_vs_S1"]))
    out["excluding_qc_failed"] = excluding_qc_failed(setting, res, qc_failed)  # D1 か S1 がなくても、残りの対は出す
    return out


def tau_sensitivity(res_a: dict) -> dict:
    """設定 A・seed 42 の D1 について、τ ± 0.02 の run の指標（"0" は元の τ）。run のファイルがなければ、その行を省く."""
    sens = {}
    for d in ("-0.02", "+0.02"):
        p = EXP / f"var_desc_tau{d}.json"
        if not p.exists():
            print(f"[A] tau{d}: {p.name} がない（感度の行を省く）")
            continue
        row = seed_row(load_json(p)["results"], f"D1-tau{d}", SENS_SEED)
        sens[d] = {k: row[k] for k in METRICS}
    if "D1" in res_a:
        row = seed_row(res_a, "D1", SENS_SEED)
        sens["0"] = {k: row[k] for k in METRICS}
    return sens


def cmd_stats(args) -> None:
    vd = load_json(EXP / "var_descriptions.json")["cases"]  # {"meta": ..., "cases": ...}。var_conditions は import しない
    qc = load_json(EXP / "var_descriptions_qc.json")
    conflict = {cid for cid, c in vd.items() if any(v["meaning_conflict"] for v in c["variables"].values())}
    qc_failed = {cid for cid, c in vd.items() if c["qc_failed"]}  # 最終のレコードが QC を通らなかったケース（finalize がどのレコードにも書く）
    ratio = qc["meaning_conflict_ratio"]
    exclusion = ratio > 0.05
    out = {"spec": SPEC, "meaning_conflict": {"ratio": round(ratio, 4), "n_cases_with_conflict": len(conflict),
                                              "exclusion_applied": exclusion},
           "qc_failed": {"n_cases": len(qc_failed)}}
    docs = {setting: load_json(EXP / f"var_desc_{setting}.json") for setting in ("A", "B")}
    for setting, doc in docs.items():
        out[setting] = setting_stats(setting, doc, conflict, exclusion, qc_failed)
    out["A"]["tau_sensitivity_seed42"] = tau_sensitivity(docs["A"]["results"])
    stats_json = EXP / "var_desc_stats.json"
    write_json(stats_json, out)
    print(f"Saved: {stats_json}")


def cmd_figure(args):
    """図を FIG_DIR に pdf・png で保存し、Figure を返す（テストが軸を調べる。main() は戻り値を使わないので CLI の動作は同じ）."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["pdf.fonttype"] = 42
    # 高さ 5.0 in: y 軸ラベル（10 pt で約 3.75 in の 1 行）が軸の高さに収まり、切れない
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    xs = [0, 50, 100]
    n_seeds = set()
    for ax, setting, title in zip(axes, ["A", "B"], ["(a) Setting A", "(b) Setting B"]):
        res = load_json(EXP / f"var_desc_{setting}.json")["results"]

        def ms(label):
            seeds, v = per_seed(res, label, "Recall@K_correct")
            n_seeds.add(len(seeds))
            return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

        s = [ms(lab) for lab in ("S1", "S1-r50", "S1-r100")]
        d_m, d_sd = ms("D1")
        c_m, _ = ms("D0")
        # D1・D0 は x ∈ [0, 100] の線分にする。axhline で軸の端まで引くと、右に置く直接ラベルを貫くため
        ax.fill_between([0, 100], d_m - d_sd, d_m + d_sd, color=C_DESC, alpha=0.15, lw=0, zorder=1)
        ax.plot([0, 100], [d_m, d_m], color=C_DESC, lw=2, zorder=2,
                label="Matching by descriptions (D1, uses no symbols)")
        ax.plot([0, 100], [c_m, c_m], color=C_COPY, lw=1.5, ls="--", zorder=2,
                label="Copied source descriptions (D0, reference)")
        ax.errorbar(xs, [m for m, _ in s], yerr=[sd for _, sd in s], color=C_SYM, lw=2, marker="o",
                    ms=7, capsize=4, zorder=3, label="Matching by symbols (S1)")
        for y, text in ((d_m, "D1"), (c_m, "D0"), (s[-1][0], "S1")):  # 線分の端（x=100）のすぐ右
            ax.annotate(text, xy=(100, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK)
        ax.set_xticks(xs)
        ax.set_xlim(-8, 124)
        ax.set_xlabel("Input/output variables given with a different symbol (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    n_txt = "/".join(str(n) for n in sorted(n_seeds))
    axes[0].set_ylabel(f"Recall@K (mean over {n_txt} seeds; bars and band: ± SD)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_var_description.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'fig_var_description.pdf'} (+ .png)")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("precheck", "stats", "figure"):
        sub.add_parser(name)
    args = ap.parse_args()
    {"precheck": cmd_precheck, "stats": cmd_stats, "figure": cmd_figure}[args.cmd](args)


if __name__ == "__main__":
    main()
