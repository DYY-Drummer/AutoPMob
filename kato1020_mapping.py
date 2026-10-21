"""8/31 報告の数値の旧値 → 新値の対応表（2026-10-20 の条件：91 件を外し、定数を検証用のケースで選び、検定はしない）.

出力: docs/加藤先生指示_2026-10-20_旧値新値対応表.md（報告書の本文は変えない。2026-10-14 版の対応表は残す）
入力: experiments/kato1020/{excluded_cases,n1_rule,baselines,selected,main,greedy,mech,pfi,sweep,topic_diag_final,case_outcomes_final}.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
K = ROOT / "experiments" / "kato1020"
OUTMD = ROOT / "docs" / "加藤先生指示_2026-10-20_旧値新値対応表.md"


def J(name):
    p = K / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ms(v):
    return "—" if not v else (f"{v['mean']:.3f} ± {v['std']:.3f}" if v.get("std") is not None else f"{v['mean']:.3f}")


def sd(x, nd=3):
    if x is None:
        return "—"
    if x == 0:
        return "0（差なし）"
    d = nd if abs(x) >= 0.0005 else (4 if abs(x) >= 0.00005 else 5)
    return f"{x:+.{d}f}".replace("-", "−")


def wc(c, m="Recall@20"):
    """counts の結果 → '+0.074（10 通り中 10 通りで上回る）'."""
    if not c or not c.get(m):
        return "（未計算）"
    v = c[m]
    t = f"、同じ {v['ties']}" if v.get("ties") else ""
    return f"{sd(v['mean_delta'])}（上回る {v['wins']}/{v['n']}、下回る {v['losses']}{t}）"


def _num(x):
    s = f"{x:.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def wfmt(w):
    """'w00-100' → '(0.0, 1.0)'、'w05-95' → '(0.05, 0.95)'."""
    a, b = w[1:].split("-")
    return f"({_num(int(a) / 100)}, {_num(int(b) / 100)})"


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def build(tag="", selected="selected.json", choice_note=""):
    ex, rule, bl, sel = J("excluded_cases.json"), J("n1_rule.json"), J("baselines.json"), J(selected)
    main, gr, mech, pfi, sw = J(f"main{tag}.json"), J(f"greedy{tag}.json"), J(f"mech{tag}.json"), J(f"pfi{tag}.json"), J(f"sweep{tag}.json")
    td, co = J(f"topic_diag_final{tag}.json"), J(f"case_outcomes_final{tag}.json")
    L = ["# 8/31 報告の数値：旧値（報告書）→ 新値（2026-10-20 の条件）\n",
         "報告書 `docs/progress_report_2026-08-31_methods_results.tex` の本文は変えていない。2026-10-14 の条件の対応表"
         "（`docs/加藤先生指示_2026-10-14_旧値新値対応表.md`）は、この表に置き換わる。\n",
         "## 条件（2026-10-14 からの変更点）\n"]
    if ex:
        L.append(f"1. 未知の変数が決まらない出典横断の {ex['n_excluded']} 件を評価と訓練から外した（本人の決定）。判定は "
                 "`analyze_kato_0831_followup.gt_checks` と同じ（正解式と未知の変数の完全マッチングがない）。"
                 f"{ex['n_excluded']} 件すべてで、どれかの正解式の変数がすべて入力変数になっている（{ex['n_derived_io_variants']} 件は入出力を選び直した派生）。"
                 f"設定 A は {ex['setting_A_after']:,} 件、設定 B（合成 1,000 件）は変わらない。一覧は `experiments/kato1020/excluded_cases.json`。")
    L.append("2. 定数を、分割ごとに検証用のケース（分割の val の出典のケース）の Recall@20 で選んだ（同じなら MAP）。"
             "選んだのは baseline の重み（12 通り）、BM25・E5 の重み（各 12 通り）、提案手法の第 1 段の重み（4 通り）と訓練の不正解の選び方（無作為／hard negative）。"
             "ほかの定数（m = 5、γ = 0.1、不正解 8 件、15 エポック、隠れ層 64、学習率 10^-3）は固定。")
    L.append("3. N1 は、重みと分割ごとに、訓練と検証のケースで全正解式が入る値を千の位に切り上げて決めた（91 件を外した後）。"
             "選ばれた設定の N1 を、その分割のすべての節（§4〜§9.1）で使う。" + choice_note)
    L.append("4. 検定はしない。差は、10 通りの分割のうち何通りで上回ったか（下回ったか）と、差の平均で示す。PFI も低下が正の分割の数を示す。")
    L.append("5. 指標：Recall@20（主）、R-Precision（旧 Recall@K）、MAP（標準の定義）、MAP@20（上位 20 件で打ち切り）。")
    L.append("6. 別の出典にある同じ式（空白などを除いて同じ LaTeX、§3.5 の 386/3,132 本と同じ定義）を正解とみなした値も出した"
             "（並べた順に 1 対 1 で数え、1 本の正解式は 1 回だけ）。\n")

    if sel and rule:
        L.append("## 選ばれた定数（分割ごと）\n")
        for S in ("A", "B"):
            c = Counter((v["w"], v["neg"], v["N1"]) for v in sel["selected"][S].values())
            txt = "、".join(f"重み {wfmt(w)}・{'hard negative' if neg == 'hard' else '無作為'}・N1 = {n1:,}：{n} 通り"
                           for (w, neg, n1), n in c.most_common())
            L.append(f"- 提案手法（設定 {S}）：{txt}")
        if bl:
            for S in ("A", "B"):
                for sc, nm in (("tfidf", "baseline（TF-IDF）"), ("bm25", "BM25"), ("e5", "E5")):
                    c = Counter(v["w"] for v in bl["settings"][S]["scorers"][sc]["selected"].values())
                    L.append(f"- {nm}（設定 {S}）：" + "、".join(f"{wfmt(w)} {n} 通り" for w, n in c.most_common()))
        def rng(S, w):
            v = sorted({x[w]["N1"] for x in rule["settings"][S].values()})
            return f"{v[0]:,}" if len(v) == 1 else f"{v[0]:,}〜{v[-1]:,}"
        L.append("- N1 の規則の値（訓練＋検証で全正解式が入る値を千の位に切り上げ、分割ごと）：" + "、".join(
            f"{wfmt(w)} は A {rng('A', w)}・B {rng('B', w)}" for w in rule["weights"]) + "\n")

    if sel and sel.get("grid"):
        L.append("## 付録（新規）：提案手法の第 1 段の重みと不正解の選び方ごとの値（10 通りの分割の平均 ± 標準偏差）\n")
        L.append("N1 は重みと分割ごとの規則の値（(0.3, 0.7) は全分割で 2,000）。検証用のケースでの重みの差は、分割間のばらつきより小さかったので、"
                 "全正解式が候補に入る N1 が最も小さい (0.3, 0.7) に固定した（2026-10-21 本人の決定）。\n")
        L.append(row("設定", "重み", "不正解", "検証用 Recall@20", "検証用 MAP", "テスト Recall@20", "テスト MAP", "テスト MAP@20"))
        L.append(row(*["---"] * 8))
        for S in ("A", "B"):
            for k in sorted(sel["grid"][S], key=lambda x: (x.split("|")[0], x.split("|")[1])):
                w, neg = k.split("|")
                g = sel["grid"][S][k]
                L.append(row(S, wfmt(w), "hard negative" if neg == "hard" else "無作為", ms(g["val"]["Recall@20"]), ms(g["val"]["MAP"]),
                             ms(g["test"]["Recall@20"]), ms(g["test"]["MAP"]), ms(g["test"]["MAP@20"])))
        L.append("")

    if main:
        L.append("## §4 全体比較（表 3）\n")
        L.append(row("設定", "手法", "旧値（報告書）", "新値 Recall@20", "新値 R-Precision", "新値 MAP", "新値 MAP@20"))
        L.append(row(*["---"] * 7))
        old = {"A": {"baseline_tfidf": "R@K 0.607、MAP 0.692、R@20 0.907（(0.4, 0.6)）",
                     "proposed_static": "R@K 0.780、MAP 0.850、R@20 0.957（(0.3, 0.7)、N1 = 50）"},
               "B": {"baseline_tfidf": "R@K 0.559、MAP 0.627、R@20 0.827",
                     "proposed_static": "R@K 0.740、MAP 0.795、R@20 0.925"}}
        names = {"baseline_tfidf": "古典的情報検索手法（TF-IDF、重みは検証用で選択）", "baseline_bm25": "BM25（付録）",
                 "baseline_e5": "E5（付録）", "proposed_static": "提案手法（一括の並べ直し）",
                 "proposed_infer": "提案手法（逐次選択・推論のみ）", "proposed_train": "提案手法（逐次選択・訓練も逐次）"}
        for S in ("A", "B"):
            T = main["table"][S]
            for g in ("baseline_tfidf", "baseline_bm25", "baseline_e5", "proposed_static", "proposed_infer", "proposed_train"):
                if g in T:
                    L.append(row(S, names[g], old[S].get(g, "—"), ms(T[g]["Recall@20"]), ms(T[g]["R-Precision"]),
                                 ms(T[g]["MAP"]), ms(T[g]["MAP@20"])))
        L.append("")
        L.append("差（10 通りの分割で対にした差の平均と、上回った分割の数）：\n")
        L.append(row("設定", "比較", "Recall@20", "MAP", "MAP@20"))
        L.append(row(*["---"] * 5))
        for S in ("A", "B"):
            C = main["compare"][S]
            for k in ("proposed_static vs baseline_tfidf", "proposed_train vs baseline_tfidf",
                      "proposed_static vs baseline_bm25", "proposed_static vs baseline_e5"):
                if k in C:
                    lab = k.replace("proposed_static", "提案（一括）").replace("proposed_train", "提案（訓練も逐次）").replace(
                        " vs baseline_tfidf", " − baseline").replace(" vs baseline_bm25", " − BM25").replace(" vs baseline_e5", " − E5")
                    L.append(row(S, lab, wc(C[k]), wc(C[k], "MAP"), wc(C[k], "MAP@20")))
        L.append("")
        L.append("### 別の出典にある同じ式を正解とみなした値（主結果の補足）\n")
        L.append(row("設定", "手法", "Recall@20", "（通常）", "MAP", "（通常）", "MAP@20"))
        L.append(row(*["---"] * 7))
        for S in ("A", "B"):
            T, TQ = main["table"][S], main["table_equiv"][S]
            for g in ("baseline_tfidf", "baseline_bm25", "baseline_e5", "proposed_static", "proposed_train"):
                if g in TQ:
                    L.append(row(S, names[g], ms(TQ[g]["Recall@20"]), f3(T[g]["Recall@20"]["mean"]),
                                 ms(TQ[g]["MAP"]), f3(T[g]["MAP"]["mean"]), ms(TQ[g]["MAP@20"])))
        C = main["compare"]
        for S in ("A", "B"):
            k = "proposed_static vs baseline_tfidf (equiv)"
            if k in C[S]:
                L.append(f"\n設定 {S}：同じ式も正解とすると、提案（一括）− baseline は Recall@20 {wc(C[S][k])}、MAP {wc(C[S][k], 'MAP')}。")
        L.append("")
    if bl:
        L.append("### 付録 A：古典的情報検索手法の重み別の値（テストのケース、10 通りの分割の平均 ± 標準偏差）\n")
        L.append("各分割では検証用のケースの Recall@20 で重みを選ぶ（選ばれた回数は上の「選ばれた定数」）。この表は、重みを 1 つに固定したときのテストの値。\n")
        L.append(row("手法", "重み", "A Recall@20", "A MAP", "A R-Precision", "B Recall@20", "B MAP", "B R-Precision"))
        L.append(row(*["---"] * 8))
        for sc, nm in (("tfidf", "TF-IDF"), ("bm25", "BM25"), ("e5", "E5")):
            for wl in bl["settings"]["A"]["scorers"][sc]["by_weight"]:
                a = bl["settings"]["A"]["scorers"][sc]["by_weight"][wl]["test"]
                b = bl["settings"]["B"]["scorers"][sc]["by_weight"][wl]["test"]
                L.append(row(nm, wfmt(wl), ms(a["Recall@20"]), ms(a["MAP"]), ms(a["R-Precision"]),
                             ms(b["Recall@20"]), ms(b["MAP"]), ms(b["R-Precision"])))
        L.append("")
    if main:
        L.append("### §4 の段落と付録 B（設定 A、正解式の数・入力変数の数で分けた値。分割のテストのケースを合わせた平均）\n")
        L.append(row("層", "件数", "提案（一括）R@20", "baseline R@20", "差", "提案 MAP", "baseline MAP", "差", "旧値（R@K の差）"))
        L.append(row(*["---"] * 9))
        oldK = {"1": "−0.012", "2": "+0.154", "3": "+0.161", "4": "+0.151", "5": "+0.182", "6": "+0.146", "7": "+0.133",
                "8": "+0.142", "9": "+0.138", "10": "+0.104"}
        oldI = {"<=5": "+0.021", "6-10": "+0.088", "11-15": "+0.193", "16-20": "+0.192", ">=21": "+0.163"}
        bk = main["buckets"]["A"]
        for key, olds, order in (("n_correct", oldK, [str(i) for i in range(1, 11)]),
                                 ("n_input", oldI, ["<=5", "6-10", "11-15", "16-20", ">=21"])):
            for b in order:
                c = bk[key].get(b)
                if not c:
                    continue
                pr, br = c.get("proposed_static|Recall@20"), c.get("baseline_tfidf|Recall@20")
                pm, bm = c.get("proposed_static|MAP"), c.get("baseline_tfidf|MAP")
                lab = f"正解式 {b} 本" if key == "n_correct" else f"入力変数 {b} 個"
                L.append(row(lab, f"{c['n_records']:,}", f3(pr), f3(br), sd(pr - br), f3(pm), f3(bm), sd(pm - bm), olds[b]))
        L.append("")

    if mech:
        L.append("## §5 向上の内訳（設定 A）\n")
        L.append(row("項目", "旧値（R@K、N1 = 50、(0.7, 0.3)）", "新値 Recall@20", "新値 MAP"))
        L.append(row(*["---"] * 4))
        S_, st = mech["summary"], mech["steps"]
        r2 = lambda g: f3(S_[g]["Recall@20"]["mean"]) if g in S_ else "—"  # noqa: E731
        L.append(row("(i) 同じ重みの baseline → x1・x2 だけの MLP", "0.521 → 0.601（+0.080）",
                     f"{r2('baseline_same_w')} → {r2('reranker-2')}、{wc(st['(i) baseline_same_w -> reranker-2'])}",
                     wc(st['(i) baseline_same_w -> reranker-2'], 'MAP')))
        L.append(row("(ii) 2 → 7 特徴量", "0.601 → 0.693（+0.092）", f"{r2('reranker-2')} → {r2('reranker-7')}、{wc(st['(ii) reranker-2 -> reranker-7'])}",
                     wc(st['(ii) reranker-2 -> reranker-7'], 'MAP')))
        spec = st["(ii') reranker-2 -> reranker-2+Spec"]
        L.append(row("(ii) 2 特徴量 ＋ 式の特化度 x6 だけ", "0.691（+0.090）", f"{r2('reranker-2_Spec')}、{wc(spec)}", wc(spec, "MAP")))
        L.append(row("(iii) 7 → 10 特徴量（集合の 3 個）", "0.693 → 0.743（+0.050）",
                     f"{r2('reranker-7')} → {r2('reranker-10S')}、{wc(st['(iii) reranker-7 -> reranker-10S'])}",
                     wc(st['(iii) reranker-7 -> reranker-10S'], 'MAP')))
        for x, nm, o in (("reranker-7_Comp", "x8（補完性）", "+0.052"), ("reranker-7_Coh", "x9（一貫性）", "+0.050"),
                         ("reranker-7_Dom", "x10（分野ラベルの一致度（集合））", "−0.0001")):
            t = mech["single_set_additions"].get(x)
            L.append(row(f"7 特徴量 ＋ {nm} − 7 特徴量", o, wc(t), wc(t, "MAP")))
        for key, lab, olds in (("lift_by_n_correct", "集合の特徴量の上乗せ（正解式の数別）", "1 本 +0.002、3 本 +0.102（最大）"),
                               ("lift_by_n_input", "集合の特徴量の上乗せ（入力変数の数別）", "11〜15 個 +0.078、16 個以上 +0.035")):
            rr = mech[key].get("reranker-10S", {})
            t1 = "、".join(f"{r['bin']} {sd(r.get('mean_delta'))}" for r in rr.get("Recall@20", []) if r.get("n_records"))
            t2 = "、".join(f"{r['bin']} {sd(r.get('mean_delta'))}" for r in rr.get("MAP", []) if r.get("n_records"))
            L.append(row(lab, olds, t1, t2))
        L.append("")
    if pfi:
        L.append("### §5 の PFI（図 2、設定 A）\n")
        L.append(row("項目", "旧値（R@K、N1 = 50）", "新値 Recall@20", "新値 MAP"))
        L.append(row(*["---"] * 4))
        b, t = pfi["base"], pfi["targets"]
        L.append(row("シャッフル前", "0.743", f3(b["Recall@20"]["mean"]), f3(b["MAP"]["mean"])))
        for lab, nm, old in (("GROUP_var", "変数の 6 特徴量をまとめて", "0.065（低下 0.678）"), ("specificity", "式の特化度 x6 だけ", "0.126"),
                             ("GROUP_topic", "話題の 4 特徴量をまとめて", "0.737（低下 0.006）"), ("gDom", "x10 だけ", "低下 −0.0003")):
            v, m2 = t[lab]["Recall@20"], t[lab]["MAP"]
            L.append(row(nm, old, f"{f3(v['shuffled_mean'])}（低下 {sd(v['drop_mean'])} ± {v['drop_std']:.3f}、正 {v['n_positive']}/{v['n_splits']}）",
                         f"{f3(m2['shuffled_mean'])}（低下 {sd(m2['drop_mean'])}、正 {m2['n_positive']}/{m2['n_splits']}）"))
        L.append(row("図 2", "fig_pfi_recall_ja.png", f"docs/figures/fig_pfi_recall20_ja_final{tag}.png（旧図は上書きしない）", ""))
        L.append("")

    if gr:
        L.append("## §6 逐次選択（付録 C）\n")
        L.append(row("設定", "項目", "旧値（R@K、N1 = 200、(0.7, 0.3)）", "新値 Recall@20", "新値 MAP"))
        L.append(row(*["---"] * 5))
        oldS = {"B": "(1) 0.724、(2) 0.756、(3) 0.765", "A": "(1) 0.757、(2) 0.787、(3) 0.797"}
        lab6 = {"infer - static": "(2) − (1)", "train - infer": "(3) − (2)", "train - static": "(3) − (1)"}
        for S in ("B", "A"):
            s = gr["summary"][S]["all"]
            L.append(row(S, "(1) 一括、(2) 推論のみ、(3) 訓練も逐次", oldS[S],
                         "、".join(f"({i}) {f3(s[k]['Recall@20']['mean'])}" for i, k in ((1, "static"), (2, "infer"), (3, "train")) if k in s),
                         "、".join(f"({i}) {f3(s[k]['MAP']['mean'])}" for i, k in ((1, "static"), (2, "infer"), (3, "train")) if k in s)))
            for sub in ("all", "K>=8"):
                for k, v in gr["compare"][S][sub].items():
                    L.append(row(S, f"{lab6[k]}{'（正解式 8 本以上）' if sub != 'all' else ''}", "", wc(v), wc(v, "MAP")))
        L.append(row("", "§6 で N1 を 200 に広げた理由、付録 C の N1 = 50 と 200 の比較", "8.0% / 2.2% が候補外",
                     "すべての節で同じ N1（選ばれた設定）なので不要。N1 の比較は付録（下の N1 の表）", ""))
        L.append("")

    L.append("## §7 自由度ゼロ停止\n\n削除のまま（2026-10-14 の指示 4）。\n")

    if mech:
        L.append("## §8 話題の特徴量（設定 A）\n")
        L.append(row("項目", "旧値（R@K、N1 = 50）", "新値 Recall@20", "新値 MAP"))
        L.append(row(*["---"] * 4))
        for k, lab, o in (("x7", "x1・x2 ＋ x7（分野ラベルの一致）− x1・x2", "+0.002（t 検定 p = 0.06）"),
                          ("x3", "x1・x2 ＋ x3（潜在意味類似度）− x1・x2", "−0.002（p = 0.48）"),
                          ("x10", "7 特徴量 ＋ x10 − 7 特徴量", "−0.0001（p = 0.98）")):
            t = mech["topic_additions"].get(k)
            L.append(row(lab, o, wc(t), wc(t, "MAP")))
        chk = mech["domain_merge"].get("orig_equals_reranker-2_Dom")
        if chk:
            L.append(row("確認：群で数える x7 を合併なしで使う", "—", f"per-case の順位が {chk['identical_ranks']:,}/{chk['n']:,} 件一致", ""))
        for mg, lab in (("L1", "表記をそろえた 1,958 種"), ("K100", "k-means で 100 群"), ("K40", "k-means で 40 群")):
            t = mech["domain_merge"].get(mg)
            L.append(row(f"分野ラベルをまとめる（{lab}）：x1・x2 ＋ x7 − x1・x2", "最大 +0.0006（旧版の 100 群・40 群は x7 がほぼ常に 0 で無効）",
                         wc(t), wc(t, "MAP")))
        L.append("")
    if td:
        L.append(f"### §8 の表 4（AUC、{td['config']['n_cases']:,} ケースの平均 ± 標準誤差、候補 {td['config']['top_k']:,} 件）\n")
        L.append(row("特徴量", "旧値 全 DB", "旧値 候補 50 件", "新値 全 DB", "新値 候補の中"))
        L.append(row(*["---"] * 5))
        old = {"x2_io_jaccard": ("変数の一致度 x2", "0.999", "0.947"), "x6_specificity": ("式の特化度 x6", "—", "0.965"),
               "x3_svd_sim": ("潜在意味類似度 x3", "0.950", "0.594"), "x1_text_sim": ("文章類似度 x1", "0.913", "0.437")}
        for f, (nm, o1, o2) in old.items():
            a, c = td["auc_full_db"][f], td["auc_within_candidates"][f]
            L.append(row(nm, o1, o2, f"{a['mean']:.3f} ± {a['sem']:.3f}", f"{c['mean']:.3f} ± {c['sem']:.3f}"))
        c = td["corr_text_jaccard_within_candidates_seed42_test"]
        L.append(f"\n候補内の相関 corr(x1, x2)（種 42 の分割のテスト {td['n_test_seed42']} ケース）：旧値 −0.404 → 新値 {sd(c['mean'])}。\n")
    if pfi:
        c, cm_ = pfi["cancellation"]["gDom"]["Recall@20"], pfi["cancellation"]["gDom"]["MAP"]
        L.append("### §8 (3) x10 の PFI の打ち消し合い（ケース単位に集約、検定はしない）\n")
        L.append(row("項目", "旧値（R@K）", "新値 Recall@20", "新値 MAP"))
        L.append(row(*["---"] * 4))
        L.append(row("延べ件数 / 異なるケース", "4,594 / 1,592", f"{c['n_records']:,} / {c['n_cases']:,}", "同左"))
        L.append(row("sep > 0（正解式の方で x10 が高い）のケース", "1,169 件（低下 0.012）", f"{c['n_sep_pos']:,} 件（低下 {sd(c['mean_drop_sep_pos'], 4)}）",
                     f"{cm_['n_sep_pos']:,} 件（低下 {sd(cm_['mean_drop_sep_pos'], 4)}）"))
        L.append(row("sep < 0 のケース", "414 件（低下 −0.034）", f"{c['n_sep_neg']:,} 件（低下 {sd(c['mean_drop_sep_neg'], 4)}）",
                     f"{cm_['n_sep_neg']:,} 件（低下 {sd(cm_['mean_drop_sep_neg'], 4)}）"))
        L.append("")

    if co:
        o20, oK = co["top20"], co["topK"]
        L.append(f"## §9.1 誤りの位置（種 42、テスト {o20['n_cases']} ケース、正解式 {o20['n_gt']:,} 本）\n")
        L.append(row("項目", "旧値（上位 K 件、N1 = 50 / 200）", "新値（上位 20 件）", "新値（上位 K 件、参考）"))
        L.append(row(*["---"] * 4))
        L.append(row("漏れた正解式", "435 本 / 373 本", f"{o20['missed_total']} 本", f"{oK['missed_total']} 本"))
        L.append(row("うち第 1 段の見逃し", "269 本（61.8%）/ 106 本", f"{o20['stage1_miss']} 本", f"{oK['stage1_miss']} 本"))
        L.append(row("うち第 2 段の見逃し", "166 本 / 267 本", f"{o20['stage2_miss']} 本", f"{oK['stage2_miss']} 本"))
        L.append(row("全正解式が入ったケース / 漏れのあるケース", "212 / 211", f"{o20['solved']} / {o20['unsolved']}", f"{oK['solved']} / {oK['unsolved']}"))
        for a, lab, old in (("n_correct", "正解式の数の平均（全部 / 漏れあり）", "2.13 / 5.34"),
                            ("n_input", "入力変数の数の平均（全部 / 漏れあり）", "10.8 / 17.8")):
            v, w = o20["solved_vs_unsolved"][a], oK["solved_vs_unsolved"][a]
            L.append(row(lab, old, f"{v['mean_solved']:.2f} / {v['mean_unsolved']:.2f}", f"{w['mean_solved']:.2f} / {w['mean_unsolved']:.2f}"))
        L.append("\n（旧版の Mann–Whitney の p 値は、検定をやめたので新値では出さない。）\n")

    if sw:
        L.append("## 付録 C（新規）：N1 を変えたとき（第 1 段の重み (0.3, 0.7)、不正解は検証用で選んだもの）\n")
        L.append("一括の並べ直し（N1 = 2,000 は本研究の設定）。被覆率はテストのケースで候補に入った正解式の割合。時間は 1 回の訓練と評価の壁時計の時間"
                 "（6 本を並列に実行）。図は `docs/figures/fig_n1_full_coverage_ja_1020.png`（全正解式が候補に入るケースの割合と被覆率、"
                 "比べた N1 を点線、本研究の N1 を実線。全ケースで全正解式が入るのは N1 = 1,880）。\n")
        L.append(row("設定", "N1", "Recall@20", "R-Precision", "MAP", "MAP@20", "被覆率", "訓練＋評価の時間（秒）"))
        L.append(row(*["---"] * 8))
        for S in ("A", "B"):
            for n1, v in sw["N1"][S].items():
                if n1.startswith("selected"):
                    continue
                tt = v.get("time_total_sec")
                L.append(row(S, f"{int(n1):,}", ms(v["Recall@20"]), ms(v["R-Precision"]), ms(v["MAP"]), ms(v["MAP@20"]),
                             f3(v["coverage"]["mean"]), f"{tt['mean']:.0f} ± {tt['std']:.0f}" if tt else "—"))
            for k, nm in (("infer", "逐次選択（推論のみ）"), ("train", "逐次選択（訓練も逐次）")):
                v = sw.get("sequential_N1_2000", {}).get(S, {}).get(k)
                if v:
                    tt = v.get("time_total_sec")
                    L.append(row(S, f"2,000（{nm}）", ms(v["Recall@20"]), ms(v["R-Precision"]), ms(v["MAP"]), ms(v["MAP@20"]),
                                 f3(v["coverage"]["mean"]), f"{tt['mean']:.0f} ± {tt['std']:.0f}（検証用も評価）" if tt else "—"))
        chk = sw.get("check_sweep2000_equals_selected")
        if chk:
            ok = "設定 A・B とも一致" if all(chk.values()) else f"一致しない設定あり：{chk}"
            L.append(f"\n確認：N1 = 2,000 の行は、§4 の一括の並べ直し（同じ設定の別の実行）と値が一致する（{ok}）。")
        L.append(f"\n{sw['note']}\n")
    return "\n".join(L) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--selected", default="selected.json")
    ap.add_argument("--choice-note", default="")
    a = ap.parse_args()
    md = build(a.tag, a.selected, a.choice_note)
    OUTMD.write_text(md, encoding="utf-8")
    print(f"Saved: {OUTMD}")


if __name__ == "__main__":
    main()
