"""8/31 報告（docs/progress_report_2026-08-31_methods_results.tex）の数値の、旧値 → 新値の対応表を作る.

2026-10-14 の加藤先生の指示 3・4・5 による再実験の結果（experiments/kato1014/*.json）から、報告書に載っている
数値ごとに新しい値を引き、docs/加藤先生指示_2026-10-14_旧値新値対応表.md に書き出す（報告書を書き直すときに使う）。
旧値は報告書の本文の値をそのまま写した（出典は各行のメモ）。
使い方: python kato1014_mapping.py --tag n2000_hard --n1 2000 --neg "hard negative（点数の高い 8 件）"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kato1014_common import OUT, ROOT


def J(name):
    p = OUT / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def ms(d, m="Recall@20", nd=3):
    """{'mean','std'} → '0.957 ± 0.011'."""
    if d is None or d.get(m) is None or d[m].get("mean") is None:
        return "（未計算）"
    v = d[m]
    return f"{v['mean']:.{nd}f} ± {v['std']:.{nd}f}" if v.get("std") is not None else f"{v['mean']:.{nd}f}"


def sd(x, nd=3):
    """符号つきの差。0.0005 未満は 4 桁にする（+0.000・−0.000 と書かない）。負号は U+2212."""
    if x == 0:
        return "0（差なし）"
    d = nd if abs(x) >= 0.0005 else 4
    return f"{x:+.{d}f}".replace("-", "−")


def dp(t, m="Recall@20", nd=3):
    """paired の結果 → '+0.050（10/10、p = 0.00195）'."""
    if t is None or t.get(m) is None:
        return "（未計算）"
    v = t[m]
    p = v.get("p_wilcoxon")
    ps = "—" if p is None else (f"{p:.3g}" if p >= 1e-3 else f"{p:.2e}")
    return f"{sd(v['mean_delta'], nd)}（{v['wins']}/{v['n_seeds']}、p = {ps}）"


def wfmt(w):
    """'w30-70' → '(0.3, 0.7)'."""
    a, b = w[1:].split("-")
    return f"({int(a) / 100:g}, {int(b) / 100:g})"


def summary_text(main, mech, gr, pfi, td, co, n1):
    """結論が変わったところ（数値は JSON から）."""
    out = []
    if main:
        a, b = main["table"]["A"], main["table"]["B"]
        ta, tb = main["tests"]["A"], main["tests"]["B"]
        ka = next(x for x in ta if x.startswith("proposed_static vs baseline_tfidf_w"))
        kb = next(x for x in tb if x.startswith("proposed_static vs baseline_tfidf_w"))
        bname = ka.split(' vs ')[1]
        out.append(f"- **§4**：最良の baseline（重み {wfmt(bname.split('_')[-1])}、R@20 {a[bname]['Recall@20']['mean']:.3f}）に対し、"
                   f"提案手法（一括、N1 = {n1:,}）は R@20 {a['proposed_static']['Recall@20']['mean']:.3f}、MAP "
                   f"{a['proposed_static']['MAP']['mean']:.3f}（A）。R@20 でも {dp(ta[ka])} と大きく上回る"
                   f"（旧版の重み (0.7, 0.3) では R@20 の差がなかった）。B も {dp(tb[kb])}。")
    if mech:
        st = mech["steps"]
        si = st['(i) baseline_w30-70 -> reranker-2']
        out.append(f"- **§5 (i)**：同じ重みの baseline から x1・x2 だけの MLP にすると、上がらずにわずかに下がる（R@20 {dp(si)}、"
                   f"t 検定 p = {si['Recall@20']['p_ttest']:.2g}。MAP {dp(si, 'MAP')}）。旧版の +0.080 は、重み (0.7, 0.3) より良い重みを学習で見つけた分と考えられる"
                   "（旧版の 2 特徴量の MLP の R@K 0.601 は、最良の重みの baseline の 0.607 とほぼ同じ）。最良の重みから始めると「重みの学習」の寄与はない。")
        spec = st["(ii') reranker-2 -> reranker-2+Spec"]
        out.append(f"- **§5 (ii)**：基本の特徴量の向上はほぼ式の特化度 x6（R@20 {dp(spec)}）で、旧版と同じ結論。")
        out.append(f"- **§5 (iii)**：集合の特徴量は R@20 では小さく（{dp(st['(iii) reranker-7 -> reranker-10S'])}）、MAP で大きい"
                   f"（{dp(st['(iii) reranker-7 -> reranker-10S'], 'MAP')}）。R@20 が 0.98 付近で頭打ちのため、効果は正解式をより上位に並べる形で出る。")
    if pfi:
        t = pfi["targets"]
        out.append(f"- **§5 PFI**：変数の 6 特徴量をまとめてシャッフルすると R@20 が {pfi['base']['Recall@20']['mean']:.3f} → "
                   f"{t['GROUP_var']['Recall@20']['shuffled_mean']:.3f}（旧版と同じ結論で、より極端）。話題の 4 特徴量は "
                   f"{t['GROUP_topic']['Recall@20']['shuffled_mean']:.3f}（低下 {sd(t['GROUP_topic']['Recall@20']['drop_mean'])}、"
                   f"p = {t['GROUP_topic']['Recall@20']['p']:.1e}）で、小さいが 0 と有意に異なる（旧版は p = 0.10）。")
    if gr:
        g = gr["tests"]
        out.append(f"- **§6**：一括 < 逐次（推論のみ）< 逐次（訓練も逐次）の順は MAP で保たれる（訓練も逐次 − 一括："
                   f"A {dp(g['A']['all']['train - static'], 'MAP')}、B {dp(g['B']['all']['train - static'], 'MAP')}。"
                   f"正解式 8 本以上で A {dp(g['A']['K>=8']['train - static'], 'MAP')}）。"
                   f"R@20 は天井に近く、A の差は有意でない（{dp(g['A']['all']['train - static'])}）。"
                   f"N1 = {n1:,} で全正解式が候補に入るので、§6 で N1 を 200 に広げた理由と付録 C の N1 の比較は不要になる。")
    if td or mech or pfi:
        txt = "- **§8**："
        if td:
            txt += (f"N1 = {n1:,} の候補の中でも話題の特徴量は正解式を見分ける（AUC：潜在意味類似度 "
                    f"{td['auc_within_candidates']['x3_svd_sim']['mean']:.3f}、文章類似度 {td['auc_within_candidates']['x1_text_sim']['mean']:.3f}。"
                    f"旧版の候補 50 件では 0.594 と 0.437）。候補内の相関も {sd(td['corr_text_jaccard_within_candidates_seed42_test']['mean'])}"
                    "（旧版 −0.404）。旧版の原因の (1) 候補の話題がそろう、(2) 候補の選び方が作る逆相関、は N1 = 50 で生じていたもので、N1 = 2,000 ではほぼ消える。")
        if mech:
            ta_ = mech["topic_additions"]
            x3 = ta_['x3: reranker-2+Svd vs reranker-2']
            txt += (f"それでも x7・x3 を足して訓練し直しても上がらない（x7 {dp(ta_['x7: reranker-2+Dom vs reranker-2'])}。"
                    f"x3 はむしろわずかに下がる：{dp(x3)}、t 検定 p = {x3['Recall@20']['p_ttest']:.2g}）。x10 は R@20 では効かないが MAP をわずかに上げる"
                    f"（{dp(ta_['x10: reranker-7+Dom vs reranker-7'], 'MAP')}）。")
            dm = mech.get("domain_merge", {})
            if dm.get("K40"):
                txt += ("分野ラベルを群にまとめて x7 を群で数える（説明文が群のどれかのラベルを含めば 1）と、粗くするほどむしろ下がる（R@20："
                        + "、".join(f"{ {'L1': '表記をそろえた 1,958 種', 'K100': '100 群', 'K40': '40 群'}[mg]} {dp(dm[mg])}"
                                    for mg in ("L1", "K100", "K40") if dm.get(mg)) + "）"
                        + "。旧版の k-means の 100 群・40 群は、式のラベルを群の名前に置き換えたので x7 がほとんど 1 にならず（1,823 ケース中 5 件）、"
                          "合併の効果を測れていなかった。")
        if pfi:
            c = pfi["cancellation"]["gDom"]["Recall@20"]
            txt += (f"(3) x10 の打ち消し合いは見えなくなった（sep > 0 の {c['n_sep_pos']:,} 件と sep < 0 の {c['n_sep_neg']:,} 件で、"
                    f"Mann–Whitney p = {c['mannwhitney_p']:.2g}。旧版は p = 1.3e-8）。")
        out.append(txt)
    if co:
        o = co["top20"]
        out.append(f"- **§9.1**：N1 = {n1:,} では第 1 段の見逃しがなくなる（種 42：上位 20 件に入らない正解式は {o['n_gt']:,} 本中 {o['missed_total']} 本で、"
                   f"すべて第 2 段の並べ方による。旧版は N1 = 50 で 61.8% が第 1 段）。")
    out.append("- **§7**：自由度ゼロ停止をやめたので、節ごと削る。")
    return "\n".join(out)


def row(sec, item, old, new, note=""):
    return f"| {sec} | {item} | {old} | {new} | {note} |"


def build(tag, n1, neg):
    main = J(f"main_{tag}.json"); mech = J(f"mech_{tag}.json"); gr = J(f"greedy_{tag}.json")
    pfi = J(f"pfi_{tag}.json"); td = J(f"topic_diag_{tag}.json"); co = J(f"case_outcomes_{tag}.json")
    bs = J("baselines_stats.json"); n1j = J("n1_coverage_w30-70.json"); wj = J("weights_n50.json")
    L = []
    L.append("# 8/31 報告の数値：旧値（報告書）→ 新値（2026-10-14 の指示 3・4・5 による再実験）\n")
    L.append("報告書 `docs/progress_report_2026-08-31_methods_results.tex` の本文は変えていない。"
             "この表は、報告書を書き直すときに数値を差し替えるためのものである。\n")
    L.append("## 新しい設定と表の読み方\n")
    L.append(f"- 第 1 段の重みは (0.3, 0.7)（文章類似度 0.3、変数の一致度 0.7）。候補の件数 N1 = {n1:,}"
             "（全正解式が候補に入る最小の N1 を、分割ごとに訓練と検証のケースで求めると 1,233〜1,880。千の位に切り上げると 10 分割すべてで 2,000）。")
    L.append(f"- 訓練の不正解の選び方：{neg}。")
    L.append("- 主指標は Recall@20、あわせて MAP（標準の定義：並べた中にない正解式は 0 として K で割る）と MAP@20（上位 20 件で打ち切り）。旧主指標 Recall@K（R@K）は使わない。")
    L.append("- 自由度ゼロ停止はやめた（報告書の §2.2.4 と §7 は削る）。逐次選択は今の手順のまま 20 本まで選ぶ。")
    L.append("- 値は 10 通りの分割の平均 ± 標準偏差。差の括弧は（上回った分割の数、Wilcoxon の符号順位検定の p）。"
             "§8 の旧値の p は対応のある t 検定なので、新値にも t 検定の p を併記する。")
    L.append("- 出典の JSON は `experiments/kato1014/`。旧値の出典は各行のメモ。\n")
    L.append("## 変えた点（すべて）\n")
    L.append(f"1. 第 1 段の候補の件数 N1：50（逐次選択と自由度ゼロ停止は 200）→ {n1:,}（すべての実験で同じ）。")
    L.append(f"2. 訓練の不正解の選び方：候補から無作為に 8 件 → {neg}。本人の選択（2026-10-15）。"
             "N1 = 2,000 の一括の並べ直しで、10 分割すべてで無作為以上（Recall@20 と MAP）を確かめてから採用した。"
             "この比較には本実験と同じ 10 分割のテストのケースを使ったので、わずかに楽観的な偏りがありうる（検証用のケースでは確かめていない）。")
    L.append("3. 指標：Recall@K → Recall@20（主）、MAP（標準の定義）、MAP@20。")
    L.append("4. baseline の重み：12 通りから R@K で選んだ (0.4, 0.6) → Recall@20 で選んだ (0.3, 0.7)（A・B とも）。提案手法の第 1 段の重みと同じになった。")
    L.append("5. 自由度ゼロ停止をやめた（§2.2.4、§7）。逐次選択は 20 本まで選ぶ手順のまま。")
    L.append("6. 変えていないこと：特徴量 10 個の定義、一括の並べ直しの参照集合（第 1 段の上位 5 件）、MLP の形と学習の定数（マージン 0.1、不正解 8 件、15 エポック）、"
             "層化分割と乱数の種 10 通り。")
    L.append("7. 実装：第 2 段の特徴量をまとめて計算する高速版（`set_aware_reranker.py --fast`、`stage2_fast.py`）で実行した。"
             "既定の計算とビット単位で同じ値を返し、種 42 で既存の実行（N1 = 50 と 200、一括と逐次）の per-case と 423/423 件一致する。\n")
    L.append("## 結論が変わったところ（要約）\n")
    L.append("%%SUMMARY%%\n")

    # ---------------- §1〜§3 ----------------
    L.append("## §1〜§3（設定と手法の定数）\n")
    L.append("| 節 | 項目 | 旧値 | 新値 | メモ |\n|---|---|---|---|---|")
    L.append(row("§1", "§5〜§8 の分析の第 1 段の重み", "(0.7, 0.3)", "(0.3, 0.7)", "§1 の最後の段落と §10 の限界の 1 項目は削る"))
    L.append(row("§2.2・§3.4", "候補の件数 N1", "50（逐次選択と自由度ゼロ停止は 200）", f"{n1:,}（すべての実験で同じ）", "§3.4 の「N1 = 200 に広げる（理由は §6）」は削る"))
    L.append(row("§2.2.4・§7", "自由度ゼロ停止", "あり", "削除", "指示 4"))
    L.append(row("図 1", "提示", "評価：上位 K 件／利用時：自由度ゼロ停止までに選んだ式", "評価：上位 20 件", ""))
    L.append(row("§3.5", "主指標", "Recall@K（K = 正解式の数）", "Recall@20（と MAP）", "被覆率は N1 の説明だけに使う"))
    L.append(row("§3.5", "MAP@20（追加）", "—", "上位 20 件で打ち切った MAP（20 位より後の正解式は P(e) = 0、K で割る）",
                 "LLM 直接生成（最大 20 本）と並べるため（2026-10-15 本人の依頼）。どちらを使うかは後で決める"))
    if bs:
        b = bs["best"]
        L.append(row("§3.4", "baseline の最良の重み（12 通り）", "(0.4, 0.6)（R@K で選択）",
                     f"{wfmt(b['A']['tfidf']['w'])}（A・B とも、Recall@20 で選択）", "baselines_stats.json"))
    L.append(row("§3.4", "訓練の不正解の選び方", "候補から無作為に 8 件", neg, "pilot（種 42）は n1_decision ノート"))
    L.append("")

    # ---------------- §4 ----------------
    if main:
        L.append("## §4 全体比較（表 3）\n")
        L.append("| 節 | 項目 | 旧値 | 新値 | メモ |\n|---|---|---|---|---|")
        old = {"A": {"base": ("0.607 ± 0.043", "0.692 ± 0.046", "0.907 ± 0.016"),
                     "p50": ("0.780 ± 0.048", "0.850 ± 0.038", "0.957 ± 0.011")},
               "B": {"base": ("0.559 ± 0.038", "0.627 ± 0.036", "0.827 ± 0.027"),
                     "p50": ("0.740 ± 0.040", "0.795 ± 0.040", "0.925 ± 0.020")}}
        for S in ("A", "B"):
            T = main["table"][S]
            bb = next(k for k in T if k.startswith("baseline_tfidf_") and k not in ("baseline_tfidf_w40-60", "baseline_tfidf_w70-30"))
            L.append(row(f"表 3（{S}）", f"baseline 最良の重み：旧 (0.4, 0.6) → 新 {wfmt(bb.split('_')[-1])}",
                         f"R@K {old[S]['base'][0]}、MAP {old[S]['base'][1]}、R@20 {old[S]['base'][2]}",
                         f"R@20 {ms(T[bb])}、MAP {ms(T[bb], 'MAP')}、MAP@20 {ms(T[bb], 'MAP@20')}", "external_baselines を per-case 付きで再実行（値は既存の集計と一致）"))
            L.append(row(f"表 3（{S}）", "baseline (0.7, 0.3)", "（表に掲載）", f"R@20 {ms(T['baseline_tfidf_w70-30'])}、MAP {ms(T['baseline_tfidf_w70-30'], 'MAP')}", "値は旧表と同じ（指標を選び直すだけ）"))
            L.append(row(f"表 3（{S}）", "提案手法（一括の並べ直し）(0.3, 0.7)", f"R@K {old[S]['p50'][0]}、MAP {old[S]['p50'][1]}、R@20 {old[S]['p50'][2]}（N1 = 50）",
                         f"R@20 {ms(T.get('proposed_static'))}、MAP {ms(T.get('proposed_static'), 'MAP')}、MAP@20 {ms(T.get('proposed_static'), 'MAP@20')}（N1 = {n1:,}）", ""))
            L.append(row(f"表 3（{S}）", "提案手法（逐次選択・訓練も逐次）", "（表になし）",
                         f"R@20 {ms(T.get('proposed_train'))}、MAP {ms(T.get('proposed_train'), 'MAP')}、MAP@20 {ms(T.get('proposed_train'), 'MAP@20')}", "指示「一括の並べ直し、逐次選択の 2 通り」"))
            L.append(row(f"表 3（{S}）", "提案手法（逐次選択・推論のみ、参考）", "（表になし）",
                         f"R@20 {ms(T.get('proposed_infer'))}、MAP {ms(T.get('proposed_infer'), 'MAP')}、MAP@20 {ms(T.get('proposed_infer'), 'MAP@20')}", "§6 の比較用"))
            L.append(row(f"表 3（{S}）", "提案手法 (0.7, 0.3)、N1 = 50 の行", "R@K 0.743、MAP 0.796、R@20 0.909" if S == "A" else "R@K 0.689、MAP 0.715、R@20 0.838",
                         "削る", "重みの比較は §9.3（weights_n50.json）"))
            tests = main["tests"][S]
            k = next((x for x in tests if x.startswith("proposed_static vs baseline_tfidf_w")), None)
            oldtxt = ("R@K +0.173（(0.3, 0.7)）。重み (0.7, 0.3) では R@20 0.909 と 0.907 で差なし（p = 0.49）" if S == "A"
                      else "R@K +0.181（(0.3, 0.7)）。重み (0.7, 0.3) では R@20 0.838 と 0.827（p = 0.049）")
            L.append(row(f"§4 本文（{S}）", "提案手法（一括）− 最良の baseline", oldtxt,
                         f"R@20 {dp(tests.get(k))}、MAP {dp(tests.get(k), 'MAP')}、MAP@20 {dp(tests.get(k), 'MAP@20')}" if k else "（未計算）", ""))
            k2 = next((x for x in tests if x.startswith("proposed_train vs baseline_tfidf_w")), None)
            if k2:
                L.append(row(f"§4 本文（{S}）", "逐次選択（訓練も逐次）− 最良の baseline", "—", f"R@20 {dp(tests[k2])}、MAP {dp(tests[k2], 'MAP')}、MAP@20 {dp(tests[k2], 'MAP@20')}", ""))
            k3 = "proposed_static vs proposed_static_N1_50"
            L.append(row(f"§4 本文（{S}）", "N1 = 2,000 − N1 = 50（どちらも重み (0.3, 0.7)）", "—",
                         f"R@20 {dp(tests.get(k3))}、MAP {dp(tests.get(k3), 'MAP')}", "N1 = 50 は無作為の不正解（旧の手順）なので、N1 と不正解の選び方の両方の差"))
            L.append(row(f"§4 本文（{S}）", "被覆率（候補に入る正解式の割合）", "92.0%（(0.7, 0.3)）/ 96.7%（(0.3, 0.7)）、N1 = 50" if S == "A" else "—",
                         f"{ms(T.get('proposed_static'), 'coverage')}（テストのケース）", "N1 = 2,000 では全分割で 1.000"))
        L.append("")
        # 付録 A の BM25・E5
        L.append("### 付録 A（BM25・E5）\n")
        L.append("| 節 | 項目 | 旧値 | 新値 | メモ |\n|---|---|---|---|---|")
        for S in ("A", "B"):
            T = main["table"][S]; tests = main["tests"][S]
            for sc, oldv in (("bm25", "0.601（A）/ 0.555（B）、重み (0.1, 0.9)"), ("e5", "0.602（A）/ 0.544（B）、重み (0.1, 0.9)")):
                g = next(x for x in T if x.startswith(f"baseline_{sc}_"))
                L.append(row(f"付録 A（{S}）", f"{sc.upper()} の最良の重み {wfmt(g.split('_')[-1])}", f"R@K {oldv}",
                             f"R@20 {ms(T[g])}、MAP {ms(T[g], 'MAP')}、MAP@20 {ms(T[g], 'MAP@20')}", "Recall@20 で重みを選び直した"))
                k = f"proposed_static vs {g}"
                L.append(row(f"付録 A（{S}）", f"提案手法 − {sc.upper()}", "R@K で全分割で上回る（p = 0.00195）",
                             f"R@20 {dp(tests.get(k))}、MAP {dp(tests.get(k), 'MAP')}、MAP@20 {dp(tests.get(k), 'MAP@20')}", ""))
        if bs:
            L.append("")
            L.append("付録 A の表 6（12 通りの重み）は `baselines_stats.json` の `weights` に Recall@20・MAP を出した"
                     "（R@K の列は削る）。文章類似度だけ (1.0, 0.0) の MAP は標準の定義で 0.147（A）/ 0.170（B）。")
        L.append("")
        # 付録 B
        L.append("### §4 の段落と付録 B（正解式の数・入力変数の数で分けた差、設定 A）\n")
        bk = main["buckets"]["A"]
        bbname = next(k.split("|")[0] for k in next(iter(bk["n_correct"].values())) if k.startswith("baseline_tfidf_w") and "w70-30" not in k)
        L.append("| 層 | 件数 | 提案（一括）R@20 | baseline 最良 R@20 | 差 R@20 | 提案 MAP | baseline MAP | 差 MAP | 旧値（R@K の差、対 (0.4, 0.6)） |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        oldK = {"1": "−0.012", "2": "+0.154", "3": "+0.161", "4": "+0.151", "5": "+0.182", "6": "+0.146", "7": "+0.133",
                "8": "+0.142", "9": "+0.138", "10": "+0.104"}
        oldI = {"<=5": "+0.021", "6-10": "+0.088", "11-15": "+0.193", "16-20": "+0.192", ">=21": "+0.163"}
        for key, olds, order in (("n_correct", oldK, [str(i) for i in range(1, 11)]), ("n_input", oldI, ["<=5", "6-10", "11-15", "16-20", ">=21"])):
            for b in order:
                c = bk[key].get(b)
                if not c:
                    continue
                pr, br = c.get("proposed_static|Recall@20"), c.get(f"{bbname}|Recall@20")
                pm, bm = c.get("proposed_static|MAP"), c.get(f"{bbname}|MAP")
                lab = (f"正解式 {b} 本" if key == "n_correct" else f"入力変数 {b} 個")
                L.append(f"| {lab} | {c['n_records']:,} | {pr:.3f} | {br:.3f} | {sd(pr - br)} | {pm:.3f} | {bm:.3f} | {sd(pm - bm)} | {olds[b]} |")
        L.append("")

    # ---------------- §5 ----------------
    if mech:
        L.append("## §5 向上の内訳（設定 A）\n")
        L.append("| 節 | 項目 | 旧値（R@K、N1 = 50、(0.7, 0.3)） | 新値 R@20 | 新値 MAP | メモ |\n|---|---|---|---|---|---|")
        S_ = mech["summary"]; st = mech["steps"]
        def r2(name, key):
            return f"{S_[name]['Recall@20']['mean']:.3f}" if name in S_ else "—"
        L.append(f"| §5 (i) | baseline（同じ重み）→ 2 特徴量の MLP | 0.521 → 0.601（+0.080） | {r2('baseline_w30-70','')} → {r2('reranker-2','')}、{dp(st['(i) baseline_w30-70 -> reranker-2'])} | {dp(st['(i) baseline_w30-70 -> reranker-2'], 'MAP')} | |")
        L.append(f"| §5 (ii) | 2 → 7 特徴量 | 0.601 → 0.693（+0.092） | {r2('reranker-2','')} → {r2('reranker-7','')}、{dp(st['(ii) reranker-2 -> reranker-7'])} | {dp(st['(ii) reranker-2 -> reranker-7'], 'MAP')} | |")
        spec = st["(ii') reranker-2 -> reranker-2+Spec"]
        L.append(f"| §5 (ii) | 2 特徴量 ＋ 式の特化度 x6 だけ | 0.691（+0.090、p = 0.00195） | {r2('reranker-2_Spec','')}、{dp(spec)} | {dp(spec, 'MAP')} | |")
        L.append(f"| §5 (iii) | 7 → 10 特徴量（集合の 3 個） | 0.693 → 0.743（+0.050、p = 0.00195） | {r2('reranker-7','')} → {r2('reranker-10S','')}、{dp(st['(iii) reranker-7 -> reranker-10S'])} | {dp(st['(iii) reranker-7 -> reranker-10S'], 'MAP')} | |")
        for x, o in (("reranker-7_Comp", "+0.052（p = 0.00195）"), ("reranker-7_Coh", "+0.050（p = 0.00195）"), ("reranker-7_Dom", "−0.0001（t 検定 p = 0.98）")):
            t = mech["single_set_additions"].get(x)
            tp = "" if not t else f"（t 検定 p = {t['Recall@20']['p_ttest']:.2g}）"
            nm = {"reranker-7_Comp": "x8（補完性）", "reranker-7_Coh": "x9（一貫性）", "reranker-7_Dom": "x10（分野ラベルの一致度（集合））"}[x]
            L.append(f"| §5 | 7 特徴量 ＋ {nm} − 7 特徴量 | {o} | {dp(t)}{tp} | {dp(t, 'MAP')} | |")
        for key, lab, olds in (("lift_by_n_correct", "集合の特徴量の上乗せ（正解式の数別）", "1 本 +0.002、3 本 +0.102（最大）"),
                               ("lift_by_n_input", "集合の特徴量の上乗せ（入力変数の数別）", "11〜15 個 +0.078、16 個以上 +0.035")):
            rows = mech[key].get("reranker-10S", {}).get("Recall@20", [])
            txt = "、".join(f"{r['bin']} {sd(r['lift_mean'])}" for r in rows if r.get("n_records"))
            rows_m = mech[key].get("reranker-10S", {}).get("MAP", [])
            txt_m = "、".join(f"{r['bin']} {sd(r['lift_mean'])}" for r in rows_m if r.get("n_records"))
            L.append(f"| §5 | {lab} | {olds} | {txt} | {txt_m} | 上乗せ = reranker-10S − reranker-7（層ごと、分割で平均） |")
        L.append("")
    if pfi:
        L.append("### §5 の PFI（図 2）\n")
        L.append("| 節 | 項目 | 旧値（R@K、N1 = 50） | 新値 R@20 | 新値 MAP | メモ |\n|---|---|---|---|---|---|")
        b = pfi["base"]; t = pfi["targets"]
        L.append(f"| §5 PFI | シャッフル前 | 0.743 | {b['Recall@20']['mean']:.3f} | {b['MAP']['mean']:.3f} | |")
        for lab, nm, old in (("GROUP_var", "変数の 6 特徴量をまとめて", "0.065（低下 0.678、p < 1e-4）"),
                             ("specificity", "式の特化度 x6 だけ", "0.126"),
                             ("GROUP_topic", "話題の 4 特徴量をまとめて", "0.737（低下 0.006、p = 0.10）"),
                             ("gDom", "x10 だけ", "低下 −0.0003")):
            v, m2 = t[lab]["Recall@20"], t[lab]["MAP"]
            L.append(f"| §5 PFI | {nm} | {old} | {v['shuffled_mean']:.3f}（低下 {sd(v['drop_mean'])}、p = {v['p']:.2g}） | "
                     f"{m2['shuffled_mean']:.3f}（低下 {sd(m2['drop_mean'])}、p = {m2['p']:.2g}） | |")
        L.append(f"| 図 2 | fig_pfi_recall_ja.png | 旧図（R@K） | docs/figures/fig_pfi_recall20_ja_{tag}.png | | 旧図は上書きしていない |")
        L.append("")

    # ---------------- §6 ----------------
    if gr:
        L.append("## §6 逐次選択（付録 C）\n")
        L.append("| 節 | 項目 | 旧値（R@K、N1 = 200、(0.7, 0.3)） | 新値 R@20 | 新値 MAP | メモ |\n|---|---|---|---|---|---|")
        oldS = {"B": "(1) 0.724、(2) 0.756、(3) 0.765", "A": "(1) 0.757、(2) 0.787、(3) 0.797"}
        oldT = {"B": {"infer - static": "+0.032（p = 0.00195）", "train - infer": "+0.009（p = 0.037）", "train - static": "+0.041（p = 0.00195）"},
                "A": {"infer - static": "+0.030（p = 0.00195）", "train - infer": "+0.010（p = 0.13）", "train - static": "+0.040（p = 0.00195）"}}
        oldT8 = {"B": {"train - static": "+0.056（p = 0.00195）", "train - infer": "+0.013（p = 0.020）"},
                 "A": {"train - static": "+0.048（p = 0.00195）", "train - infer": "—"}}
        for S in ("B", "A"):
            s = gr["summary"][S]["all"]
            val = "、".join(f"({i}) {s[k]['Recall@20']['mean']:.3f}" for i, k in ((1, "static"), (2, "infer"), (3, "train")) if k in s)
            valm = "、".join(f"({i}) {s[k]['MAP']['mean']:.3f}" for i, k in ((1, "static"), (2, "infer"), (3, "train")) if k in s)
            L.append(f"| §6（{S}） | (1) 一括、(2) 逐次（推論のみ）、(3) 逐次（訓練も逐次） | {oldS[S]} | {val} | {valm} | |")
            lab6 = {"infer - static": "(2) − (1)", "train - infer": "(3) − (2)", "train - static": "(3) − (1)"}
            for k, o in oldT[S].items():
                tt = gr["tests"][S]["all"].get(k)
                L.append(f"| §6（{S}） | {lab6[k]} | {o} | {dp(tt)} | {dp(tt, 'MAP')} | |")
            for k, o in oldT8[S].items():
                tt = gr["tests"][S]["K>=8"].get(k)
                L.append(f"| §6（{S}） | {lab6[k]}（正解式 8 本以上） | {o} | {dp(tt)} | {dp(tt, 'MAP')} | |")
        L.append(f"| §6 | N1 を 200 に広げた理由（N1 = 50 で 8.0%、200 で 2.2% が候補外） | 8.0% / 2.2% | N1 = {n1:,} で 0%（全正解式が候補に入る） | | 理由の段落は削る |")
        L.append(f"| 付録 C | N1 = 50 と 200 の表 | 表 8 | N1 = {n1:,} の 1 通りにまとめる（上の §6 の値） | | |")
        L.append("")

    # ---------------- §7 ----------------
    L.append("## §7 自由度ゼロ停止\n")
    L.append("完全一致率 0.526 / 0.368、集合 F1 の差 −0.008 / −0.023、充足率 0.936 / 0.883 と 0.865 / 0.781、"
             "条件を満たさなかった 6.4% / 11.7%、止まる位置の誤りなど、§7 の数値はすべて削る（指示 4）。"
             "§1 の (4)、§2.2.4、§3.5 の最後の段落、§10 の該当項目、§11 の該当文も削る。\n")

    # ---------------- §8 ----------------
    if mech:
        L.append("## §8 話題の特徴量（設定 A）\n")
        L.append("| 節 | 項目 | 旧値（R@K、N1 = 50） | 新値 R@20 | 新値 MAP | メモ |\n|---|---|---|---|---|---|")
        for k, lab, o in (("x7: reranker-2+Dom vs reranker-2", "x1・x2 ＋ x7（分野ラベルの一致）− x1・x2", "+0.002（t 検定 p = 0.06）"),
                          ("x3: reranker-2+Svd vs reranker-2", "x1・x2 ＋ x3（潜在意味類似度）− x1・x2", "−0.002（t 検定 p = 0.48）"),
                          ("x10: reranker-7+Dom vs reranker-7", "7 特徴量 ＋ x10 − 7 特徴量", "−0.0001（t 検定 p = 0.98）")):
            t = mech["topic_additions"].get(k)
            tp = "" if not t else f"（t 検定 p = {t['Recall@20']['p_ttest']:.2g}）"
            tpm = "" if not t else f"（t 検定 p = {t['MAP']['p_ttest']:.2g}）"
            L.append(f"| §8 | {lab} | {o} | {dp(t)}{tp} | {dp(t, 'MAP')}{tpm} | |")
        chk = mech["domain_merge"].get("orig_equals_reranker-2_Dom")
        if chk:
            L.append(f"| §8 | 確認：群で数える x7 を合併なしで使う | — | per-case の順位が x1・x2 ＋ x7 の実行と {chk['identical_ranks']:,}/{chk['n']:,} 件一致 | | 群で数える x7 は元の x7 と同じ値 |")
        for mg in ("L1", "K100", "K40"):
            t = mech["domain_merge"].get(mg)
            tp = "" if not t else f"（t 検定 p = {t['Recall@20']['p_ttest']:.2g}）"
            mgl = {"L1": "表記をそろえた 1,958 種", "K100": "k-means で 100 群", "K40": "k-means で 40 群"}[mg]
            L.append(f"| §8 | 分野ラベルをまとめる（{mgl}）：x1・x2 ＋ x7 − x1・x2 | 最大 +0.0006、すべて p > 0.6 | {dp(t)}{tp} | {dp(t, 'MAP')} | "
                     + ("旧版の 100 群・40 群は x7 がほぼ常に 0 で、合併の効果を測れていなかった（新版は群で数える）" if mg != "L1" else "新版は群で数える x7") + " |")
        L.append("")
    if td:
        L.append(f"### §8 の表 4（AUC、1,823 ケースの平均 ± 標準誤差）\n")
        L.append(f"| 特徴量 | 旧値 全 DB | 旧値 候補 50 件 | 新値 全 DB | 新値 候補 {n1:,} 件 |\n|---|---|---|---|---|")
        old = {"x2_io_jaccard": ("変数の一致度 x2", "0.999 ± 0.000", "0.947 ± 0.002"),
               "x6_specificity": ("式の特化度 x6", "—", "0.965 ± 0.001"),
               "x3_svd_sim": ("潜在意味類似度 x3", "0.950 ± 0.003", "0.594 ± 0.006"),
               "x1_text_sim": ("文章類似度 x1", "0.913 ± 0.003", "0.437 ± 0.007")}
        for f, (nm, o1, o2) in old.items():
            a, c = td["auc_full_db"][f], td["auc_within_candidates"][f]
            L.append(f"| {nm} | {o1} | {o2} | {a['mean']:.3f} ± {a['sem']:.3f} | {c['mean']:.3f} ± {c['sem']:.3f} |")
        c = td["corr_text_jaccard_within_candidates_seed42_test"]
        L.append(f"\n候補内の相関 corr(x1, x2)（種 42 の分割のテスト {td['n_test_seed42']} ケース）：旧値 −0.404（N1 = 50）→ 新値 {sd(c['mean'])}（N1 = {n1:,}）。"
                 "残りの 6 特徴量の AUC も JSON にある。重み (0.7, 0.3)・N1 = 50 で回すと旧値をすべて再現する（topic_diag_check_w70-30_k50.json）。\n")
    if pfi:
        c = pfi["cancellation"]["gDom"]["Recall@20"]; cm_ = pfi["cancellation"]["gDom"]["MAP"]
        L.append("### §8 (3) x10 の PFI の打ち消し合い（ケース単位に集約してから検定）\n")
        L.append("| 項目 | 旧値（R@K） | 新値 R@20 | 新値 MAP |\n|---|---|---|---|")
        L.append(f"| 延べ件数 / 異なるケース | 4,594 / 1,592 | {c['n_records']:,} / {c['n_cases']:,} | 同左 |")
        L.append(f"| sep > 0 のケース（低下量の平均） | 1,169 件（0.012） | {c['n_sep_pos']:,} 件（{sd(c['mean_drop_sep_pos'], 4)}） | {cm_['n_sep_pos']:,} 件（{sd(cm_['mean_drop_sep_pos'], 4)}） |")
        L.append(f"| sep < 0 のケース（低下量の平均） | 414 件（−0.034） | {c['n_sep_neg']:,} 件（{sd(c['mean_drop_sep_neg'], 4)}） | {cm_['n_sep_neg']:,} 件（{sd(cm_['mean_drop_sep_neg'], 4)}） |")
        L.append(f"| Mann–Whitney の U 検定 | p = 1.3e-8 | p = {c['mannwhitney_p']:.2g} | p = {cm_['mannwhitney_p']:.2g} |")
        L.append(f"| 2 群の重み付き平均 / 全体の PFI | −0.0002 / −0.0003 | {sd(c.get('weighted_mean_drop_pos_neg', float('nan')), 4)} / {sd(pfi['targets']['gDom']['Recall@20']['drop_mean'], 4)} | {sd(cm_.get('weighted_mean_drop_pos_neg', float('nan')), 4)} / {sd(pfi['targets']['gDom']['MAP']['drop_mean'], 4)} |")
        L.append("")

    # ---------------- §9 ----------------
    if co:
        L.append("## §9.1 誤りの位置（種 42、テスト 423 ケース、正解式 1,579 本）\n")
        o20, oK = co["top20"], co["topK"]
        L.append("| 項目 | 旧値（上位 K 件、N1 = 50 / 200、(0.7, 0.3)） | 新値（上位 20 件） | 新値（上位 K 件、参考） |\n|---|---|---|---|")
        L.append(f"| 漏れた正解式 | 435 本 / 373 本 | {o20['missed_total']} 本 | {oK['missed_total']} 本 |")
        L.append(f"| うち第 1 段の見逃し（候補外） | 269 本（61.8%）/ 106 本 | {o20['stage1_miss']} 本 | {oK['stage1_miss']} 本 |")
        L.append(f"| うち第 2 段の見逃し（候補内） | 166 本（38.2%）/ 267 本（71.6%） | {o20['stage2_miss']} 本 | {oK['stage2_miss']} 本 |")
        L.append(f"| 全正解式が入ったケース / 漏れのあるケース | 212 / 211 | {o20['solved']} / {o20['unsolved']} | {oK['solved']} / {oK['unsolved']} |")
        for a, lab, old in (("n_correct", "正解式の数の平均", "2.13 / 5.34（p < 1e-14）"), ("n_input", "入力変数の数の平均", "10.8 / 17.8（p < 1e-14）"),
                            ("coverage", "被覆率", "1.00 / 0.83")):
            v, w = o20["solved_vs_unsolved"][a], oK["solved_vs_unsolved"][a]
            L.append(f"| {lab}（全部 / 漏れあり） | {old} | {v['mean_solved']:.2f} / {v['mean_unsolved']:.2f}（p = {v['p']:.2g}） | {w['mean_solved']:.2f} / {w['mean_unsolved']:.2f}（p = {w['p']:.2g}） |")
        for f, nm, old in (("svd_sim", "潜在意味類似度 x3", "0.516"), ("io_jaccard", "変数の一致度 x2", "0.375")):
            v, w = o20["rescue_fraction"][f], oK["rescue_fraction"][f]
            L.append(f"| 漏れた正解式の値 > 上位の不正解式の値 となる組の割合（{nm}） | {old} | {v['mean']:.3f}（n = {v['n']} ケース） | {w['mean']:.3f}（n = {w['n']} ケース） |")
        L.append("")
    if wj:
        L.append("## §9.3 第 1 段の重みの見直し（N1 = 50 のまま、既存の実行を Recall@20 と MAP で数え直した）\n")
        L.append("| 項目 | 旧値（R@K） | 新値 R@20 | 新値 MAP |\n|---|---|---|---|")
        for w in ("w70-30", "w50-50", "w30-70", "w00-100"):
            s = wj["static"][w]
            L.append(f"| 提案手法 重み {wfmt(w)}（被覆率 {s['coverage']['mean']:.3f}、テスト） | {s['Recall@K']['mean']:.3f} | {ms(s)} | {ms(s, 'MAP')} |")
        t = wj["static_vs_w70-30"]["w30-70"]
        L.append(f"| (0.3, 0.7) − (0.7, 0.3) | +0.037（p = 0.00195） | {dp(t)} | {dp(t, 'MAP')} |")
        t8 = wj["static_vs_w70-30_K>=8"]["w30-70"]
        L.append(f"| 同、正解式 8 本以上 | +0.094 | {dp(t8)} | {dp(t8, 'MAP')} |")
        tg = wj["train_greedy_N1_200_w30-70_vs_w70-30"]
        L.append(f"| 逐次選択（訓練も逐次）N1 = 200 の (0.3, 0.7) − (0.7, 0.3) | +0.013（p = 0.0098） | {dp(tg)} | {dp(tg, 'MAP')} |")
        bw = J("n1_full_by_weight.json")
        if bw:
            ws = bw["weights"]
            L.append("\n§9.3 は重みを選んだ根拠なので N1 = 50（と 200）のままでよい。全正解式が候補に入る N1（全ケース）は、"
                     f"重み (0.7, 0.3) で {ws['w70-30']['A']['n1_all_cases']:,}、(0.5, 0.5) で {ws['w50-50']['A']['n1_all_cases']:,}、"
                     f"(0.3, 0.7) で {ws['w30-70']['A']['n1_all_cases']:,}、(0.0, 1.0) で {ws['w00-100']['A']['n1_all_cases']:,} であり、"
                     "(0.3, 0.7) が最も小さい（n1_full_by_weight.json）。N1 = 2,000 での被覆率（設定 A）は順に "
                     + "、".join(f"{ws[w]['A']['at_2000']['coverage']:.4f}" for w in ("w70-30", "w50-50", "w30-70", "w00-100")) + "。\n")
    if n1j:
        L.append("## 指示 3 の結果（新しい節に入れる候補）\n")
        a, b = n1j["settings"]["A"], n1j["settings"]["B"]
        L.append(f"- 全ケースで全正解式が候補に入る N1：A {a['max_min_n1']:,}、B {b['max_min_n1']:,}（点数 0 の正解式はない）。"
                 f"99% のケース：A {a['n1_for_share']['0.99']}、B {b['n1_for_share']['0.99']}。")
        tail = [q for q in n1j["per_case"] if q["min_n1"] > a["n1_for_share"]["0.99"]]
        worst = [max(q["gt"], key=lambda g: g["rank_worst"]) for q in tail]
        fams = sorted({q["family"] for q in tail})
        L.append(f"- 全正解式を入れるのに N1 > {a['n1_for_share']['0.99']} が要る {len(tail)} ケース（設定 A の 1%）は"
                 f"{'すべて合成' if fams == ['synthesized'] else '、'.join(fams)}で（単一出典は N1 = "
                 f"{n1j['families']['single_source']['max_min_n1']}、出典横断は {n1j['families']['cross_source']['max_min_n1']} で全部入る）、"
                 f"正解式は {min(q['K'] for q in tail)}〜{max(q['K'] for q in tail)} 本、入出力変数は "
                 f"{min(q['n_io'] for q in tail)}〜{max(q['n_io'] for q in tail)} 個。最も下位の正解式は変数が "
                 f"{min(w['n_eq_vars'] for w in worst)}〜{max(w['n_eq_vars'] for w in worst)} 個で、変数の一致度は "
                 f"{min(w['var_jaccard'] for w in worst):.3f}〜{max(w['var_jaccard'] for w in worst):.3f}、文章類似度は "
                 f"{min(w['text_sim'] for w in worst):.3f}〜{max(w['text_sim'] for w in worst):.3f} と小さい"
                 "（T = 470.21°C のような数値例の式や、Routh の安定条件の不等式など）。")
        L.append("- 図：docs/figures/fig_n1_full_coverage_ja.png（JSON は n1_coverage_w30-70.json）。")
    md = "\n".join(L) + "\n"
    return md.replace("%%SUMMARY%%", summary_text(main, mech, gr, pfi, td, co, n1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n1", type=int, required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--out", default="docs/加藤先生指示_2026-10-14_旧値新値対応表.md")
    a = ap.parse_args()
    md = build(a.tag, a.n1, a.neg)
    p = ROOT / a.out
    p.write_text(md, encoding="utf-8")
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
