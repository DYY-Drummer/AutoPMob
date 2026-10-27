"""2026-10-21（3 回目）：出典横断の 113 件の説明文を、入出力変数に合うように書き直した training_cases.json（e995ca0）と、
外す一覧 v3（92 件＝v1 の 91 件＋core_019_v1、b7f7cd2、設定 A は 1,731 件）で、c7c4feb＋ed41231 と同じ実験と集計を
やり直した結果の、旧値 → 新値の対応表.

旧値：報告書 docs/progress_report_2026-08-31_methods_results.tex（db21007）の数値。experiments/kato1020/ から、新値と
      同じ関数で計算し直し、その数値が報告書の本文に現れることを確かめる（現れないものは文書の末尾に書き出す）。
新値：experiments/kato1021/（kato1020_*.py を KATO_OUT=experiments/kato1021 で実行した結果）。
出力：docs/加藤先生指示_2026-10-21_説明文書き直しと92件除外_旧値新値対応表.md
使い方：python kato1021_mapping.py [--report <報告書の .tex>]（既定は git show db21007:<報告書>）
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from kato1020_mapping import f3, row, sd, wfmt

ROOT = Path(__file__).parent
K20 = ROOT / "experiments" / "kato1020"
K21 = ROOT / "experiments" / "kato1021"
OUTMD = ROOT / "docs" / "加藤先生指示_2026-10-21_説明文書き直しと92件除外_旧値新値対応表.md"
REPORT_REV, REPORT = "db21007", "docs/progress_report_2026-08-31_methods_results.tex"
M4 = ("Recall@20", "R-Precision", "MAP", "MAP@20")
SEEDS = ["42", "123", "456", "789", "1024", "2024", "3141", "5926", "7777", "9999"]
WEIGHTS4 = ("w70-30", "w50-50", "w30-70", "w00-100")
TOK = re.compile(r"[+\-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+\-−]?\d+\.\d+")


class Run:
    """1 回分の結果（旧：kato1020、新：kato1021）."""

    def __init__(self, d: Path, counts_key: str, exclude: Path):
        self.d = d

        def j(n):
            p = d / n
            return json.load(open(p, encoding="utf-8")) if p.exists() else None
        self.main, self.greedy, self.mech, self.pfi, self.sweep = (j(f"{x}.json") for x in ("main", "greedy", "mech", "pfi", "sweep"))
        self.sel, self.bl, self.rule = j("selected.json"), j("baselines.json"), j("n1_rule.json")
        self.td, self.co, self.cov = j("topic_diag_final.json"), j("case_outcomes_final.json"), j("n1_coverage_w30-70.json")
        self.cnt = json.load(open(K21 / "data_counts.json", encoding="utf-8"))[counts_key]
        self.ex = json.load(open(exclude, encoding="utf-8"))

    def n1_run(self, S, w, neg="hard"):
        """select の実行で実際に使った N1（分割ごと）."""
        out = []
        for f in glob.glob(str(self.d / "raw" / "select" / f"{S}_{w}_{neg}__*.json")):
            out.append(int(json.load(open(f, encoding="utf-8"))["config"]["top_k"]))
        return out


# ---------------------------------------------------------------------------
# 書式
# ---------------------------------------------------------------------------

def mean(v):
    return f3(v["mean"])


def pm(v):
    return f"{v['mean']:.3f} ± {v['std']:.3f}"


def wins(c, met="Recall@20"):
    return f"{c[met]['wins']}/{c[met]['n']}"


def sdx(x):
    """符号付きの差。絶対値が 0.001 より小さいときは小数第 4 位まで（報告書の −0.0005 と同じ書き方）."""
    if x is not None and x != 0 and abs(x) < 0.001:
        return f"{x:+.4f}".replace("-", "−")
    return sd(x)


def dw(c, met="Recall@20", fmt=sd):
    """差（上回った分割の数）。fmt は報告書の桁数に合わせる（x10 の Recall@20 だけ sdx で小数第 4 位）."""
    return f"{fmt(c[met]['mean_delta'])}（{wins(c, met)}）"


def rng(vals, fmt=lambda x: f"{x:,}"):
    v = sorted(vals)
    return fmt(v[0]) if fmt(v[0]) == fmt(v[-1]) else f"{fmt(v[0])}〜{fmt(v[-1])}"


def wcount(ws):
    c = Counter(ws)
    return "、".join(f"{wfmt(w)} {n}" for w, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def norm_report(t):
    t = t.replace("{,}", ",").replace("$", "").replace("\\pm", "±").replace("\\%", "%").replace("−", "-")
    return re.sub(r"\s+", " ", t)


# ---------------------------------------------------------------------------
# 文書
# ---------------------------------------------------------------------------

class Doc:
    def __init__(self, o: Run, n: Run, report_text: str):
        self.o, self.n, self.rep = o, n, norm_report(report_text)
        self.L, self.miss = [], []
        self.sec = ""

    def h(self, t):
        self.sec = t
        self.L.append(f"\n## {t}\n")

    def h3(self, t):
        self.L.append(f"\n### {t}\n")

    def p(self, t):
        self.L.append(t + "\n")

    def head(self, cols=("項目", "旧値（報告書 db21007）", "新値", "")):
        self.L.append(row(*cols))
        self.L.append(row(*["---"] * len(cols)))

    def val(self, f, r):
        try:
            return f(r)
        except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
            return "（未計算）"

    def tr(self, item, f, note=None, chk=None, old=None):
        """1 行。f(run) で旧値と新値を同じ関数で作る。old を与えると旧値はその文字列（報告書の文言そのもの）。"""
        ov = old if old is not None else self.val(f, self.o)
        nv = self.val(f, self.n)
        self.check(item, ov, chk)
        if note is None:
            note = "同じ" if ov == nv else ""
        self.L.append(row(item, ov, nv, note))
        return ov, nv

    def check(self, item, ov, chk=None):
        toks = TOK.findall(ov) if chk is None else chk
        for t in toks:
            t2 = t.replace("−", "-")
            # 符号付きの数（差）は符号まで一致を求める。符号のない数は、そのまま探す
            if not re.search(r"(?<![\d.])" + re.escape(t2) + r"(?![\d])", self.rep):
                self.miss.append((self.sec, item, t))


def claims(o: Run, n: Run):
    """結論（勝ち数と主張）を旧と新で比べる。(節, 内容, 旧, 新)."""
    C = []

    def add(sec, lab, f):
        def g(r):
            try:
                return f(r)
            except (KeyError, TypeError, IndexError, ValueError):
                return "（未計算）"
        C.append((sec, lab, g(o), g(n)))

    def minw(r, S, a, bs, mets=M4):
        cm = r.main["compare"][S]
        return min(cm[f"{a} vs {b}"][m]["wins"] for b in bs for m in mets)

    for S in ("A", "B"):
        add("§4", f"設定 {S}：提案（一括）と提案（訓練も逐次）が baseline（TF-IDF）を上回った分割の数（4 指標の最小）",
            lambda r, S=S: f"{min(minw(r, S, 'proposed_static', ['baseline_tfidf']), minw(r, S, 'proposed_train', ['baseline_tfidf']))}/10")
        add("§4", f"設定 {S}：提案（一括）が BM25・E5 を上回った分割の数（4 指標の最小）",
            lambda r, S=S: f"{minw(r, S, 'proposed_static', ['baseline_bm25', 'baseline_e5'])}/10")
    add("§4", "設定 A：テストの被覆率（N1 = 2,000、10 分割の平均）", lambda r: f"{r.sweep['N1']['A']['2000']['coverage']['mean']:.4f}")

    def bucket_gap(r, k):
        b = r.main["buckets"]["A"]["n_correct"][k]
        return b["proposed_static|Recall@20"] - b["baseline_tfidf|Recall@20"]
    add("§4", "設定 A：Recall@20 の差（一括 − baseline）が正解式 2・5・10 本の順に大きい",
        lambda r: "はい" if bucket_gap(r, "2") < bucket_gap(r, "5") < bucket_gap(r, "10") else "いいえ")
    st = lambda r, k: r.mech["steps"][k]  # noqa: E731
    add("§5", "(i) x1・x2 だけの MLP が同じ重みの baseline を上回った分割（Recall@20／MAP）",
        lambda r: f"{wins(st(r, '(i) baseline_same_w -> reranker-2'))}／{wins(st(r, '(i) baseline_same_w -> reranker-2'), 'MAP')}")
    add("§5", "(ii) 2 → 7 特徴量（Recall@20／MAP）",
        lambda r: f"{wins(st(r, '(ii) reranker-2 -> reranker-7'))}／{wins(st(r, '(ii) reranker-2 -> reranker-7'), 'MAP')}")
    k2s = "(ii') reranker-2 -> reranker-2+Spec"
    add("§5", "(ii) 2 特徴量 ＋ x6（Recall@20／MAP）", lambda r: f"{wins(st(r, k2s))}／{wins(st(r, k2s), 'MAP')}")
    add("§5", "(iii) 7 → 10 特徴量（Recall@20／MAP）",
        lambda r: f"{wins(st(r, '(iii) reranker-7 -> reranker-10S'))}／{wins(st(r, '(iii) reranker-7 -> reranker-10S'), 'MAP')}")
    add("§5", "集合の特徴量を 1 個ずつ加えたときの MAP（x8／x9／x10 で上回った分割）",
        lambda r: "／".join(wins(r.mech["single_set_additions"][k], "MAP") for k in ("reranker-7_Comp", "reranker-7_Coh", "reranker-7_Dom")))
    add("§5", "集合の特徴量の上乗せ（MAP）が正解式 2 本以上のどの本数でも上回った分割の数（最小）",
        lambda r: f"{min(b['wins'] for b in r.mech['lift_by_n_correct']['reranker-10S']['MAP'] if b['bin'] != '1')}/10")
    for lab, nm in (("GROUP_var", "変数の 6 特徴量"), ("GROUP_topic", "話題の 4 特徴量")):
        add("§5", f"PFI：{nm}をまとめてシャッフルしたとき低下が正の分割（Recall@20／MAP）",
            lambda r, lab=lab: f"{r.pfi['targets'][lab]['Recall@20']['n_positive']}/10／{r.pfi['targets'][lab]['MAP']['n_positive']}/10")
    for S in ("A", "B"):
        g = lambda r, S=S, sub="all", k="infer - static", m="MAP": r.greedy["compare"][S][sub][k][m]  # noqa: E731
        add("§6", f"設定 {S}：(2) − (1) の MAP、(3) − (2) の MAP で上回った分割",
            lambda r, S=S: f"{g(r, S)['wins']}/10、{g(r, S, k='train - infer')['wins']}/10")
        add("§6", f"設定 {S}：正解式 8 本以上で (3) − (1) が上回った分割（MAP／R-Precision／Recall@20）",
            lambda r, S=S: "／".join(f"{g(r, S, 'K>=8', 'train - static', m)['wins']}/10" for m in ("MAP", "R-Precision", "Recall@20")))
    add("§7", "話題の特徴量を加えたとき上回った分割（x7 の Recall@20／x3 の Recall@20／x10 の Recall@20／x10 の MAP）",
        lambda r: "／".join([wins(r.mech["topic_additions"]["x7"]), wins(r.mech["topic_additions"]["x3"]),
                            wins(r.mech["topic_additions"]["x10"]), wins(r.mech["topic_additions"]["x10"], "MAP")]))
    add("§7", "分野ラベルの合併（表記をそろえる／100 群／40 群）で x7 を加えて上回った分割（Recall@20）",
        lambda r: "／".join(wins(r.mech["domain_merge"][k]) for k in ("L1", "K100", "K40")))

    def auc_order(r):
        a = r.td["auc_within_candidates"]
        var = min(a["x2_io_jaccard"]["mean"], a["x6_specificity"]["mean"])
        top = max(a[k]["mean"] for k in ("x3_svd_sim", "x1_text_sim", "x10_dom_set", "x7_domain"))
        return "はい" if var > top else "いいえ"
    add("§7", "候補の中の AUC で、変数の特徴量（x2・x6）が話題の特徴量（x1・x3・x7・x10）より高い", auc_order)
    add("§7", "x10 の PFI：正解式の方で x10 が高いケースで MAP が下がり、低いケースで上がる",
        lambda r: "はい" if (r.pfi["cancellation"]["gDom"]["MAP"]["mean_drop_sep_pos"] > 0 > r.pfi["cancellation"]["gDom"]["MAP"]["mean_drop_sep_neg"]) else "いいえ")
    add("§8.1", "種 42：上位 20 件に入らない正解式のうち第 1 段の見逃し",
        lambda r: f"{r.co['top20']['stage1_miss']} 本（全 {r.co['top20']['missed_total']} 本）")
    add("§8.1", "種 42：救える割合の最大（潜在意味類似度・文章類似度・式の特化度・変数の一致度）",
        lambda r: f"{max(r.co['top20']['rescue_fraction'][k]['mean'] for k in ('svd_sim', 'text_sim', 'specificity', 'io_jaccard')):.2f}")
    add("§3.4・付録 A", "設定 A：hard negative が、4 つの重みのすべてで検証用の Recall@20 で無作為を上回る",
        lambda r: f"{sum(r.sel['grid']['A'][f'{w}|hard']['val']['Recall@20']['mean'] > r.sel['grid']['A'][f'{w}|random']['val']['Recall@20']['mean'] for w in WEIGHTS4)}/4")

    def wdiff(r, S):
        g = r.sel["grid"][S]
        ms = [g[f"{w}|hard"]["val"]["Recall@20"]["mean"] for w in WEIGHTS4]
        sd_ = float(np.mean([g[f"{w}|hard"]["val"]["Recall@20"]["std"] for w in WEIGHTS4]))
        return f"重みの差 {max(ms) - min(ms):.4f} {'<' if max(ms) - min(ms) < sd_ else '≥'} 分割間の標準偏差 {sd_:.4f}"
    for S in ("A", "B"):
        add("§3.4", f"設定 {S}：検証用の Recall@20（hard negative）で、重みの差（4 つの重みの最大 − 最小）が分割間の標準偏差（4 行の平均）より小さい",
            lambda r, S=S: wdiff(r, S))
    add("§3.4", "設定 A：重み (0.3, 0.7) の N1 の規則の値（分割ごと）",
        lambda r: rng([v["w30-70"]["N1"] for v in r.rule["settings"]["A"].values()]))
    add("§3.4", "不正解の選び方で hard negative が選ばれた分割（設定 A・B の 20 通り）",
        lambda r: f"{sum(v['neg'] == 'hard' for S in ('A', 'B') for v in r.sel['selected'][S].values())}/20")

    def flat500(r):
        a = r.sweep["N1"]["A"]
        return f"{a['2000']['Recall@20']['mean'] - a['500']['Recall@20']['mean']:+.4f}"
    add("付録 C", "設定 A：N1 = 2,000 と 500 の Recall@20 の差", flat500)
    return C


def summary(o: Run, n: Run):
    """報告書の本文で直す必要のある主張を、節ごとに 1〜2 文で（数値はすべて結果のファイルから計算する）."""
    L = []
    T = lambda r, S, g, m: r.main["table"][S][g][m]["mean"]  # noqa: E731
    st = lambda r, k, m: r.mech["steps"][k][m]  # noqa: E731
    sm = lambda r, k, m="Recall@20": r.mech["summary"][k][m]["mean"]  # noqa: E731
    ws = lambda c, m="Recall@20": f"{c[m]['wins']}/{c[m]['n']}"  # noqa: E731
    wn = lambda c: f"{c['wins']}/{c['n']}"  # noqa: E731（1 つの指標の比較の、上回った分割の数）
    k1, k2, k2s, k3 =("(i) baseline_same_w -> reranker-2", "(ii) reranker-2 -> reranker-7",
                       "(ii') reranker-2 -> reranker-2+Spec", "(iii) reranker-7 -> reranker-10S")
    same_b = n.main["table"]["B"] == o.main["table"]["B"]
    L.append(f"- **§4（主結果）は変わらない**：提案手法は、どちらの設定でも、どの指標でも、10 通りの分割のすべてで baseline（TF-IDF、BM25、E5）を上回る。"
             f"設定 A の MAP は、baseline {f3(T(o, 'A', 'baseline_tfidf', 'MAP'))} → {f3(T(n, 'A', 'baseline_tfidf', 'MAP'))}、"
             f"一括の並べ直し {f3(T(o, 'A', 'proposed_static', 'MAP'))} → {f3(T(n, 'A', 'proposed_static', 'MAP'))}、"
             f"逐次選択（訓練も逐次）{f3(T(o, 'A', 'proposed_train', 'MAP'))} → {f3(T(n, 'A', 'proposed_train', 'MAP'))}。"
             f"Recall@20 は baseline {f3(T(n, 'A', 'baseline_tfidf', 'Recall@20'))}、一括 {f3(T(n, 'A', 'proposed_static', 'Recall@20'))} のまま"
             f"（訓練も逐次は {f3(T(o, 'A', 'proposed_train', 'Recall@20'))} → {f3(T(n, 'A', 'proposed_train', 'Recall@20'))}）。"
             f"{'設定 B の値は前回と同じ。' if same_b else '設定 B の値が前回と違う（要確認）。'}")
    L.append(f"- **§5 (i)**：x1・x2 だけの MLP が同じ重みの baseline を上回った分割（Recall@20 と MAP）は、"
             f"{wn(st(o, k1, 'Recall@20'))} と {wn(st(o, k1, 'MAP'))} → {wn(st(n, k1, 'Recall@20'))} と {wn(st(n, k1, 'MAP'))}"
             f"（Recall@20 {f3(sm(n, 'baseline_same_w'))} → {f3(sm(n, 'reranker-2'))}、MAP {f3(sm(n, 'baseline_same_w', 'MAP'))} → "
             f"{f3(sm(n, 'reranker-2', 'MAP'))}）。「重みを学び直しても精度は上がらない」はそのまま。")
    L.append(f"- **§5 (ii)**：2 → 7 特徴量の向上は Recall@20 {sd(st(o, k2, 'Recall@20')['mean_delta'])} → {sd(st(n, k2, 'Recall@20')['mean_delta'])}、"
             f"MAP {sd(st(o, k2, 'MAP')['mean_delta'])} → {sd(st(n, k2, 'MAP')['mean_delta'])}、x6 だけでも Recall@20 "
             f"{sd(st(o, k2s, 'Recall@20')['mean_delta'])} → {sd(st(n, k2s, 'Recall@20')['mean_delta'])}（どれも 10 通りすべて）。"
             f"小さくなったのは、出発点の x1・x2 だけの MLP が上がったため（Recall@20 {f3(sm(o, 'reranker-2'))} → {f3(sm(n, 'reranker-2'))}。"
             f"旧は種 3141 の分割だけ {o.mech['summary']['reranker-2']['Recall@20']['per_seed']['3141']:.3f} と低かった。新では "
             f"{n.mech['summary']['reranker-2']['Recall@20']['per_seed']['3141']:.3f}）。「この段階の向上はほとんど式の特化度による」はそのまま。")
    L.append(f"- **§5 (iii)**：集合の特徴量の Recall@20 {sd(st(n, k3, 'Recall@20')['mean_delta'])}・MAP {sd(st(n, k3, 'MAP')['mean_delta'])} は同じだが、"
             f"Recall@20 で上回った分割が {wn(st(o, k3, 'Recall@20'))} → {wn(st(n, k3, 'Recall@20'))}"
             f"（MAP は {wn(st(n, k3, 'MAP'))}）。「どちらも 10 通りすべて」は直す必要がある。"
             f"1 個ずつ加えると、x10 の MAP が {sd(o.mech['single_set_additions']['reranker-7_Dom']['MAP']['mean_delta'])}"
             f"（{ws(o.mech['single_set_additions']['reranker-7_Dom'], 'MAP')}）→ {sd(n.mech['single_set_additions']['reranker-7_Dom']['MAP']['mean_delta'])}"
             f"（{ws(n.mech['single_set_additions']['reranker-7_Dom'], 'MAP')}）になり、「どれも 10 通りすべて」は x10 について成り立たない"
             f"（x8 {sd(n.mech['single_set_additions']['reranker-7_Comp']['MAP']['mean_delta'])}、x9 "
             f"{sd(n.mech['single_set_additions']['reranker-7_Coh']['MAP']['mean_delta'])} はどちらも 10 通りすべて）。")
    pt = lambda r, lab, m="Recall@20": r.pfi["targets"][lab][m]  # noqa: E731
    L.append(f"- **§5 の PFI は変わらない**：変数の 6 特徴量をまとめてシャッフルすると Recall@20 {f3(n.pfi['base']['Recall@20']['mean'])} → "
             f"{f3(pt(n, 'GROUP_var')['shuffled_mean'])}（旧 {f3(pt(o, 'GROUP_var')['shuffled_mean'])}）、x6 だけで {f3(pt(n, 'specificity')['shuffled_mean'])}"
             f"（旧 {f3(pt(o, 'specificity')['shuffled_mean'])}）、話題の 4 特徴量で {f3(pt(n, 'GROUP_topic')['shuffled_mean'])}"
             f"（低下 {f3(pt(n, 'GROUP_topic')['drop_mean'])}、正 {pt(n, 'GROUP_topic')['n_positive']}/10）。")
    gc = lambda r, S, sub, k, m="MAP": r.greedy["compare"][S][sub][k][m]  # noqa: E731
    L.append(f"- **§6**：設定 A の (3) − (2) の MAP は {sd(gc(n, 'A', 'all', 'train - infer')['mean_delta'])} で、上回った分割が "
             f"{gc(o, 'A', 'all', 'train - infer')['wins']}/10 → {gc(n, 'A', 'all', 'train - infer')['wins']}/10。"
             f"(2) − (1)（{sd(gc(n, 'A', 'all', 'infer - static')['mean_delta'])}、{gc(n, 'A', 'all', 'infer - static')['wins']}/10）と、"
             f"正解式 8 本以上の (3) − (1)（MAP {sd(gc(n, 'A', 'K>=8', 'train - static')['mean_delta'])}、{gc(n, 'A', 'K>=8', 'train - static')['wins']}/10）は変わらない。"
             "設定 B は同じ値。")
    ta = lambda r, k: r.mech["topic_additions"][k]  # noqa: E731
    dm = lambda r, k: r.mech["domain_merge"][k]  # noqa: E731
    L.append(f"- **§7**：話題の特徴量を加えたときの Recall@20 の差は、x7 {dw(ta(o, 'x7'))} → {dw(ta(n, 'x7'))}、x3 {dw(ta(o, 'x3'))} → {dw(ta(n, 'x3'))}、"
             f"x10 {dw(ta(o, 'x10'), fmt=sdx)} → {dw(ta(n, 'x10'), fmt=sdx)}（MAP は {dw(ta(o, 'x10'), 'MAP')} → {dw(ta(n, 'x10'), 'MAP')}）。"
             f"分野ラベルの合併は {'、'.join(sd(dm(o, k)['Recall@20']['mean_delta']) for k in ('L1', 'K100', 'K40'))} → "
             f"{'、'.join(sd(dm(n, k)['Recall@20']['mean_delta']) for k in ('L1', 'K100', 'K40'))}（上回った分割は "
             f"{rng([dm(n, k)['Recall@20']['wins'] for k in ('L1', 'K100', 'K40')])} 通り）。"
             "「加えても一貫して上がらない」はそのままだが、「x10 は MAP で 10 通りすべてで上がる」と、合併の差の値は直す必要がある。"
             "特徴量が 2〜3 個の MLP は、1 つの分割だけ Recall@20 が大きく下がることがあり（例：表記をそろえた合併の種 789 で "
             f"{n.mech['summary']['Dom@L1']['Recall@20']['per_seed']['789']:.3f}）、差の平均はその分割に左右される。")
    cc = lambda r: r.pfi["cancellation"]["gDom"]["MAP"]  # noqa: E731
    L.append(f"- **§7 の打ち消し合い**：x10 の PFI で、不正解式の方で x10 が高いケースの MAP の上昇は "
             f"{f3(-cc(o)['mean_drop_sep_neg'])}（{cc(o)['n_sep_neg']} 件）→ {f3(-cc(n)['mean_drop_sep_neg'])}（{cc(n)['n_sep_neg']} 件）で、ほぼ 0 になった。"
             f"正解式の方が高いケースの低下は {f3(cc(n)['mean_drop_sep_pos'])}（{cc(n)['n_sep_pos']:,} 件）。")
    t2o, t2n = o.co["top20"], n.co["top20"]
    L.append(f"- **§8.1**：種 42 の分割のテストは {t2o['n_cases']} → {t2n['n_cases']} 件（core_019_v1 を外した）。上位 20 件に入らない正解式は "
             f"{t2o['missed_total']} → {t2n['missed_total']} 本で、第 1 段の見逃しは {t2n['stage1_miss']} 本のまま。全部並べたケースと漏れのあるケースは "
             f"{t2o['solved']}／{t2o['unsolved']} → {t2n['solved']}／{t2n['unsolved']} 件。")
    L.append(f"- **§3.4**：不正解の選び方は、20 通りの分割のうち "
             f"{sum(v['neg'] == 'hard' for S in ('A', 'B') for v in o.sel['selected'][S].values())} 通り → "
             f"{sum(v['neg'] == 'hard' for S in ('A', 'B') for v in n.sel['selected'][S].values())} 通りで hard negative"
             "（旧は設定 A の種 7777 だけ無作為）。重みの差が分割のばらつきより小さいことは変わらない。")
    swo, swn = o.sweep["N1"]["A"], n.sweep["N1"]["A"]
    L.append(f"- **付録 C**：設定 A の Recall@20 は N1 = 500 で {mean(swo['500']['Recall@20'])} → {mean(swn['500']['Recall@20'])}、"
             f"2,000 で {mean(swn['2000']['Recall@20'])}。1 回の訓練と評価の時間は、N1 = 50 で {swo['50']['time_total_sec']['mean']:.0f} → "
             f"{swn['50']['time_total_sec']['mean']:.0f} 秒、2,000 で {swo['2000']['time_total_sec']['mean']:.0f} → {swn['2000']['time_total_sec']['mean']:.0f} 秒"
             f"（「約 3 倍」は約 {swn['2000']['time_total_sec']['mean'] / swn['50']['time_total_sec']['mean']:.1f} 倍になる）。")
    L.append("- **§3.2・§9**：入出力を選び直した 111 件の説明文が入出力変数と合っていない、という記述と限界は当てはまらなくなる（説明文を書き直した）。"
             "単一出典の 92 件のうち 1 件が 2 冊の教科書の式から成る、という記述も、その 1 件（core_019_v1）を外したので要らなくなる。")
    return L


def build(o: Run, n: Run, report_text: str, rev: str):
    D = Doc(o, n, report_text)
    dc = json.load(open(K21 / "data_counts.json", encoding="utf-8"))
    ctx = dc["context_changes_vs_db21007"]
    D.L.append("# 8/31 報告（db21007）の数値：旧値 → 新値（2026-10-21、出典横断の 113 件の説明文を書き直し、core_019_v1 を外した）\n")
    D.p(f"報告書 `{REPORT}`（{rev}）の数値を、節・表ごとに新しい値と対にした。報告書の本文は変えていない。"
        "旧値は `experiments/kato1020/`、新値は `experiments/kato1021/`（同じファイル名）から、同じ関数で計算した。"
        "旧値として載せた数値が報告書の本文に現れることは、スクリプト（`kato1021_mapping.py`）で照合した（照合できなかった数値は末尾）。")

    # ---- 条件 ----
    D.h("何を変えたか")
    D.p(f"1. 出典横断で入出力変数を選び直したケース {ctx['by_variant'].get('multisource_random_io', 0)} 件と、元のケース "
        f"{ctx['by_variant'].get('multisource_original', 0)} 件（{'、'.join(ctx['case_ids_not_random_io'])}）の説明文（context）を、"
        "入出力変数に合うように LLM（Claude Opus 4.8）で書き直した（本人の決定、2026-10-21。commit e995ca0、記録は "
        "`experiments/reselected_io_contexts.json`）。"
        f"報告書（{ctx['rev']}）の時点の `training_cases.json` との違いは、{ctx['n_changed']} 件の "
        f"{'・'.join(ctx['fields'])} だけである（入出力変数・正解集合・式は同じ。外したケースの中に {ctx['n_changed_excluded']} 件、"
        f"合成のケースに {ctx['n_changed_setting_B']} 件）。")
    newly = dc.get("newly_excluded_in_kato1020_runs", {})
    D.p(f"2. 外すケースは {n.ex['n_excluded']} 件（`experiments/kato1020/excluded_cases_v3.json`、commit b7f7cd2）：前回の "
        f"{o.ex['n_excluded']} 件（未知の変数が決まらない）に、"
        + "、".join(newly) + "（単一出典のうち、正解集合が 2 冊の教科書（Seborg ほか、Marlin）の式から成る唯一のケース。"
        "同じ作り方のケースがほかになく、1 件だけでは学習にも評価にも意味がないので外す。本人の決定、2026-10-21）を加えた。"
        f"設定 A は {o.ex['setting_A_after']:,} 件 → {n.ex['setting_A_after']:,} 件。"
        "途中で止めた 2 つの実行（111 件を外す v2、91 件の除外での説明文の書き直し）の出力は、`experiments/kato1021_v2_中止/` と "
        "`experiments/kato1021_v1除外_中止/` に移し、commit していない。")
    D.p("3. 説明文は、第 1 段の文章類似度と、第 2 段の話題の特徴量（x1・x3・x7）に入るので、設定 A の実験をすべてやり直した。"
        "手順と定数の決め方は c7c4feb＋ed41231 と同じ：baseline の重み（TF-IDF・BM25・E5 の各 12 通り）と不正解の選び方を、分割ごとに"
        "検証用のケースで選び、提案手法の第 1 段の重みは (0.3, 0.7) に固定した。")
    sp = dc.get("splits_match_kato1020_runs", {})
    where = "".join(f"{cid} は、c7c4feb の分割のうち種 {'・'.join(s for s, v in d.items() if v == 'test')} でテスト、"
                    f"種 {'・'.join(s for s, v in d.items() if v == 'train')} で訓練に入っていた"
                    + (f"（検証は種 {'・'.join(s for s, v in d.items() if v == 'val')}）" if any(v == 'val' for v in d.values()) else "")
                    + "。" for cid, d in newly.items())
    D.p("4. 分割は、c7c4feb の実行に記録された検証・テストのケース ID から、新しく外したケースを除くと、"
        f"{'設定 A・B とも 10 通りすべてで一致した（層化で選ばれる分割は変わらなかった）' if sp and all(sp.values()) else f'一致しない（{sp}）'}。"
        f"{where}（`kato1021_counts.py`）")
    if n.rule and o.rule:
        # 比べるのは N1 と、その元になる「訓練と検証のケースで全正解式が入る最小の N1」の最大値（ケース数の欄は除く）
        same_rule = all(n.rule["settings"][S][s][w][k] == o.rule["settings"][S][s][w][k]
                        for S in ("A", "B") for s in o.rule["settings"][S] for w in o.rule["weights"]
                        for k in ("N1", "n1_trainval_max"))
        r30 = sorted({v["w30-70"]["N1"] for v in n.rule["settings"]["A"].values()})
        D.p(f"5. N1 の規則（訓練と検証のケースで全正解式が入る値を千の位に切り上げ）で、重み (0.3, 0.7) の N1 は、"
            f"{'全分割で 2,000' if r30 == [2000] else rng(r30)} になった"
            f"（{'4 つの重み、設定 A・B、10 通りの分割のすべてで前回と同じ値' if same_rule else '前回と違う値がある'}）。")
    D.p("6. 設定 B（合成 1,000 件）は説明文が変わっていないので、前回の実行結果（`experiments/kato1020/raw/`）をそのまま使った。"
        "古典的情報検索手法の設定 B は計算し直して、前回と完全に一致した。")
    D.p("7. 有料の LLM は呼んでいない（E5 の文の埋め込みは、書き直した説明文について手元で計算し直した）。")
    D.p("8. 旧値は報告書の db21007 の版の数値である。その後の版（f1076cd）では、表 2 の単一出典の行（91 件、1.33 (4)、1.00）、"
        "要件 3 が自明に満たされるケース（1,724 件）、設定 A の件数（1,731 件）と、「手法の定数」を「ハイパーパラメータ」とよぶ書き方が"
        "すでに直されている（§3.4 以降の数値は db21007 のまま）。")

    # ---- 結論の比較 ----
    C = claims(o, n)
    D.h("結論が変わった点（勝ち数と主張）")
    ch = [c for c in C if c[2] != c[3]]
    same = [c for c in C if c[2] == c[3]]
    D.p("報告書の本文で直す必要のある主張を、節ごとにまとめる（数値は旧 → 新。詳しい値は下の節ごとの表）。\n")
    D.p("\n".join(summary(o, n)))
    D.p("\n10 通りの分割で上回った数と、報告書の主張の根拠になっている大小関係を、旧と新で比べた。変わったものを先に並べる"
        "（数値が少し動いただけで主張は同じものも、文字列が違えば「変わった」に入る）。")
    D.head(("節", "内容", "旧", "新"))
    for sec, lab, a, b in ch:
        D.L.append(row(sec, f"**{lab}**", a, f"**{b}**"))
    for sec, lab, a, b in same:
        D.L.append(row(sec, lab, a, b))
    D.L.append("")
    D.p(f"変わった項目：{len(ch)} 件、変わらなかった項目：{len(same)} 件（上の表で太字が変わった項目）。")

    # ---- §1・§2.1 ----
    D.h("§1 本報告の位置づけ・§2.1 問題設定")
    D.head()
    D.tr("(3) 評価と訓練から除いたケース", lambda r: f"{r.ex['n_excluded']} 件", chk=["91"],
         note="新は、未知の変数が決まらない 91 件に core_019_v1 を加えた 92 件（報告書の (3) には core_019_v1 の除外を書き足す必要がある）")
    D.tr("評価に使うケース（設定 A）", lambda r: f"{r.cnt['n_cases']:,} ケース")
    D.tr("要件 1〜3 を満たすケース", lambda r: f"{min(r.cnt['requirements_1_3'].values()):,} 件（全件）", chk=[])
    D.tr("割り当てのできない（未知の変数が決まらない）ケース", lambda r: f"{len(r.ex.get('v1_ids', r.ex['excluded_case_ids']))} 件")
    D.tr("入出力変数以外の変数がない（要件 3 が自明）ケース", lambda r: f"{r.cnt['req3_trivial_no_intermediate']:,} 件")

    # ---- §3.1 ----
    D.h("§3.1 数式データベース")
    D.p("変わらない（データベースの数え方はケースによらない：11,146 式、361 出典、記号 13,220 種、分野ラベル 1,998 種、"
        "2 本以上の式に現れる記号 6,374 種のうち 4,208 種で定義が違う）。")

    # ---- §3.2 ----
    D.h("§3.2 評価ケース（表 2 を含む）")
    D.head()
    D.tr("評価に使うケース（除いた数）", lambda r: f"{r.cnt['n_cases']:,} 件（{r.ex['n_excluded']} 件を除いた）")
    D.tr("単一出典（表 2 の行：件数、式数の平均（最大）、出典数の平均）",
         lambda r: (lambda t: f"{t['n']} 件、{t['mean_K']:.2f} ({t['max_K']})、{t['mean_sources']:.2f}")(r.cnt["table2"]["単一出典"]))
    sb = lambda r: r.cnt["single_source_breakdown"]  # noqa: E731
    D.tr("単一出典のうち、正解集合が 1 つの出典の式だけから成るケース", lambda r: f"{sb(r)['n']} 件のうち {sb(r)['one_source']} 件",
         note="新では、2 冊の教科書の式から成る 1 件（core_019_v1）を外したので、全件が 1 つの出典")
    D.tr("単一出典のうち、正解式が 1 本のケース", lambda r: f"{sb(r)['n']} 件中 {sb(r)['K1']} 件")
    D.tr("単一出典のうち、基礎式集の式だけから成るケース", lambda r: f"{sb(r)['n']} 件中 {sb(r)['handbook_only']} 件")
    D.tr("出典横断（表 2 の行：件数、式数の平均（最大）、出典数の平均）",
         lambda r: (lambda t: f"{t['n']} 件、{t['mean_K']:.2f} ({t['max_K']})、{t['mean_sources']:.2f}")(r.cnt["table2"]["出典横断"]))
    D.tr("出典横断のうちプログラムで作ったもの", lambda r: f"{r.cnt['by_variant']['multisource_v3']} 件（作った 500 件から 4 件を除いた）",
         chk=["496"])
    D.tr("同上のうち、共有する記号の定義の文章が式によって違うケース", lambda r: "496 件中 471 件", note="同じ（496 件は変わらない）", chk=["471"])
    D.tr("出典横断の残り（LLM が書いた組と、その入出力の選び直し）",
         lambda r: f"{r.cnt['by_variant'].get('multisource_original', 0) + r.cnt['by_variant'].get('multisource_random_io', 0)} 件"
                   f"（{r.cnt['by_variant'].get('multisource_original', 0)} 件＋{r.cnt['by_variant'].get('multisource_random_io', 0)} 件）",
         chk=["144", "33", "111"])
    D.tr("同上のうち、どの 2 本の正解式も変数記号を共有しないケース",
         lambda r: (lambda v: f"{sum(v.values())} 件（33 件のうち {v.get('multisource_original', 0)} 件"
                              + (f"、111 件のうち {v['multisource_random_io']} 件" if v.get("multisource_random_io") else "") + "）")(
             r.cnt["cross_source_no_shared_symbol_by_variant"]), chk=["105", "24", "81"])
    D.tr("合成（表 2 の行）", lambda r: (lambda t: f"{t['n']:,} 件、{t['mean_K']:.2f} ({t['max_K']})、{t['mean_sources']:.2f}")(r.cnt["table2"]["合成"]),
         chk=["1,000", "5.50", "1.10"])
    D.tr("設定 A の件数", lambda r: f"{r.cnt['n_cases']:,} 件")
    D.tr("正解式の数の平均（設定 A）", lambda r: f"{r.cnt['setting_A_mean_K']:.2f}")
    D.tr("正解式が 2 本以上のケースの割合（設定 A）", lambda r: f"{r.cnt['setting_A_share_K_ge_2'] * 100:.1f}%")
    D.tr("正解式の数の平均（設定 B）", lambda r: "5.50", chk=["5.50"])
    D.p("\n§3.2 の「説明文は元のケースの文をそのまま使ったので、111 件の説明文は入出力変数と合っていない」と、§9 の限界の同じ趣旨の項目は、"
        f"当てはまらなくなる（入出力を選び直した {ctx['by_variant'].get('multisource_random_io', 0)} 件と元のケース "
        f"{ctx['by_variant'].get('multisource_original', 0)} 件の説明文を、入出力変数に合うように書き直した）。"
        "§3.2 の単一出典の段落の「92 件のうち 91 件の正解集合は 1 つの出典の式だけから成る…残る 1 件の正解集合は、2 冊の教科書…の式から成る」は、"
        "core_019_v1 を外したので、評価に使う 91 件はすべて 1 つの出典の式だけから成る、になる（LLM が作った 123 件のうち条件を満たした 92 件から、"
        "core_019_v1 を外した 91 件を使う）。表 2 の単一出典の行と、設定 A の件数も変わる。")

    # ---- §3.3 ----
    D.h("§3.3 訓練、検証、テストの分割")
    D.head()
    D.tr("設定 A のテストのケース（分割ごとの範囲）", lambda r: rng([v["n_test"] for v in r.sel["selected"]["A"].values()]) + " 件")
    D.tr("設定 A の検証のケース（分割ごとの範囲）", lambda r: rng([v["n_val"] for v in r.sel["selected"]["A"].values()]) + " 件")
    D.tr("設定 B のテストのケース（参考）", lambda r: rng([v["n_test"] for v in r.sel["selected"]["B"].values()]) + " 件", chk=[])
    D.tr("設定 B の検証のケース（参考）", lambda r: rng([v["n_val"] for v in r.sel["selected"]["B"].values()]) + " 件", chk=[])

    # ---- §3.4 ----
    D.h("§3.4 比較手法と手法の定数")
    D.head()
    for S in ("A", "B"):
        D.tr(f"baseline（TF-IDF）の選ばれた重み（設定 {S}、分割の数）",
             lambda r, S=S: wcount(v["w"] for v in r.bl["settings"][S]["scorers"]["tfidf"]["selected"].values()), chk=[])
    for S in ("A", "B"):
        for sc, nm in (("bm25", "BM25"), ("e5", "E5")):
            D.tr(f"{nm} の選ばれた重み（設定 {S}）",
                 lambda r, S=S, sc=sc: wcount(v["w"] for v in r.bl["settings"][S]["scorers"][sc]["selected"].values()), chk=[])
    for S in ("A", "B"):
        D.tr(f"提案手法：重みによる検証用の Recall@20 の差（hard negative、設定 {S}）",
             lambda r, S=S: rng([r.sel["grid"][S][f"{w}|hard"]["val"]["Recall@20"]["mean"] for w in WEIGHTS4], f3))
    D.tr("同上：分割の違いによるばらつき（検証用の Recall@20 の分割間の標準偏差。設定 A・B の hard negative の 8 行の平均）",
         lambda r: "約 " + f"{np.mean([r.sel['grid'][S][f'{w}|hard']['val']['Recall@20']['std'] for S in ('A', 'B') for w in WEIGHTS4]):.3f}")
    D.tr("N1：重み (0.3, 0.7) の規則の値（設定 A）", lambda r: rng([v["w30-70"]["N1"] for v in r.rule["settings"]["A"].values()]),
         chk=["2,000"])
    D.tr("N1：ほかの重みの規則の値（設定 A）",
         lambda r: rng([v[w]["N1"] for v in r.rule["settings"]["A"].values() for w in ("w70-30", "w50-50", "w00-100")]))
    D.tr("不正解の選び方で hard negative が選ばれた分割（設定 A・B の 20 通り）",
         lambda r: f"20 通り中 {sum(v['neg'] == 'hard' for S in ('A', 'B') for v in r.sel['selected'][S].values())} 通り"
                   + (lambda x: f"（無作為：{x}）" if x else "")("、".join(f"設定 {S} の種 {s}" for S in ("A", "B")
                                                                for s, v in r.sel["selected"][S].items() if v["neg"] != "hard")),
         chk=["19"])

    # ---- §3.5 ----
    D.h("§3.5 評価方法")
    D.head()
    D.tr("正解式のうち、同じ文字列の式が別の出典にもあるもの",
         lambda r: f"{r.cnt['gt_distinct_equations']:,} 本のうち {r.cnt['gt_identical_latex_in_other_source']} 本"
                   f"（{r.cnt['gt_identical_latex_in_other_source'] / r.cnt['gt_distinct_equations'] * 100:.1f}%）")

    # ---- §4 ----
    D.h("§4 結果 1：全体比較（表 3）")
    D.head()
    D.tr("テストの被覆率（N1 = 2,000、設定 A）", lambda r: f"{r.sweep['N1']['A']['2000']['coverage']['mean']:.3f}")
    for S in ("A", "B"):
        T = lambda r, S=S: r.main["table"][S]  # noqa: E731
        Cm = lambda r, S=S: r.main["compare"][S]  # noqa: E731
        D.tr(f"設定 {S}：Recall@20 baseline → 一括（差）",
             lambda r, S=S: f"{mean(T(r)['baseline_tfidf']['Recall@20'])} → {mean(T(r)['proposed_static']['Recall@20'])}"
                            f"（{sd(Cm(r)['proposed_static vs baseline_tfidf']['Recall@20']['mean_delta'])}）")
        D.tr(f"設定 {S}：MAP baseline → 一括（差）",
             lambda r, S=S: f"{mean(T(r)['baseline_tfidf']['MAP'])} → {mean(T(r)['proposed_static']['MAP'])}"
                            f"（{sd(Cm(r)['proposed_static vs baseline_tfidf']['MAP']['mean_delta'])}）")
        for a, an in (("proposed_static", "一括"), ("proposed_train", "訓練も逐次")):
            D.tr(f"設定 {S}：{an} − baseline の上回った分割（Recall@20／MAP／R-Precision／MAP@20）",
                 lambda r, S=S, a=a: "／".join(wins(Cm(r)[f"{a} vs baseline_tfidf"], m) for m in ("Recall@20", "MAP", "R-Precision", "MAP@20")),
                 chk=[])
        for b, bn in (("baseline_bm25", "BM25"), ("baseline_e5", "E5")):
            D.tr(f"設定 {S}：一括 − {bn} の上回った分割（Recall@20／MAP／R-Precision／MAP@20）",
                 lambda r, S=S, b=b: "／".join(wins(Cm(r)[f"proposed_static vs {b}"], m) for m in ("Recall@20", "MAP", "R-Precision", "MAP@20")),
                 chk=[])

    D.h3("表 3（10 通りの分割の平均 ± 標準偏差。括弧内は別の出典にある同じ式も正解とみなした値）")
    D.head()
    names = {"baseline_tfidf": "baseline", "proposed_static": "一括の並べ直し", "proposed_train": "逐次選択（訓練も逐次）"}
    for S in ("A", "B"):
        for g, gn in names.items():
            for m in ("Recall@20", "MAP", "R-Precision"):
                D.tr(f"設定 {S}・{gn}・{m}",
                     lambda r, S=S, g=g, m=m: f"{pm(r.main['table'][S][g][m])} ({mean(r.main['table_equiv'][S][g][m])})")
    D.tr("表 3 の設定 A の件数", lambda r: f"{r.cnt['n_cases']:,} 件")

    D.h3("§4 の段落：同じ式を正解とみなした値（設定 A の MAP）")
    D.head()
    for g, gn in names.items():
        D.tr(f"{gn}：ID による判定 → 同じ式も正解", lambda r, g=g: f"{mean(r.main['table']['A'][g]['MAP'])} → {mean(r.main['table_equiv']['A'][g]['MAP'])}")
    D.tr("一括 − baseline の MAP の差（ID／同じ式）",
         lambda r: f"{sd(r.main['compare']['A']['proposed_static vs baseline_tfidf']['MAP']['mean_delta'])}／"
                   f"{sd(r.main['compare']['A']['proposed_static vs baseline_tfidf (equiv)']['MAP']['mean_delta'])}")

    D.h3("§4 の段落：正解式の数と入力変数の数で分けた差（設定 A、付録 B）")
    D.head()
    bk = lambda r, key, b: r.main["buckets"]["A"][key][b]  # noqa: E731
    D.tr("延べ件数／異なるケース", lambda r: f"{sum(v['n_records'] for v in r.main['buckets']['A']['n_correct'].values()):,} 件／"
                                       f"{r.pfi['cancellation']['gDom']['MAP']['n_cases']:,} 件")
    D.tr("正解式 1 本：Recall@20（一括／baseline）、MAP（一括／baseline）",
         lambda r: f"{f3(bk(r, 'n_correct', '1')['proposed_static|Recall@20'])}／{f3(bk(r, 'n_correct', '1')['baseline_tfidf|Recall@20'])}、"
                   f"{f3(bk(r, 'n_correct', '1')['proposed_static|MAP'])}／{f3(bk(r, 'n_correct', '1')['baseline_tfidf|MAP'])}")
    for b in ("2", "5", "10"):
        D.tr(f"Recall@20 の差（一括 − baseline）：正解式 {b} 本",
             lambda r, b=b: sd(bk(r, "n_correct", b)["proposed_static|Recall@20"] - bk(r, "n_correct", b)["baseline_tfidf|Recall@20"]))
    for b, bn in (("<=5", "5 個以下"), (">=21", "21 個以上")):
        D.tr(f"Recall@20 の差（一括 − baseline）：入力変数 {bn}",
             lambda r, b=b: sd(bk(r, "n_input", b)["proposed_static|Recall@20"] - bk(r, "n_input", b)["baseline_tfidf|Recall@20"]))

    # ---- §5 ----
    D.h("§5 結果 2：向上の内訳と PFI（設定 A）")
    D.head()
    st = lambda r, k: r.mech["steps"][k]  # noqa: E731
    sm = lambda r, k, m="Recall@20": f3(r.mech["summary"][k][m]["mean"])  # noqa: E731
    D.tr("同じ重み (0.3, 0.7) の baseline → 10 特徴量（Recall@20／MAP の差）",
         lambda r: f"{sd(st(r, 'total baseline_same_w -> reranker-10S')['Recall@20']['mean_delta'])}／"
                   f"{sd(st(r, 'total baseline_same_w -> reranker-10S')['MAP']['mean_delta'])}")
    k1 = "(i) baseline_same_w -> reranker-2"
    D.tr("(i) Recall@20：baseline → x1・x2 だけの MLP（上回った分割）",
         lambda r: f"{sm(r, 'baseline_same_w')} → {sm(r, 'reranker-2')}（{wins(st(r, k1))}）")
    D.tr("(i) MAP（上回った分割）", lambda r: f"{sm(r, 'baseline_same_w', 'MAP')} → {sm(r, 'reranker-2', 'MAP')}（{wins(st(r, k1), 'MAP')}）")
    k2 = "(ii) reranker-2 -> reranker-7"
    D.tr("(ii) Recall@20：2 → 7 特徴量（差、上回った分割）",
         lambda r: f"{sm(r, 'reranker-2')} → {sm(r, 'reranker-7')}（{dw(st(r, k2))}）")
    D.tr("(ii) MAP", lambda r: f"{sm(r, 'reranker-2', 'MAP')} → {sm(r, 'reranker-7', 'MAP')}（{dw(st(r, k2), 'MAP')}）")
    k2s = "(ii') reranker-2 -> reranker-2+Spec"
    D.tr("(ii) x6 だけを加える：Recall@20（差）", lambda r: f"{sm(r, 'reranker-2_Spec')}（{dw(st(r, k2s))}）")
    D.tr("(ii) x6 だけを加える：MAP（差）", lambda r: f"{sm(r, 'reranker-2_Spec', 'MAP')}（{dw(st(r, k2s), 'MAP')}）")
    k3 = "(iii) reranker-7 -> reranker-10S"
    D.tr("(iii) Recall@20：7 → 10 特徴量", lambda r: f"{sm(r, 'reranker-7')} → {sm(r, 'reranker-10S')}（{dw(st(r, k3))}）")
    D.tr("(iii) MAP", lambda r: f"{sm(r, 'reranker-7', 'MAP')} → {sm(r, 'reranker-10S', 'MAP')}（{dw(st(r, k3), 'MAP')}）")
    for k, nm in (("reranker-7_Comp", "x8（補完性）"), ("reranker-7_Coh", "x9（一貫性）"), ("reranker-7_Dom", "x10（分野ラベルの一致度（集合））")):
        D.tr(f"7 特徴量 ＋ {nm} の MAP の差", lambda r, k=k: dw(r.mech["single_set_additions"][k], "MAP"))
    lc = lambda r: r.mech["lift_by_n_correct"]["reranker-10S"]["MAP"]  # noqa: E731
    li = lambda r: r.mech["lift_by_n_input"]["reranker-10S"]["MAP"]  # noqa: E731
    D.tr("集合の特徴量の上乗せ（MAP）：正解式 1 本", lambda r: sd(next(b for b in lc(r) if b["bin"] == "1")["mean_delta"]))
    D.tr("同：正解式 2 本以上（本数ごとの範囲、上回った分割の最小）",
         lambda r: rng([b["mean_delta"] for b in lc(r) if b["bin"] != "1"], sd) + f"（{min(b['wins'] for b in lc(r) if b['bin'] != '1')}/10）")
    D.tr("同：入力変数 3 個以下／4〜6 個", lambda r: "／".join(sd(next(b for b in li(r) if b["bin"] == x)["mean_delta"]) for x in ("1-3", "4-6")))
    D.tr("同：入力変数 7 個以上（範囲）", lambda r: rng([b["mean_delta"] for b in li(r) if b["bin"] in ("7-10", "11-15", "16+")], sd))

    D.h3("§5 の PFI（図 2）")
    D.head()
    pb = lambda r, m="Recall@20": f3(r.pfi["base"][m]["mean"])  # noqa: E731
    pt = lambda r, lab, m="Recall@20": r.pfi["targets"][lab][m]  # noqa: E731
    D.tr("シャッフル前（Recall@20／MAP）", lambda r: f"{pb(r)}／{pb(r, 'MAP')}")
    D.tr("変数の 6 特徴量をまとめて：Recall@20（低下が正の分割）",
         lambda r: f"{pb(r)} → {f3(pt(r, 'GROUP_var')['shuffled_mean'])}（{pt(r, 'GROUP_var')['n_positive']}/10）")
    D.tr("同：MAP", lambda r: f"{pb(r, 'MAP')} → {f3(pt(r, 'GROUP_var', 'MAP')['shuffled_mean'])}（{pt(r, 'GROUP_var', 'MAP')['n_positive']}/10）")
    D.tr("式の特化度 x6 だけ：Recall@20", lambda r: f"{pb(r)} → {f3(pt(r, 'specificity')['shuffled_mean'])}")
    D.tr("話題の 4 特徴量をまとめて：Recall@20／MAP", lambda r: f"{f3(pt(r, 'GROUP_topic')['shuffled_mean'])}／{f3(pt(r, 'GROUP_topic', 'MAP')['shuffled_mean'])}")
    D.tr("同：低下（Recall@20／MAP、低下が正の分割）",
         lambda r: f"{f3(pt(r, 'GROUP_topic')['drop_mean'])}／{f3(pt(r, 'GROUP_topic', 'MAP')['drop_mean'])}"
                   f"（{pt(r, 'GROUP_topic')['n_positive']}/10／{pt(r, 'GROUP_topic', 'MAP')['n_positive']}/10）")
    D.tr("図 2 のファイル", lambda r: "figures/fig_pfi_recall20_ja_final.png" if r is o else "figures/fig_pfi_recall20_ja_final_1021.png",
         chk=[], note="旧図は上書きしていない。キャプションの「シャッフル前の Recall@20（0.981）」は上の値")

    # ---- §6 ----
    D.h("§6 結果 3：逐次選択（表 4）")
    D.head()
    gs = lambda r, S, sub, k, m: r.greedy["summary"][S][sub][k][m]["mean"]  # noqa: E731
    for S in ("A", "B"):
        for m in ("Recall@20", "MAP", "R-Precision"):
            D.tr(f"設定 {S}・{m}：(1)／(2)／(3)", lambda r, S=S, m=m: "／".join(f3(gs(r, S, "all", k, m)) for k in ("static", "infer", "train")))
        D.tr(f"設定 {S}・8 本以上・MAP：(1)／(2)／(3)", lambda r, S=S: "／".join(f3(gs(r, S, "K>=8", k, "MAP")) for k in ("static", "infer", "train")))
    gc = lambda r, S, sub, k: r.greedy["compare"][S][sub][k]  # noqa: E731
    for S in ("A", "B"):
        D.tr(f"設定 {S}：(2) − (1) の MAP（上回った分割）", lambda r, S=S: dw(gc(r, S, "all", "infer - static"), "MAP"))
        D.tr(f"設定 {S}：(3) − (2) の MAP", lambda r, S=S: dw(gc(r, S, "all", "train - infer"), "MAP"))
    for S in ("A", "B"):
        D.tr(f"設定 {S}：正解式 8 本以上の延べ件数", lambda r, S=S: f"{sum(r.main['buckets'][S]['n_correct'][k]['n_records'] for k in ('8', '9', '10')):,} 件")
        for m in ("MAP", "R-Precision", "Recall@20"):
            D.tr(f"設定 {S}：8 本以上で (3) − (1) の {m}", lambda r, S=S, m=m: dw(gc(r, S, "K>=8", "train - static"), m))

    # ---- §7 ----
    D.h("§7 結果 4：話題の近さを測る特徴量（設定 A）")
    D.head()
    ta = lambda r, k: r.mech["topic_additions"][k]  # noqa: E731
    D.tr("x1・x2 ＋ x7 − x1・x2（Recall@20）", lambda r: dw(ta(r, "x7")))
    D.tr("x1・x2 ＋ x3 − x1・x2（Recall@20）", lambda r: dw(ta(r, "x3")))
    D.tr("7 特徴量 ＋ x10 − 7 特徴量（Recall@20）", lambda r: dw(ta(r, "x10"), fmt=sdx))
    D.tr("同（MAP）", lambda r: dw(ta(r, "x10"), "MAP"))
    for k, nm in (("L1", "表記をそろえた 1,958 種"), ("K100", "k-means で 100 群"), ("K40", "k-means で 40 群")):
        D.tr(f"分野ラベルの合併（{nm}）：x1・x2 ＋ x7 − x1・x2（Recall@20）", lambda r, k=k: dw(r.mech["domain_merge"][k]))
    D.tr("AUC の平均をとったケースの数", lambda r: f"{r.td['config']['n_cases']:,} ケース")
    for f, nm in (("x2_io_jaccard", "変数の一致度 x2"), ("x6_specificity", "式の特化度 x6"), ("x3_svd_sim", "潜在意味類似度 x3"),
                  ("x1_text_sim", "文章類似度 x1"), ("x10_dom_set", "分野ラベルの一致度（集合）x10"), ("x7_domain", "分野ラベルの一致 x7")):
        D.tr(f"AUC（表 5）：{nm}（全 DB／候補 2,000 件の中）",
             lambda r, f=f: f"{r.td['auc_full_db'][f]['mean']:.3f} ± {r.td['auc_full_db'][f]['sem']:.3f}／"
                            f"{r.td['auc_within_candidates'][f]['mean']:.3f} ± {r.td['auc_within_candidates'][f]['sem']:.3f}")
    D.tr("候補の中の相関 corr(x1, x2)（種 42 のテストのケース数）",
         lambda r: f"{sd(r.td['corr_text_jaccard_within_candidates_seed42_test']['mean'])}（{r.td['n_test_seed42']} ケース）")
    cc = lambda r: r.pfi["cancellation"]["gDom"]["MAP"]  # noqa: E731
    D.tr("x10 の PFI：正解式の方で x10 が高いケース（MAP の低下）", lambda r: f"{cc(r)['n_sep_pos']:,} 件（{f3(cc(r)['mean_drop_sep_pos'])}）")
    D.tr("同：不正解式の方が高いケース（MAP の上昇）", lambda r: f"{cc(r)['n_sep_neg']:,} 件（{f3(-cc(r)['mean_drop_sep_neg'])}）")

    # ---- §8 ----
    D.h("§8 結果 5：誤りの分析")
    D.head()
    t2 = lambda r: r.co["top20"]  # noqa: E731
    D.tr("§8.1 種 42 のテストのケース／正解式（延べ）", lambda r: f"{t2(r)['n_cases']} 件／{t2(r)['n_gt']:,} 本")
    D.tr("§8.1 種 42 の Recall@20（10 通りの平均）", lambda r: f"{f3(t2(r)['mean_recall'])}（{mean(r.main['table']['A']['proposed_static']['Recall@20'])}）")
    D.tr("§8.1 上位 20 件に入らなかった正解式（うち第 1 段の見逃し）", lambda r: f"{t2(r)['missed_total']} 本（{t2(r)['stage1_miss']} 本）")
    D.tr("§8.1 全部並べたケース／そうでないケース", lambda r: f"{t2(r)['solved']} 件／{t2(r)['unsolved']} 件")
    D.tr("§8.1 正解式の数の平均（全部／漏れあり）",
         lambda r: f"{t2(r)['solved_vs_unsolved']['n_correct']['mean_solved']:.2f}／{t2(r)['solved_vs_unsolved']['n_correct']['mean_unsolved']:.2f}")
    D.tr("§8.1 入力変数の数の平均（全部／漏れあり）",
         lambda r: f"{t2(r)['solved_vs_unsolved']['n_input']['mean_solved']:.1f}／{t2(r)['solved_vs_unsolved']['n_input']['mean_unsolved']:.1f}")
    D.tr("§8.1 救える割合（潜在意味類似度／文章類似度／式の特化度／変数の一致度）",
         lambda r: "／".join(f"{t2(r)['rescue_fraction'][k]['mean']:.2f}" for k in ("svd_sim", "text_sim", "specificity", "io_jaccard")))
    D.tr("§8.2 同じ文字列の式が別の出典にもある正解式",
         lambda r: f"{r.cnt['gt_distinct_equations']:,} 本のうち {r.cnt['gt_identical_latex_in_other_source']} 本", note="同じ")
    D.tr("§8.2 設定 A の MAP：一括（ID → 同じ式）", lambda r: f"{mean(r.main['table']['A']['proposed_static']['MAP'])} → "
                                                         f"{mean(r.main['table_equiv']['A']['proposed_static']['MAP'])}")
    D.tr("§8.2 設定 A の MAP：逐次（訓練も逐次）", lambda r: f"{mean(r.main['table']['A']['proposed_train']['MAP'])} → "
                                                     f"{mean(r.main['table_equiv']['A']['proposed_train']['MAP'])}")

    # ---- §9・§10 ----
    D.h("§9 現時点の限界・§10 まとめ")
    D.head()
    D.tr("出典横断のうち説明文と入出力変数が合っていないケース",
         lambda r: f"{r.cnt['by_variant'].get('multisource_random_io', 0) - (ctx['by_variant'].get('multisource_random_io', 0) if r is n else 0)} 件",
         chk=["111"], note="新では、入出力を選び直したケースの説明文をすべて書き直したので、この項目は当てはまらない")
    D.tr("2 本以上の正解式に現れる記号の定義の文章が式によって違うケース", lambda r: f"{r.cnt['same_symbol_different_definition_cases']:,} 件")
    D.tr("LLM（Claude Opus 5.5）が別の量を指すと判定したケース", lambda r: f"{r.cnt['llm_meaning_conflict_cases']} 件",
         note="既存の判定（experiments/var_descriptions.json）を数え直した。LLM は呼んでいない")
    D.tr("変数の 6 特徴量をまとめてシャッフルしたときの Recall@20 の低下", lambda r: f3(r.pfi["targets"]["GROUP_var"]["Recall@20"]["drop_mean"]))
    D.tr("§10：Recall@20（設定 A：baseline → 一括、設定 B）",
         lambda r: f"{mean(r.main['table']['A']['baseline_tfidf']['Recall@20'])} → {mean(r.main['table']['A']['proposed_static']['Recall@20'])}、"
                   f"{mean(r.main['table']['B']['baseline_tfidf']['Recall@20'])} → {mean(r.main['table']['B']['proposed_static']['Recall@20'])}")
    D.tr("§10：MAP（設定 A、設定 B）",
         lambda r: f"{mean(r.main['table']['A']['baseline_tfidf']['MAP'])} → {mean(r.main['table']['A']['proposed_static']['MAP'])}、"
                   f"{mean(r.main['table']['B']['baseline_tfidf']['MAP'])} → {mean(r.main['table']['B']['proposed_static']['MAP'])}")
    D.tr("§10：集合の特徴量（MAP）", lambda r: sd(st(r, k3)["MAP"]["mean_delta"]))
    D.tr("§10：逐次選択、正解式 8 本以上の MAP（設定 A の (3) − (1)）", lambda r: sd(gc(r, "A", "K>=8", "train - static")["MAP"]["mean_delta"]))

    # ---- 付録 A ----
    D.h("付録 A：比較手法と提案手法の重み（表 6・表 7）")
    D.p("設定 B の列は前回の実行結果と同じ（新旧で一致することを確かめた。下の「設定 B が変わらないことの確認」）。設定 A の列だけを示す。\n")
    D.head(("重み (w_t, w_v)", "旧値（検証 R@20／テスト R@20／テスト MAP）", "新値", ""))
    bw = lambda r, w: r.bl["settings"]["A"]["scorers"]["tfidf"]["by_weight"][w]  # noqa: E731
    for w in o.bl["settings"]["A"]["scorers"]["tfidf"]["by_weight"]:
        D.tr(wfmt(w), lambda r, w=w: f"{f3(bw(r, w)['val']['Recall@20']['mean'])}／{pm(bw(r, w)['test']['Recall@20'])}／{pm(bw(r, w)['test']['MAP'])}")
    for g, gn in (("baseline_tfidf", "TF-IDF"), ("baseline_bm25", "BM25"), ("baseline_e5", "E5")):
        D.tr(f"{gn}、分割ごとに選択（テスト R@20／MAP）", lambda r, g=g: f"{pm(r.main['table']['A'][g]['Recall@20'])}／{pm(r.main['table']['A'][g]['MAP'])}")
    D.L.append("")
    D.head()
    D.tr("本文：検証の Recall@20、w_t ≤ 0.5 の範囲（設定 A）",
         lambda r: rng([bw(r, w)["val"]["Recall@20"]["mean"] for w in ("w50-50", "w40-60", "w30-70", "w20-80", "w10-90", "w05-95", "w00-100")], f3))
    D.tr("本文：(1.0, 0.0) の検証の Recall@20（設定 A）", lambda r: f3(bw(r, "w100-00")["val"]["Recall@20"]["mean"]))
    D.tr("本文：テストの Recall@20、BM25／E5／TF-IDF（設定 A）",
         lambda r: "／".join(mean(r.main["table"]["A"][g]["Recall@20"]) for g in ("baseline_bm25", "baseline_e5", "baseline_tfidf")))
    D.L.append("")
    D.h3("表 7（第 1 段の重みと不正解の選び方、設定 A。N1 は実行に使った値の範囲）")
    D.head(("重み・不正解", "旧値（N1／検証 R@20／テスト R@20／テスト MAP）", "新値", ""))
    for w in WEIGHTS4:
        for neg, nn in (("random", "無作為"), ("hard", "hard negative")):
            D.tr(f"{wfmt(w)}・{nn}",
                 lambda r, w=w, neg=neg: f"{rng(r.n1_run('A', w, neg))}／{f3(r.sel['grid']['A'][f'{w}|{neg}']['val']['Recall@20']['mean'])}／"
                                         f"{f3(r.sel['grid']['A'][f'{w}|{neg}']['test']['Recall@20']['mean'])}／"
                                         f"{f3(r.sel['grid']['A'][f'{w}|{neg}']['test']['MAP']['mean'])}")
    D.L.append("")
    D.head()
    D.tr("本文：hard negative が無作為を検証の Recall@20 で上回った重みの数（設定 A）",
         lambda r: f"{sum(r.sel['grid']['A'][f'{w}|hard']['val']['Recall@20']['mean'] > r.sel['grid']['A'][f'{w}|random']['val']['Recall@20']['mean'] for w in WEIGHTS4)}/4",
         chk=[])
    D.tr("本文：hard negative のときの検証の Recall@20 の範囲（設定 A）",
         lambda r: rng([r.sel["grid"]["A"][f"{w}|hard"]["val"]["Recall@20"]["mean"] for w in WEIGHTS4], f3))
    D.tr("本文：重み (0.3, 0.7) の N1 の規則の値（設定 A）", lambda r: rng([v["w30-70"]["N1"] for v in r.rule["settings"]["A"].values()]),
         chk=["2,000"])
    D.tr("本文：規則の N1 でテストの正解式が候補から漏れる分割があった重み（設定 A）",
         lambda r: "、".join(wfmt(w) for w in WEIGHTS4 if any(v[w]["test_full_at_N1"] < 1 for v in r.rule["settings"]["A"].values())) or "なし",
         chk=[])

    # ---- 付録 B ----
    D.h("付録 B：正解式の数と入力変数の数で分けた結果（表 8、設定 A）")
    D.head(("層", "旧値（件数；R@20 baseline／一括／逐次；MAP baseline／一括／逐次）", "新値", ""))
    for key, order, lab in (("n_correct", [str(i) for i in range(1, 11)], lambda b: f"正解式 {b} 本"),
                            ("n_input", ["<=5", "6-10", "11-15", "16-20", ">=21"],
                             lambda b: {"<=5": "入力変数 5 個以下", "6-10": "入力変数 6〜10 個", "11-15": "入力変数 11〜15 個",
                                        "16-20": "入力変数 16〜20 個", ">=21": "入力変数 21 個以上"}[b])):
        for b in order:
            D.tr(lab(b), lambda r, key=key, b=b: (lambda c: f"{c['n_records']:,}；"
                                                           + "／".join(f3(c[f'{g}|Recall@20']) for g in ('baseline_tfidf', 'proposed_static', 'proposed_train'))
                                                           + "；" + "／".join(f3(c[f'{g}|MAP']) for g in ('baseline_tfidf', 'proposed_static', 'proposed_train')))(
                r.main["buckets"]["A"][key][b]))
    D.tr("本文：延べ件数", lambda r: f"{sum(v['n_records'] for v in r.main['buckets']['A']['n_correct'].values()):,} 件")

    # ---- 付録 C ----
    D.h("付録 C：候補の件数 N1 を変えた結果（図 3・表 9）")
    D.head()
    D.tr("全ケースで正解式がすべて候補に入る N1（設定 A／B）", lambda r: f"{r.cov['settings']['A']['max_min_n1']:,}／{r.cov['settings']['B']['max_min_n1']:,}",
         chk=["1,880"])
    D.tr("図 3 のファイル", lambda r: "figures/fig_n1_full_coverage_ja_1020.png" if r is o else "figures/fig_n1_full_coverage_ja_1021.png", chk=[],
         note="旧図は上書きしていない")
    D.L.append("")
    D.head(("N1（設定 A）", "旧値（被覆率；R@20；MAP；R-P；時間）", "新値", ""))
    for n1 in ("50", "200", "500", "1000", "2000"):
        D.tr(f"{int(n1):,}", lambda r, n1=n1: (lambda v: f"{v['coverage']['mean']:.4f}；{pm(v['Recall@20'])}；{pm(v['MAP'])}；{pm(v['R-Precision'])}；"
                                                        f"{v['time_total_sec']['mean']:.0f} 秒")(r.sweep["N1"]["A"][n1]))
    D.L.append("")
    D.head()
    D.tr("本文：設定 A の Recall@20（N1 = 500／2,000）", lambda r: f"{mean(r.sweep['N1']['A']['500']['Recall@20'])}／{mean(r.sweep['N1']['A']['2000']['Recall@20'])}")
    D.tr("本文：1 回の訓練と評価の時間（設定 A、N1 = 50／2,000）",
         lambda r: f"{r.sweep['N1']['A']['50']['time_total_sec']['mean']:.0f} 秒／{r.sweep['N1']['A']['2000']['time_total_sec']['mean']:.0f} 秒"
                   f"（約 {r.sweep['N1']['A']['2000']['time_total_sec']['mean'] / r.sweep['N1']['A']['50']['time_total_sec']['mean']:.1f} 倍）",
         chk=["54", "162"], note="時間は計算機の負荷で変わる（6 本を並べて実行）")
    if n.sweep:
        chk = n.sweep.get("check_sweep2000_equals_selected")
        if chk:
            D.p(f"\n確認：N1 = 2,000 の行は、§4 の一括の並べ直しと値が一致する（{'設定 A・B とも一致' if all(chk.values()) else chk}）。")

    # ---- 設定 B の確認 ----
    D.h("設定 B が変わらないことの確認")
    same_b = []
    for nm, f in (("表 3（設定 B の 6 手法 × 4 指標）", lambda r: r.main["table"]["B"]),
                  ("表 3 の同じ式も正解の値（設定 B）", lambda r: r.main["table_equiv"]["B"]),
                  ("逐次選択（設定 B）", lambda r: r.greedy["summary"]["B"]),
                  ("baseline の重み別の値（設定 B）", lambda r: r.bl["settings"]["B"]),
                  ("選んだ設定と表 7（設定 B）", lambda r: (r.sel["selected"]["B"], r.sel["grid"]["B"])),
                  ("付録 C（設定 B）", lambda r: {k: v for k, v in r.sweep["N1"]["B"].items()})):
        try:
            a, b = f(o), f(n)
            same_b.append(f"- {nm}：{'一致' if a == b else '**不一致**'}")
        except (KeyError, TypeError):
            same_b.append(f"- {nm}：（未計算）")
    D.p("\n".join(same_b))

    # ---- 照合 ----
    D.h("旧値と報告書の照合")
    if D.miss:
        D.p("次の旧値の数値は、報告書の本文に見つからなかった（書き方の違いか、報告書の数値が実験の値と違う）。")
        D.head(("節", "項目", "数値", ""))
        for sec, item, t in D.miss:
            D.L.append(row(sec, item, t, ""))
    else:
        D.p("旧値として載せた数値（小数と、3 桁ごとのコンマのある数）は、すべて報告書の本文に現れた。")
    return "\n".join(D.L) + "\n", D.miss, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None, help="報告書の .tex（既定は git show db21007:<報告書>）")
    ap.add_argument("--out", default=str(OUTMD))
    a = ap.parse_args()
    if a.report:
        text, rev = Path(a.report).read_text(encoding="utf-8"), Path(a.report).name
    else:
        text = subprocess.run(["git", "show", f"{REPORT_REV}:{REPORT}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        rev = REPORT_REV
    o = Run(K20, "v1_check", K20 / "excluded_cases.json")
    n = Run(K21, "new", K20 / "excluded_cases_v3.json")
    md, miss, C = build(o, n, text, rev)
    Path(a.out).write_text(md, encoding="utf-8")
    print(f"Saved: {a.out}")
    print(f"照合できなかった旧値：{len(miss)} 件")
    for m in miss:
        print("  ", m)
    print("結論の比較：")
    for sec, lab, x, y in C:
        print(f"  {'★' if x != y else ' '} {sec} {lab}：{x} → {y}")


if __name__ == "__main__":
    main()
