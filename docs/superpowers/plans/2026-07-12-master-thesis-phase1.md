# 修士論文 Phase 1（TeX基盤・追加実験・英語図・全章アウトライン）実行計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修士論文の7月マイルストーン成果物一式を作る — コンパイル可能なTeXプロジェクト、有意差検定、greedy 3者比較の設定A拡張、英語ラベル・箱ひげ図の論文用図、そして先生方に見せられる全章3階層アウトラインPDF。

**Architecture:** スペック `docs/superpowers/specs/2026-07-12-master-thesis-design.md` の7章構成に従う。TeXは `thesis/TexSource/`（金上テンプレート）を `thesis/master_thesis/` に移植（uplatex+dvipdfmx+latexmk）。実験は既存の `set_aware_reranker.py` を再利用し、新規スクリプトは検定1本・実行シェル1本・集計1本・図生成1本のみ。アウトラインは各章TeXファイルに英語のitemize 3階層（トピックセンテンス→サポート→小結論）として直接書き、骨子段階でPDF化する。

**Tech Stack:** LaTeX (jsbook, uplatex, dvipdfmx, latexmk), Python 3 (numpy, scipy 1.16, matplotlib, pytest 9), 既存実験基盤 (set_aware_reranker.py / two_stage_query_conditioned.py)。

## Global Constraints

- 論文本文は英語。表紙・謝辞のみ日本語。アウトラインのトピックセンテンスも英語（そのまま本文の段落先頭文になる）。
- **数値はすべて実ファイルから引用**し、TeX中の数値の直前行に `% source: <file> <key>` コメントを付ける。引用元: `experiments/strat_A.json`, `experiments/strat_B.json`, `experiments/strat_full.json`, `experiments/greedy_3way_stats.json`, `experiments/greedy_3way_A_stats.json`(Task 3で生成), `experiments/dof_stop_stats.json`, `experiments/feature_x_difficulty_stats.json`, `experiments/significance_stats.json`(Task 2で生成), `llm_set_full_equiv_results.json`, `llm_set_A_equiv_results.json`, `llm_set_B_equiv_results.json`。
- 「significant / 有意」は検定済みの比較にのみ使う。JSONの `p_ttest: 0.0` は丸めなので本文では `p < 0.001` と書く。
- **X≥8 の +0.0560 は「学習版greedy vs 静的」の差**（vs 推論greedyは+0.0132）。誤帰属禁止。
- 設定Aの variants 文字列は正確に `original,multisource_,dae_`（=1,823件: original 92 + multisource_original 33 + multisource_random_io 198 + multisource_v3 500 + dae 1,000）。旧 `original,multisource_v3,dae_`（1,592件）は使わない。
- シード列は全実験共通: `42 123 456 789 1024 2024 3141 5926 7777 9999`。
- 10シード比較の図は箱ひげ図（G7）。LLM直接生成はシードなし単発評価（n=48〜50）なので点表示。
- 図は `thesis/master_thesis/figures/` に .png と .pdf の両方を英語ラベル・matplotlib既定フォント（日本語フォント設定を入れない）で出力。suptitleは付けない（キャプションはTeX側）。
- コミットはタスク末ごと。**post-commitフックが自動pushする**。`experiments/xg_A/` の生シード別出力はコミットしない（既存の xg/ xd/ xd_dof/ と同じ扱い）。stats JSON・スクリプト・図はコミットする。
- 各実験系タスクの最後に `docs/development_log.tex` へ追記する（先に `tail -40 docs/development_log.tex` で既存書式を確認し、同じ書式で日付・変更内容2〜3文を追加）。
- TeXビルドコマンドは常に: `cd thesis/master_thesis && latexmk main.tex`（latexmkrcがuplatex+dvipdfmxを指定）。成功判定は `main.pdf` の更新とexitコード0。

## File Structure

| パス | 責務 |
|---|---|
| `thesis/master_thesis/main.tex` | プリアンブル・表紙・Abstract・章input・References・謝辞枠 |
| `thesis/master_thesis/latexmkrc` | uplatex→dvipdfmx ビルド定義 |
| `thesis/master_thesis/.gitignore` | ビルド生成物除外 |
| `thesis/master_thesis/{Introduction,RelatedWork,Method,Dataset,Experiment,ResultsAndDiscussion,Conclusion,Appendix}.tex` | 各章（Phase 1ではアウトライン） |
| `thesis/master_thesis/figures/` | 論文用英語図（Task 4-5が生成） |
| `analyze_significance.py` + `tests/test_significance.py` | baseline vs 本手法の対応あり検定（新規） |
| `run_greedy_3way_A.sh` | 設定Aでの3者比較実行（30ジョブ） |
| `analyze_greedy_3way_A.py` | 設定A 3者比較の集計・検定 |
| `generate_figures_thesis.py` | 論文用英語図の一括生成 |

---

### Task 1: TeXプロジェクト骨格の作成とコンパイル確認

**Files:**
- Create: `thesis/master_thesis/main.tex`
- Create: `thesis/master_thesis/latexmkrc`
- Create: `thesis/master_thesis/.gitignore`
- Create: `thesis/master_thesis/Introduction.tex`（ほか章ファイル7本、内容はStep 3のとおり）
- Create: `thesis/master_thesis/figures/.gitkeep`

**Interfaces:**
- Produces: ビルドコマンド `cd thesis/master_thesis && latexmk main.tex` → `main.pdf`。図ディレクトリ `thesis/master_thesis/figures/`（Task 4-5が書き込む）。章ファイル名（Task 6-12が上書きする）。

- [ ] **Step 1: ディレクトリと latexmkrc / .gitignore を作る**

`thesis/master_thesis/latexmkrc`:
```perl
$latex = 'uplatex -interaction=nonstopmode -halt-on-error %O %S';
$bibtex = 'upbibtex %O %B';
$dvipdf = 'dvipdfmx %O -o %D %S';
$makeindex = 'mendex %O -o %D %S';
$pdf_mode = 3;
$max_repeat = 5;
```

`thesis/master_thesis/.gitignore`:
```
*.aux
*.log
*.dvi
*.toc
*.out
*.synctex.gz
*.fls
*.fdb_latexmk
*.bbl
*.blg
```

- [ ] **Step 2: main.tex を書く**

`thesis/TexSource/main.tex` を土台に以下の完成形を書く（graphicx二重読込は解消済み、documentclassに `uplatex` を明示）:

```latex
\documentclass[uplatex,a4j,onecolumn,11pt,openany,english,oneside]{jsbook}
\usepackage{array,enumerate}
\usepackage[dvipdfmx]{graphicx,color}
\usepackage{amsmath}
\usepackage{bm}
\usepackage{amssymb}
\usepackage{comment}
\usepackage{algorithmic,algorithm}
\usepackage{otf}
\usepackage{float}
\usepackage{colortbl}
\usepackage{remreset}
\usepackage{ascmac}
\usepackage{multirow}
\usepackage{appendix}
\usepackage{url}
\usepackage{setspace}
\usepackage{subcaption}
\usepackage{tabularx}
\usepackage{lscape}
\usepackage{threeparttable}
\usepackage{caption}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{arydshln}
\captionsetup[table]{labelsep=period, labelfont=bf, justification=centering, singlelinecheck=off}
\captionsetup[figure]{labelsep=period, labelfont=bf, justification=centering, singlelinecheck=off}

% 図表を通し番号にする（テンプレート踏襲）
\makeatletter
	\@removefromreset{figure}{chapter}
	\def\thefigure{\arabic{figure}}
	\@removefromreset{table}{chapter}
	\def\thetable{\arabic{table}}
	\@removefromreset{equation}{chapter}
	\def\theequation{\arabic{equation}}
\makeatother
\setlength{\textwidth}{\fullwidth}
\setlength{\evensidemargin}{\oddsidemargin}
\setstretch{1.3}

\newcommand{\argmax}{\mathop{\rm arg~max}\limits}
\newcommand{\argmin}{\mathop{\rm arg~min}\limits}

\renewcommand{\figurename}{Fig. }
\renewcommand{\tablename}{Table }
\renewcommand{\bibname}{References}
\renewcommand{\contentsname}{Contents}
\renewcommand{\prechaptername}{Chapter }
\renewcommand{\postchaptername}{}

\makeatletter
 \def\@cite#1{\textsuperscript{#1)}}
 \def\@biblabel#1{#1)}
\makeatother

\setlength{\abovecaptionskip}{0pt}
\setlength{\belowcaptionskip}{5pt}

\begin{document}
\thispagestyle{empty}
%%%%% 表紙 %%%%%
\begin{center}
\vspace*{3mm}
{\huge  修士論文}
\vspace{10mm}

{\huge Set-aware Retrieval of Equation Sets\\ from Multiple Documents\\ for Automated Physical Model Building\\}
\vspace{5mm}
{\LARGE 物理モデル自動構築に向けた複数文献からの\\
\vspace{5mm}
集合を考慮した数式検索}

\vspace{12mm}

{\huge ~指導教員　　加納　学　　教授}\\
\vspace{3mm}
\vspace{12mm}

{\huge 京都大学大学院情報学研究科}\\
\vspace{3mm}
{\huge システム科学専攻修士課程}\\
\vspace{3mm}
{\huge 令和 7 年度入学}\\
\vspace{30mm}

{\huge DING YIYANG}\\
\vspace{12mm}
{\huge 令和 9 年 2 月提出}
\end{center}

\newpage
%%%%% Abstract（初稿。全章完成後に書き直す） %%%%%
\thispagestyle{empty}
{\huge \bf Abstract}
\\
\\
\\
\\
\\
Physical models are essential for digital twins and play an important role in process design and operation in the process industry.
Building a physical model requires engineers to survey a vast number of documents and to assemble the equations scattered across them into a solvable system.
To automate this work, we aim to develop an automated physical model builder (AutoPMoB).
This study addresses one of its core tasks: given a description of the desired model and its input and output variables, retrieve the set of equations that composes the model from an equation database built from multiple documents.

We propose a two-stage retrieval method.
The first stage narrows candidates with seven query--equation features such as text similarity, and the second stage reranks them with three set-aware features that evaluate each candidate against the already selected set: complementarity, coherence, and domain agreement.
We further introduce a learned greedy selection trained with teacher forcing and a self-terminating criterion that stops retrieval when the selected equations form a closed system with zero degrees of freedom.
% source: experiments/strat_A.json results.*.Recall@K_correct
On a database of 11,146 equations from 361 sources, the proposed method improves Recall@K from 0.521 to 0.743 against a classical information-retrieval baseline and also outperforms direct generation by a large language model.
Analyses show that the structural set-aware features cause the improvement, that the improvement grows on harder cases, and that the self-terminating criterion returns solvable systems without knowing the answer size.

\newpage
%%%%% 目次 %%%%%
\pagenumbering{roman}
\setcounter{tocdepth}{4}
\tableofcontents
\thispagestyle{empty}
\newpage

\newpage
\pagenumbering{arabic}
\chapter{Introduction}
\label{Introduction}
\input{Introduction}
\clearpage

\chapter{Related Work}
\label{RelatedWork}
\input{RelatedWork}
\clearpage

\chapter{Set-aware Equation-Set Retrieval}
\label{Method}
\input{Method}
\clearpage

\chapter{Equation Database and Test Case Construction}
\label{Dataset}
\input{Dataset}
\clearpage

\chapter{Experiments}
\label{Experiment}
\input{Experiment}
\clearpage

\chapter{Results and Discussion}
\label{ResultsAndDiscussion}
\input{ResultsAndDiscussion}
\clearpage

\chapter{Conclusion}
\label{Conclusion}
\input{Conclusion}
\clearpage

%%%%% References（Phase 2で拡充。現時点は確実に使う実在文献のみ） %%%%%
\begin{thebibliography}{99}

\bibitem{simulink}
Simulink - Simulation and Model-Based Design - MATLAB \& Simulink.
\ \url{https://www.mathworks.com/products/simulink.html}.
\ (Accessed on 2026/07/12).

\bibitem{aspen}
AspenTech | Asset Optimization Software.
\ \url{https://www.aspentech.com/}.
\ (Accessed on 2026/07/12).

\bibitem{ProcessBERT}
金上和毅.
\ Development of ProcessBERT for Judging Equivalence of Variables among Multiple Documents.
\ 京都大学大学院情報学研究科システム科学専攻 修士論文, 2022.

\bibitem{BERT}
J. Devlin, M. Chang, K. Lee, K. Toutanova.
\ BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
\ {\it Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies}, {\bf 1}, pp.4171-4186, 2019.

\end{thebibliography}

%%%%% 謝辞（最終段階で執筆） %%%%%
%\newpage
%\chapter*{謝辞}

\newpage
\appendix
\def\thechapter{Appendix \Alph{chapter}}
\input{Appendix}
\thispagestyle{empty}
\end{document}
```

- [ ] **Step 3: 章ファイル8本を作る**

各章ファイルは1行のTeXコメントのみで作成（内容はTask 6-12で上書きする）。例 `Introduction.tex`:
```latex
% Chapter 1: Introduction — outline inserted by Task 6 of docs/superpowers/plans/2026-07-12-master-thesis-phase1.md
```
同様に `RelatedWork.tex`（Task 7）, `Method.tex`（Task 8）, `Dataset.tex`（Task 9）, `Experiment.tex`（Task 10）, `ResultsAndDiscussion.tex`（Task 11）, `Conclusion.tex`（Task 12）, `Appendix.tex`（Task 12）。

- [ ] **Step 4: コンパイル確認**

Run: `cd thesis/master_thesis && latexmk main.tex && ls -la main.pdf`
Expected: exitコード0、`main.pdf` 生成（表紙・Abstract・目次・7章見出し・References ≈ 12〜16ページ）。jsbook+subcaptionの "Unsupported document class" 警告は出ても無視してよい（旧修論でも動作実績あり）。

- [ ] **Step 5: 目視確認**

Run: `cd thesis/master_thesis && pdftotext -f 1 -l 3 main.pdf - | head -40`
Expected: 表紙に「Set-aware Retrieval of Equation Sets」「DING YIYANG」「令和 9 年 2 月提出」、Abstractの冒頭文が見える。

- [ ] **Step 6: Commit**

```bash
git add thesis/master_thesis
git commit -m "feat(thesis): 修論TeXプロジェクト骨格を新設（uplatex+dvipdfmx、7章構成）"
```

---

### Task 2: 有意差検定スクリプト（TDD）

**Files:**
- Create: `tests/test_significance.py`
- Create: `analyze_significance.py`
- Output: `experiments/significance_stats.json`

**Interfaces:**
- Consumes: `experiments/strat_A.json` / `experiments/strat_B.json` の `results[mode]["per_seed"]`（長さ10のlist、各要素は `{"seed": int, "Recall@K_correct": float, "MAP": float, "Recall@20": float, ...}`。seed順は両mode共通 `[42,123,456,789,1024,2024,3141,5926,7777,9999]`）。mode名は `"baseline"` と `"reranker-10S"`。
- Produces: `analyze_significance.py` の関数 `load_per_seed(path, mode, metric) -> list[float]` と `paired_stats(base: list, rer: list) -> dict`（キー: n, mean_base, mean_rer, mean_delta, std_delta, p_ttest, p_wilcoxon, cohen_dz）。出力 `experiments/significance_stats.json` は `{"A": {"file":..., "seeds": [...], "Recall@K_correct": {...paired_stats...}, "MAP": {...}, "Recall@20": {...}}, "B": {...}}`。Task 11（第6章アウトライン）がこのJSONのp値を引用する。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_significance.py`:
```python
import json

from analyze_significance import load_per_seed, paired_stats


def test_paired_stats_known_values():
    base = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52, 0.47, 0.51]
    rer = [0.70, 0.74, 0.69, 0.72, 0.71, 0.75, 0.73, 0.74, 0.66, 0.72]
    s = paired_stats(base, rer)
    assert s["n"] == 10
    # mean(rer)=0.716, mean(base)=0.503 → delta=0.213
    assert abs(s["mean_delta"] - 0.213) < 1e-9
    # 10ペア全てで rer > base のとき Wilcoxon 両側の最小 p = 2/2^10 ≈ 0.00195
    assert abs(s["p_wilcoxon"] - 0.00195) < 1e-4
    assert s["p_ttest"] < 0.001
    assert s["cohen_dz"] > 3.0


def test_load_per_seed(tmp_path):
    doc = {"config": {}, "results": {"baseline": {"per_seed": [
        {"seed": 42, "Recall@K_correct": 0.5, "MAP": 0.6, "Recall@20": 0.8},
        {"seed": 123, "Recall@K_correct": 0.4, "MAP": 0.5, "Recall@20": 0.7},
    ]}}}
    p = tmp_path / "x.json"
    p.write_text(json.dumps(doc))
    assert load_per_seed(p, "baseline", "Recall@K_correct") == [0.5, 0.4]
    assert load_per_seed(p, "baseline", "MAP") == [0.6, 0.5]
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd /Users/kazuhiromiyamura/Desktop/AutoPMob && python3 -m pytest tests/test_significance.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'analyze_significance'`）

- [ ] **Step 3: 実装を書く**

`analyze_significance.py`:
```python
#!/usr/bin/env python3
"""baseline vs reranker-10S の対応あり有意差検定（層化分割・10乱数）.

入力: experiments/strat_A.json / experiments/strat_B.json の results[mode]["per_seed"]
出力: experiments/significance_stats.json
      （Wilcoxon符号順位検定・対応あり t 検定・Cohen dz、指標は Recall@K_correct / MAP / Recall@20）
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

METRICS = ["Recall@K_correct", "MAP", "Recall@20"]
SETTINGS = {"A": "strat_A.json", "B": "strat_B.json"}


def load_per_seed(path, mode, metric):
    doc = json.load(open(path))
    return [p[metric] for p in doc["results"][mode]["per_seed"]]


def paired_stats(base, rer):
    b = np.asarray(base, dtype=float)
    r = np.asarray(rer, dtype=float)
    d = r - b
    t = stats.ttest_rel(r, b)
    w = stats.wilcoxon(r, b)
    return {
        "n": int(len(d)),
        "mean_base": round(float(b.mean()), 4),
        "mean_rer": round(float(r.mean()), 4),
        "mean_delta": round(float(d.mean()), 4),
        "std_delta": round(float(d.std(ddof=1)), 4),
        "p_ttest": round(float(t.pvalue), 5),
        "p_wilcoxon": round(float(w.pvalue), 5),
        "cohen_dz": round(float(d.mean() / d.std(ddof=1)), 3),
    }


def main():
    out = {}
    for name, fname in SETTINGS.items():
        path = EXP / fname
        doc = json.load(open(path))
        out[name] = {
            "file": fname,
            "seeds": [p["seed"] for p in doc["results"]["baseline"]["per_seed"]],
        }
        for m in METRICS:
            out[name][m] = paired_stats(
                load_per_seed(path, "baseline", m),
                load_per_seed(path, "reranker-10S", m),
            )
    dst = EXP / "significance_stats.json"
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python3 -m pytest tests/test_significance.py -v`
Expected: 2 passed

- [ ] **Step 5: 実データで実行し結果を確認**

Run: `python3 analyze_significance.py`
Expected: `experiments/significance_stats.json` 生成。値の妥当性チェック:
- `A.Recall@K_correct.mean_base ≈ 0.5206`, `mean_rer ≈ 0.7429`, `mean_delta ≈ +0.2223`
- `B.Recall@K_correct.mean_base ≈ 0.4661`, `mean_rer ≈ 0.6890`
- 10シード全てで reranker > baseline のはずなので `p_wilcoxon = 0.00195`（両側最小値）になる見込み。

- [ ] **Step 6: development_log.tex に追記**

`tail -40 docs/development_log.tex` で書式確認後、同書式で追記（内容例: 「baseline と reranker-10S の10乱数対応あり Wilcoxon 符号順位検定を実装（analyze_significance.py）。設定A/Bとも Recall@K・MAP・Recall@20 の全指標で p=0.00195。」実際のp値は significance_stats.json の値を使う）。

- [ ] **Step 7: Commit**

```bash
git add tests/test_significance.py analyze_significance.py experiments/significance_stats.json docs/development_log.tex
git commit -m "feat: baseline vs 本手法の対応ありWilcoxon検定を追加（設定A/B・10シード）"
```

---

### Task 3: greedy 3者比較の設定A拡張（30ジョブ実行＋集計）

**Files:**
- Create: `run_greedy_3way_A.sh`
- Create: `analyze_greedy_3way_A.py`（`analyze_greedy_3way.py` のコピーに3点の編集）
- Output: `experiments/xg_A/{static,infer,train}__{seed}.json`（30本・コミットしない）、`experiments/greedy_3way_A_stats.json`、`docs/figures/fig_greedy_3way_A.png`

**Interfaces:**
- Consumes: `set_aware_reranker.py` のCLI（`--modes reranker-10S [--greedy|--train-greedy] --seed-list <s> --split stratified --variants "original,multisource_,dae_" --top-k 200 --save-per-case --output <path>`）。
- Produces: `experiments/greedy_3way_A_stats.json` — 構造は既存 `experiments/greedy_3way_stats.json` と同一（`overall.{static,infer,train}.{mean,std,n_seeds}`、`tests.{train_vs_static,train_vs_infer,infer_vs_static}`、`tests_hard_X8plus`、`by_n_correct`）。Task 5（図）と Task 11（第6章アウトライン）が読む。

- [ ] **Step 1: run_greedy_3way_A.sh を書く**

既存 `run_greedy_3way.sh` と同じ構造で、差分は OUTDIR / VARIANTS の2行:
```bash
#!/bin/bash
# 3者比較（設定A: 原型+複数文献+DAE = 1,823件）。run_greedy_3way.sh の設定A版。
set -u
cd "$(dirname "$0")"
export OUTDIR=experiments/xg_A
export VARIANTS="original,multisource_,dae_"
export TOPK=200
mkdir -p "$OUTDIR"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"
run_one() {
  local config="$1" seed="$2"
  local out="$OUTDIR/${config}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $config seed=$seed (exists)"; return 0; fi
  local flag=""
  case "$config" in
    static) flag="" ;;
    infer)  flag="--greedy" ;;
    train)  flag="--train-greedy" ;;
  esac
  if python3 set_aware_reranker.py --modes reranker-10S $flag \
       --seed-list "$seed" --split stratified --variants "$VARIANTS" --top-k "$TOPK" \
       --save-per-case --output "$out" > "$OUTDIR/${config}__${seed}.log" 2>&1; then
    echo "done $config seed=$seed"
  else
    echo "FAIL $config seed=$seed (see $OUTDIR/${config}__${seed}.log)"
  fi
}
export -f run_one
JOBS=$(mktemp)
for c in static infer train; do for s in $SEEDS; do echo "$c $s"; done; done > "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"
xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"' < "$JOBS"
rm -f "$JOBS"
echo "ALL JOBS FINISHED"
```
Run: `chmod +x run_greedy_3way_A.sh`

- [ ] **Step 2: スモークテスト（1ジョブだけ直接実行）**

Run:
```bash
mkdir -p experiments/xg_A
python3 set_aware_reranker.py --modes reranker-10S \
  --seed-list 42 --split stratified --variants "original,multisource_,dae_" \
  --top-k 200 --save-per-case --output experiments/xg_A/static__42.json
python3 -c "import json; d=json.load(open('experiments/xg_A/static__42.json')); r=d['results']['reranker-10S']; print('n per_case:', len(r['per_case']), 'R@K mean:', r['Recall@K_correct'])"
```
Expected: JSONが生成され、per_case は約400〜460件（テスト分割ぶん）、Recall@K_correct の mean は 0.70〜0.80 程度（strat_A.json の 0.7429 と同水準。top-k が50→200 なので厳密一致はしない）。

- [ ] **Step 3: 全30ジョブを実行**

Run: `./run_greedy_3way_A.sh 2>&1 | tee experiments/xg_A/run.log`
（1ジョブ=MLP訓練込みのため数分×30。スクリプトは非空JSONをskipする再開可能設計なので、中断しても再実行すればよい。バックグラウンド実行し完了を待つ間に Task 4 を進めてよい — Task 5 と Task 11 の前までに完了していること。）
Expected: 最終行 `ALL JOBS FINISHED`、`ls experiments/xg_A/*.json | wc -l` → 30、FAIL行なし。

- [ ] **Step 4: analyze_greedy_3way_A.py を作る**

`cp analyze_greedy_3way.py analyze_greedy_3way_A.py` してから、次の3点だけを編集する:
1. 入力ディレクトリ: `XG = ROOT / "experiments" / "xg"` → `XG = ROOT / "experiments" / "xg_A"`
2. 出力: `experiments/greedy_3way_stats.json` → `experiments/greedy_3way_A_stats.json`、`docs/figures/fig_greedy_3way.png` → `docs/figures/fig_greedy_3way_A.png`
3. 図中の文字列: 「DAEのみ」→「設定A」、「(DAE)」→「(設定A)」、suptitle「学習版greedy 3者比較（DAEのみ・層化分割）」→「学習版greedy 3者比較（設定A・層化分割）」
（ファイル名正規表現 `(static|infer|train)__(\d+).json` と METRIC="Recall@K_correct" は変更しない。）

- [ ] **Step 5: 集計を実行して確認**

Run: `python3 analyze_greedy_3way_A.py && python3 -c "import json; d=json.load(open('experiments/greedy_3way_A_stats.json')); print(json.dumps(d['overall'], indent=1)); print(json.dumps(d['tests']['train_vs_infer'], indent=1))"`
Expected: `overall` に static/infer/train の mean±std（n_seeds=10）。static の mean は 0.72〜0.78 程度（Step 2 と同水準）。DAE版（0.7237<0.7556<0.7649）と同様に static < infer ≤ train の順になるかを確認し、**ならなかった場合も値をそのまま記録**（結果の向きは論文で正直に書く。G2）。

- [ ] **Step 6: development_log.tex に追記**

書式確認の上、追記内容: 設定A（1,823件）で3者比較を10シード実行した旨、overall 3値と train_vs_infer / train_vs_static の p_wilcoxon（greedy_3way_A_stats.json の実値）。

- [ ] **Step 7: Commit**

```bash
git add run_greedy_3way_A.sh analyze_greedy_3way_A.py experiments/greedy_3way_A_stats.json docs/figures/fig_greedy_3way_A.png docs/development_log.tex
git commit -m "feat: greedy3者比較を設定A(1,823件)に拡張（10シード・検定つき）"
```
（`experiments/xg_A/` はコミットしない。）

---

### Task 4: 論文用英語図 その1（データセット・分割・手法比較・特性別）

**Files:**
- Create: `generate_figures_thesis.py`
- Output: `thesis/master_thesis/figures/fig_dataset.{png,pdf}`, `fig_split_balance.{png,pdf}`, `fig_method_comparison.{png,pdf}`, `fig_characteristic.{png,pdf}`

**Interfaces:**
- Consumes: `training_cases.json`、`experiments/strat_{full,A,B}.json`（`per_seed` / `per_case`）、`llm_set_{full,A,B}_equiv_results.json`（トップレベル `Recall@K_correct` と `coverage` のスカラー）、既存 `generate_figures_strat.py`（`dataset()` L53-101 と `characteristic()` L128-252 の移植元）、`generate_figures_v6.py`（`fig_split_balance()` L586-635 の移植元）。
- Produces: `generate_figures_thesis.py` — 関数 `dataset()`, `split_balance()`, `method_comparison()`, `characteristic()`（Task 5 が同ファイルに関数を追加する）。共有ヘルパ `FIG`(出力先Path), `save(fig, name)`, `_box(ax, data, positions, colors)`。

- [ ] **Step 1: スクリプトの骨格と共有ヘルパを書く**

`generate_figures_thesis.py` の冒頭（この内容で新規作成）:
```python
#!/usr/bin/env python3
"""修士論文用の英語ラベル図を一括生成する.

出力: thesis/master_thesis/figures/*.{png,pdf}
データ: training_cases.json / experiments/strat_*.json / llm_set_*_equiv_results.json /
        experiments/xg/ xg_A/ xd_dof/（シード別生データ、箱ひげ用）/
        experiments/ablation_x_difficulty.json
方針: 10シード比較は箱ひげ図（G7）。LLM直接生成はシードなし単発評価のため点表示。
      日本語フォント設定は入れない（英語ラベルのみ）。suptitleは付けない。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
FIG = ROOT / "thesis" / "master_thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams["pdf.fonttype"] = 42

C_BASE = "#9aa0a6"   # baseline: gray
C_PROP = "#1a73e8"   # proposed: blue
C_LLM = "#d93025"    # LLM: red
C_INFER = "#f9ab00"  # inference greedy: amber
C_TRAIN = "#188038"  # trained greedy: green


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / name}.png/.pdf")


def _box(ax, data, positions, colors, widths=0.55):
    """10シード値の箱ひげ＋ジッター散点（fig_v6_boxplot のスタイル踏襲）."""
    bp = ax.boxplot(data, positions=positions, widths=widths, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, data):
        x = rng.normal(pos, 0.05, size=len(vals))
        ax.scatter(x, vals, s=12, color="black", alpha=0.5, zorder=3)
    return bp


def _per_seed(path, mode, metric="Recall@K_correct"):
    doc = json.load(open(path))
    return [p[metric] for p in doc["results"][mode]["per_seed"]]
```

- [ ] **Step 2: method_comparison() を書く（箱ひげ版・新規実装）**

同ファイルに追記:
```python
def method_comparison():
    """3設定（全データ/設定A/設定B）× baseline・本手法の箱ひげ＋LLM点."""
    settings = [
        ("strat_full.json", "llm_set_full_equiv_results.json", "All cases\n(incl. augmentation, 2,838)"),
        ("strat_A.json", "llm_set_A_equiv_results.json", "Setting A\n(1,823)"),
        ("strat_B.json", "llm_set_B_equiv_results.json", "Setting B\n(DAE only, 1,000)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    centers = np.arange(len(settings)) * 2.6
    for i, (sf, lf, label) in enumerate(settings):
        base = _per_seed(EXP / sf, "baseline")
        prop = _per_seed(EXP / sf, "reranker-10S")
        _box(ax, [base, prop], [centers[i] - 0.45, centers[i] + 0.45], [C_BASE, C_PROP])
        llm = json.load(open(ROOT / lf))
        ax.scatter([centers[i] + 1.05], [llm["Recall@K_correct"]], marker="D", s=55,
                   color=C_LLM, zorder=4)
        ax.hlines(llm["coverage"], centers[i] + 0.85, centers[i] + 1.25,
                  colors=C_LLM, linestyles="--", lw=1.4)
    ax.set_xticks(centers)
    ax.set_xticklabels([s[2] for s in settings])
    ax.set_ylabel("Recall@K")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_BASE, alpha=0.85, label="Baseline (classical IR)"),
        plt.Rectangle((0, 0), 1, 1, fc=C_PROP, alpha=0.85, label="Proposed (reranker-10S)"),
        plt.Line2D([], [], marker="D", ls="", color=C_LLM, label="LLM direct generation (single run, n≈50)"),
        plt.Line2D([], [], ls="--", color=C_LLM, label="LLM coverage (rank-free)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, ncol=2)
    save(fig, "fig_method_comparison")
```

- [ ] **Step 3: dataset() と characteristic() を移植する**

`generate_figures_strat.py` から `dataset()`（L53-101）と `characteristic()`（バケット集計 `bucket_stats` L132-147 を含む L128-252）を `generate_figures_thesis.py` にコピーし、次を適用する。**集計ロジック・バケット境界は一切変更しない。**
- 出力: `FIG` は本ファイルの定義（thesis/master_thesis/figures）を使い、保存は `save(fig, "fig_dataset")` / `save(fig, "fig_characteristic")` に置き換える。
- `suptitle` 行は削除。
- ラベル文字列の置換表（左＝現状の日本語、右＝置換後の英語）:

| 日本語 | 英語 |
|---|---|
| 原型 | Original |
| 言い換え | Paraphrase (aug.) |
| 無作為入替 | Random I/O (aug.) |
| 複数文献 | Multi-source |
| DAE\n（微分代数・難） | DAE (hard) |
| ケース数 | Number of cases |
| (a) 訓練ケースの内訳（…） | (a) Composition of the 2,838 cases |
| 設定 A：評価対象（計 1,823 件） | Setting A: evaluation target (1,823 cases) |
| 正解モデルに含まれる数式の数 | Number of equations in the true model |
| (b) 正解式数（問題の難しさ）の分布（…） | (b) Distribution of the number of correct equations |
| 単一式 | Single-equation |
| 複数式 | Multi-equation |
| 5 以下 / 6-10 / 11-15 / 16-20 / 21 以上 | ≤5 / 6–10 / 11–15 / 16–20 / ≥21 |
| baseline（古典 IR） | Baseline (classical IR) |
| reranker-10S（本手法） | Proposed (reranker-10S) |
| Recall@K（正解集合の再現率） | Recall@K |
| (a) 数式が多いほど難しくなる | (a) By number of equations |
| 1 つ / 2 つ / 3 つ以上 | 1 / 2 / ≥3 |
| 数式が由来する文献（ソース）の数 | Number of source documents |
| (b) ソース数による差は小さい（…） | (b) By number of sources |
| 入力変数の数 | Number of input variables |
| (c) 入力変数が多いほど難しくなる | (c) By number of input variables |

- [ ] **Step 4: split_balance() を移植する**

`generate_figures_v6.py` の `fig_split_balance()`（L586-635）を `generate_figures_thesis.py` に `split_balance()` としてコピーする。この関数は `set_aware_reranker.stratified_src_split` と `two_stage_query_conditioned` を import して seed=42 の分割をライブ再計算する — **import と計算ロジックは変更しない**。変更は保存先（`save(fig, "fig_split_balance")`）、suptitle削除、ラベル置換のみ:

| 日本語 | 英語 |
|---|---|
| 正解式数 | #Equations |
| 入力変数数 | #Input variables |
| 出力変数数 | #Output variables |
| 横断ソース数 | #Sources spanned |
| 訓練 | Train |
| テスト | Test |
| 割合 | Fraction |
| 値 | Value |

- [ ] **Step 5: `__main__` を付けて実行**

ファイル末尾:
```python
if __name__ == "__main__":
    dataset()
    split_balance()
    method_comparison()
    characteristic()
```
Run: `python3 generate_figures_thesis.py && ls -la thesis/master_thesis/figures/`
Expected: `fig_dataset` / `fig_split_balance` / `fig_method_comparison` / `fig_characteristic` の .png と .pdf 計8ファイル。エラーなし。

- [ ] **Step 6: 目視確認**

各 .png を開き、(1) 日本語が一切残っていない（豆腐□も出ていない）、(2) fig_method_comparison が箱ひげ＋LLM点になっている、(3) 値の水準がゼミ報告の図（docs/figures/fig_strat_*.png）と同じ、を確認する。

- [ ] **Step 7: Commit**

```bash
git add generate_figures_thesis.py thesis/master_thesis/figures
git commit -m "feat(thesis): 論文用英語図その1（データセット・分割・手法比較箱ひげ・特性別）"
```

---

### Task 5: 論文用英語図 その2（機構・greedy・DoF停止）

**Files:**
- Modify: `generate_figures_thesis.py`（関数3本を追加）
- Output: `thesis/master_thesis/figures/fig_mechanism.{png,pdf}`, `fig_greedy_3way.{png,pdf}`, `fig_dof_stop.{png,pdf}`

**Interfaces:**
- Consumes: `experiments/ablation_x_difficulty.json`（`results[mode]["per_case"]`、per_case キー: seed, n_correct, n_input, Recall@K_correct）、`experiments/xg/` と `experiments/xg_A/`（Task 3。per_case キー: seed, n_correct, Recall@K_correct）、`experiments/xd_dof/{dae,A}__{seed}.json`（per_case キー: seed, n_correct, set_f1_dof, set_f1_oracleK, set_exact_dof, closed_dof, closed_oracleK）、`experiments/dof_stop_stats.json`（閉包率の注記値）。
- Produces: 図3点。Task 11（第6章アウトライン）が参照名を使う。

- [ ] **Step 0: 前提データの存在確認**

Run:
```bash
ls experiments/ablation_x_difficulty.json experiments/xd_dof/*.json | head -5
ls experiments/xg/*.json | wc -l; ls experiments/xg_A/*.json | wc -l
python3 -c "import json; d=json.load(open('experiments/ablation_x_difficulty.json')); print(list(d['results'].keys()))"
```
Expected: ファイルが存在し、xg=30以上・xg_A=30。最後のコマンドで mode キー一覧（`reranker-7`, `reranker-7+Comp`, `reranker-7+Coh`, `reranker-7+Dom`, `reranker-10S` の5つ）が表示される。**もし表示されたmode名がこれと異なる場合は、以降のコードの MODE 定数を表示された実名に合わせる。** もし `ablation_x_difficulty.json` が無ければ `./run_feature_x_difficulty.sh && python3 merge_xd.py` で再生成（再開可能）。

- [ ] **Step 1: mechanism() を追加**

```python
MODES_MECH = [
    ("reranker-7+Comp", "+Complementarity (gComp)", "#1a73e8"),
    ("reranker-7+Coh", "+Coherence (gCoh)", "#188038"),
    ("reranker-7+Dom", "+Domain (gDom)", "#9aa0a6"),
    ("reranker-10S", "All three (10S)", "#d93025"),
]
BASE_MECH = "reranker-7"


def _seed_bucket_means(per_case, bucket_fn):
    """per_case を (seed, bucket) 別に平均 → {bucket: {seed: mean}}."""
    acc = {}
    for rec in per_case:
        b = bucket_fn(rec)
        if b is None:
            continue
        acc.setdefault(b, {}).setdefault(rec["seed"], []).append(rec["Recall@K_correct"])
    return {b: {s: float(np.mean(v)) for s, v in d.items()} for b, d in acc.items()}


def _lift_sem(base_by, mode_by):
    """シード対応リフトの mean±SEM を bucket ごとに返す."""
    out = {}
    for b in sorted(set(base_by) & set(mode_by)):
        seeds = sorted(set(base_by[b]) & set(mode_by[b]))
        d = np.array([mode_by[b][s] - base_by[b][s] for s in seeds])
        if len(d) >= 2:
            out[b] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))
    return out


def mechanism():
    doc = json.load(open(EXP / "ablation_x_difficulty.json"))
    res = doc["results"]
    nin_edges = [(1, 3, "1–3"), (4, 6, "4–6"), (7, 10, "7–10"), (11, 15, "11–15"), (16, 10**9, "≥16")]

    def by_x(rec):
        return rec["n_correct"] if 1 <= rec["n_correct"] <= 10 else None

    def by_nin(rec):
        for lo, hi, lab in nin_edges:
            if lo <= rec["n_input"] <= hi:
                return lab
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for bucket_fn, ax, xlabel, order in [
        (by_x, axes[0], "Number of equations in the true model", list(range(1, 11))),
        (by_nin, axes[1], "Number of input variables", [e[2] for e in nin_edges]),
    ]:
        base_by = _seed_bucket_means(res[BASE_MECH]["per_case"], bucket_fn)
        for mode, label, color in MODES_MECH:
            ls = _lift_sem(base_by, _seed_bucket_means(res[mode]["per_case"], bucket_fn))
            xs = [b for b in order if b in ls]
            ax.errorbar(range(len(xs)), [ls[b][0] for b in xs], yerr=[ls[b][1] for b in xs],
                        marker="o", ms=4, lw=1.6, capsize=3, label=label, color=color)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len([b for b in order if b in ls])))
        ax.set_xticklabels([str(b) for b in order if b in ls])
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall@K lift over 7-feature reranker")
    axes[0].set_title("(a) Stratified by number of equations")
    axes[1].set_title("(b) Stratified by number of input variables")
    axes[0].legend(fontsize=8.5)
    save(fig, "fig_mechanism")
```

- [ ] **Step 2: greedy_3way() を追加**

```python
def _greedy_seed_means(dirname, config):
    vals = {}
    for f in sorted((EXP / dirname).glob(f"{config}__*.json")):
        doc = json.load(open(f))
        for mode_res in doc["results"].values():
            for rec in mode_res.get("per_case", []):
                vals.setdefault(rec["seed"], []).append(rec["Recall@K_correct"])
    return [float(np.mean(v)) for _, v in sorted(vals.items())]


GREEDY_CONFIGS = [("static", "Static", C_BASE), ("infer", "Greedy (inference)", C_INFER),
                  ("train", "Greedy (trained)", C_TRAIN)]


def greedy_3way():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, dirname, title in [(axes[0], "xg", "(a) Setting B (DAE only)"),
                               (axes[1], "xg_A", "(b) Setting A")]:
        data = [_greedy_seed_means(dirname, c) for c, _, _ in GREEDY_CONFIGS]
        _box(ax, data, [0, 1, 2], [c for _, _, c in GREEDY_CONFIGS])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([lab for _, lab, _ in GREEDY_CONFIGS], fontsize=8.5)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Recall@K (seed mean)")
    # (c) DAE の正解式数別（シード別平均の mean±SEM）
    ax = axes[2]
    for config, label, color in GREEDY_CONFIGS:
        acc = {}
        for f in sorted((EXP / "xg").glob(f"{config}__*.json")):
            doc = json.load(open(f))
            for mode_res in doc["results"].values():
                for rec in mode_res.get("per_case", []):
                    acc.setdefault(rec["n_correct"], {}).setdefault(rec["seed"], []).append(
                        rec["Recall@K_correct"])
        xs = sorted(x for x in acc if 1 <= x <= 10)
        means, sems = [], []
        for x in xs:
            sm = np.array([np.mean(v) for v in acc[x].values()])
            means.append(sm.mean())
            sems.append(sm.std(ddof=1) / np.sqrt(len(sm)))
        ax.errorbar(xs, means, yerr=sems, marker="o", ms=4, lw=1.6, capsize=3,
                    label=label, color=color)
    ax.set_xlabel("Number of equations in the true model (DAE)")
    ax.set_title("(c) By difficulty (Setting B)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    save(fig, "fig_greedy_3way")
```

- [ ] **Step 3: dof_stop() を追加**

```python
def _dof_seed_means(setting, key):
    vals = {}
    for f in sorted((EXP / "xd_dof").glob(f"{setting}__*.json")):
        doc = json.load(open(f))
        for mode_res in doc["results"].values():
            for rec in mode_res.get("per_case", []):
                vals.setdefault(rec["seed"], []).append(rec[key])
    return [float(np.mean(v)) for _, v in sorted(vals.items())]


def dof_stop():
    stats_doc = json.load(open(EXP / "dof_stop_stats.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    groups = [("dae", "Setting B (DAE)"), ("A", "Setting A")]
    for i, (setting, glabel) in enumerate(groups):
        oracle = _dof_seed_means(setting, "set_f1_oracleK")
        dof = _dof_seed_means(setting, "set_f1_dof")
        _box(ax, [oracle, dof], [i * 2.4 - 0.45, i * 2.4 + 0.45], [C_BASE, C_PROP])
    ax.set_xticks([0, 2.4])
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_ylabel("Set F1 (seed mean)")
    ax.set_title("(a) Oracle-K vs. DoF-stop")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=C_BASE, alpha=0.85, label="Oracle-K (K given)"),
               plt.Rectangle((0, 0), 1, 1, fc=C_PROP, alpha=0.85, label="DoF-stop (K unknown)")]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    series = [("set_f1_oracleK", "Set F1, oracle-K", C_BASE), ("set_f1_dof", "Set F1, DoF-stop", C_PROP),
              ("set_exact_dof", "Exact match, DoF-stop", C_TRAIN)]
    for key, label, color in series:
        acc = {}
        for f in sorted((EXP / "xd_dof").glob("dae__*.json")):
            doc = json.load(open(f))
            for mode_res in doc["results"].values():
                for rec in mode_res.get("per_case", []):
                    acc.setdefault(rec["n_correct"], {}).setdefault(rec["seed"], []).append(rec[key])
        xs = sorted(x for x in acc if 1 <= x <= 10)
        sm = [np.array([np.mean(v) for v in acc[x].values()]) for x in xs]
        ax.errorbar(xs, [m.mean() for m in sm], yerr=[m.std(ddof=1) / np.sqrt(len(m)) for m in sm],
                    marker="o", ms=4, lw=1.6, capsize=3, label=label, color=color)
    ax.set_xlabel("Number of equations in the true model (DAE)")
    ax.set_title("(b) By difficulty (Setting B)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    save(fig, "fig_dof_stop")
```

- [ ] **Step 4: `__main__` に3関数を追加して実行**

`__main__` ブロックを `dataset(); split_balance(); method_comparison(); characteristic(); mechanism(); greedy_3way(); dof_stop()` に更新。
Run: `python3 generate_figures_thesis.py && ls thesis/master_thesis/figures/ | wc -l`
Expected: 図7種 × 2形式 = 14ファイル（+.gitkeep）。エラーなし。

- [ ] **Step 5: 数値の整合確認**

fig_greedy_3way (a) の箱の中央値が greedy_3way_stats.json の overall（static 0.7237 / infer 0.7556 / train 0.7649）と同水準、fig_dof_stop (a) の DAE 箱が dof_stop_stats.json（oracle 0.7649 / DoF 0.7420）と同水準であることを目視確認。

- [ ] **Step 6: Commit**

```bash
git add generate_figures_thesis.py thesis/master_thesis/figures
git commit -m "feat(thesis): 論文用英語図その2（機構・greedy3者・DoF停止、箱ひげ）"
```

---

## アウトライン共通ルール（Task 6–12）

- 各章ファイルを以下の完成内容で**上書き**する。`\item \textbf{Pn.}` の太字文がトピックセンテンス（そのまま本文の段落先頭文になる英文）、サブ項目がサポート文の素材、`(wrap)` が段落の小結論。
- 数値の直前行の `% source:` コメントは**そのまま保持**する（本文化しても残す）。
- 参考文献が未登録の箇所は `% cite: <何の文献か>（Phase 2で実文献を検証して追加。捏造禁止）` というTeXコメントにする。本文に未定義の `\cite` を書かない（コンパイルエラー防止）。
- 各タスクの最後は共通2ステップ: (a) `cd thesis/master_thesis && latexmk main.tex` がexitコード0で通ること、(b) `git add thesis/master_thesis/<章>.tex && git commit -m "feat(thesis): 第N章アウトライン"`。

### Task 6: 第1章 Introduction アウトライン

**Files:** Modify: `thesis/master_thesis/Introduction.tex`（全置換）

- [ ] **Step 1: 以下の内容で Introduction.tex を上書き**

```latex
% ============================================================
% Chapter 1: Introduction — OUTLINE (Phase 1)
% ============================================================
\begin{itemize}
  \item \textbf{P1.} Physical models are essential for realizing digital twins and play an important role in process design and operation in the process industry.
  \begin{itemize}
    \item Purpose-built tools such as MATLAB/Simulink\cite{simulink} and Aspen Plus\cite{aspen} let engineers build models quickly, but only for the processes they support.
    \item For unsupported processes, engineers must survey a vast number of documents and refine a prototype by trial and error.
    \item (wrap) Physical model building therefore remains time-consuming and labor-intensive.
  \end{itemize}

  \item \textbf{P2.} To free engineers from this laborious work, our group aims to develop an automated physical model builder (AutoPMoB).
  \begin{itemize}
    \item AutoPMoB collects documents about a target process, extracts formulas, variables, and experimental data, and reorganizes them into a desired model.
    \item Fundamental technologies have been developed step by step, such as judging the equivalence of variable definitions across documents\cite{ProcessBERT}.
    \item (wrap) This study addresses the next fundamental step: assembling extracted equations into the model.
  \end{itemize}

  \item \textbf{P3.} Given a description of the desired model and its input and output variables, this study retrieves the set of equations that composes the model from an equation database built from multiple documents.
  \begin{itemize}
    \item A physical model is not a list of independent equations but a system of equations that share variables and must be solved simultaneously; the system is solvable when the number of equations equals the number of unknown variables (zero degrees of freedom).
    \item Example: the concentration dynamics of a continuous stirred-tank reactor (CSTR) requires three coupled equations — a mass-balance ordinary differential equation, a reaction-rate equation, and the Arrhenius equation.
    \item Text similarity alone cannot capture these inter-equation dependencies, especially when the equations come from different documents.
    \item (wrap) Retrieval must therefore evaluate candidate equations as a set, not one by one.
  \end{itemize}

  \item \textbf{P4.} We propose a two-stage retrieval method whose second stage evaluates each candidate against the already selected equations with set-aware features.
  \begin{itemize}
    \item Stage 1 narrows candidates with seven query--equation features such as text similarity and variable overlap.
    \item Stage 2 reranks candidates with a multilayer perceptron (MLP) using three additional set-aware features: complementarity, coherence, and domain agreement.
    \item We further introduce a learned greedy selection trained with teacher forcing, and a self-terminating criterion (DoF-stop) that stops retrieval when the selected set becomes a closed system.
    \item (wrap) The method returns a ranked, solvable equation set without knowing the answer size.
  \end{itemize}

  \item \textbf{P5.} To evaluate the method under realistic difficulty, we construct a large equation database and solvable test cases, and split them so that no document leaks between training and testing.
  \begin{itemize}
    % source: docs/progress_report_2026-06-14_seminar.tex §3（11,146式・361ソース・2,838件）
    \item The database holds 11,146 equations from 361 sources; 2,838 training cases include 1,000 differential--algebraic equation (DAE) cases generated to be solvable by construction.
    \item Sources are assigned wholly to train, validation, or test, and the split is stratified so that four difficulty-related distributions match between train and test.
    \item (wrap) Evaluation uses 10 random splits and reports the mean and spread.
  \end{itemize}

  \item \textbf{P6.} Experiments show that the proposed method retrieves equation sets far better than a classical information-retrieval baseline and than direct generation by a large language model (LLM), and analyses explain why.
  \begin{itemize}
    % source: experiments/strat_A.json / strat_B.json results.*.Recall@K_correct
    \item Recall@K improves from 0.521 to 0.743 in Setting A and from 0.466 to 0.689 in the hardest DAE-only setting (both +0.22, Wilcoxon signed-rank test).
    % source: llm_set_A_equiv_results.json / llm_set_B_equiv_results.json
    \item LLM direct generation reaches only 0.23--0.30 even with equivalence-based scoring.
    % source: experiments/feature_x_difficulty_stats.json overall_tests
    \item The structural set-aware features (complementarity and coherence) cause the improvement; the domain feature contributes nothing (p = 0.98).
    % source: experiments/greedy_3way_stats.json, experiments/dof_stop_stats.json
    \item Learned greedy selection recovers the saturation on the hardest cases, and DoF-stop returns closed systems more often than the oracle that knows the answer size.
    \item (wrap) These results establish set-aware retrieval as a practical component of AutoPMoB.
  \end{itemize}

  \item \textbf{P7.} The contributions of this thesis are threefold.
  \begin{itemize}
    \item A solvable-by-construction dataset: an 11,146-equation database, DAE case generation with zero degrees of freedom, and a stratified source-disjoint split.
    \item A set-aware two-stage retrieval method with learned greedy selection and self-terminating DoF-stop.
    \item An empirical account of the mechanism: which features cause the improvement, where it saturates, and how the intervention recovers it.
  \end{itemize}

  \item \textbf{P8.} The rest of this thesis is organized as follows. (Chapter roadmap: Related work → Method → Dataset → Experiments → Results and discussion → Conclusion.)
  % Fig. 1（AutoPMoB全体像と本研究の位置づけ）は Phase 2 で作成し、P2 の直後に挿入する。
\end{itemize}
```

- [ ] **Step 2: コンパイル確認 → Commit**（共通ルール参照。コミットメッセージ: `feat(thesis): 第1章アウトライン`）

### Task 7: 第2章 Related Work アウトライン

**Files:** Modify: `thesis/master_thesis/RelatedWork.tex`（全置換）

- [ ] **Step 1: 以下の内容で上書き**

```latex
% ============================================================
% Chapter 2: Related Work — OUTLINE (Phase 1)
% 各節末に必ず「本研究との相違」を置く（Kotoba Tech ルール）。
% ============================================================
\section{Component Technologies for Automated Physical Model Building}
\begin{itemize}
  \item \textbf{P1.} Automating physical model building requires a chain of component technologies, several of which have been developed in our group.
  \begin{itemize}
    \item Document collection, formula extraction, and variable-equivalence judgment (ProcessBERT\cite{ProcessBERT}) precede model assembly.
    % cite: 数式抽出・文書情報抽出の先行研究（Phase 2で実文献を検証して追加。捏造禁止）
    \item (difference) These works produce the equation database; this study addresses the downstream step of selecting a solvable equation set from it.
  \end{itemize}
\end{itemize}

\section{Information Retrieval and Reranking}
\begin{itemize}
  \item \textbf{P2.} Classical information retrieval ranks items independently by query--item relevance, and modern systems refine the top candidates with a second-stage reranker.
  \begin{itemize}
    % cite: TF-IDF/BM25 と retrieve-then-rerank の代表文献（Phase 2で検証して追加）
    \item Lexical methods (TF-IDF) and learned rerankers assume that item relevances are independent given the query.
    \item Diversification methods consider inter-item redundancy but not solvability.
    \item (difference) Our items are interdependent: a correct answer is a set that closes a system of equations, which motivates set-aware features.
  \end{itemize}
\end{itemize}

\section{Direct Generation of Domain Knowledge by Large Language Models}
\begin{itemize}
  \item \textbf{P3.} Large language models can generate domain equations directly from a description, which makes them a natural alternative to retrieval.
  \begin{itemize}
    % cite: LLMの科学知識生成・hallucination に関する代表文献（Phase 2で検証して追加）
    \item Generated equations may be physically valid yet differ in notation, and empirical equations with fitted coefficients are hard to reproduce verbatim.
    \item (difference) We use an LLM only as an external baseline (and for generating case descriptions); our method retrieves from a database, which keeps provenance and reproducibility.
  \end{itemize}
\end{itemize}

\section{Mathematical Information Retrieval}
\begin{itemize}
  \item \textbf{P4.} Mathematical information retrieval searches for single formulas similar to a query formula or text.
  \begin{itemize}
    % cite: math IR / formula search の代表文献（Phase 2で検証して追加）
    \item Existing systems rank individual formulas by structural or textual similarity.
    \item (difference) Our task returns a set of equations that jointly form a solvable model, a requirement absent from formula search.
  \end{itemize}
\end{itemize}
```

- [ ] **Step 2: コンパイル確認 → Commit**（`feat(thesis): 第2章アウトライン`）

### Task 8: 第3章 手法アウトライン

**Files:** Modify: `thesis/master_thesis/Method.tex`（全置換）

- [ ] **Step 1: 以下の内容で上書き**

```latex
% ============================================================
% Chapter 3: Set-aware Equation-Set Retrieval — OUTLINE (Phase 1)
% 特徴量定義の source: two_stage_query_conditioned.py compute_features (L212-236),
%                     set_aware_reranker.py docstring (gComp/gCoh/gDom) — 実装と一致させること
% ============================================================
\section{Problem Formulation}
\begin{itemize}
  \item \textbf{P1.} We formulate equation-set retrieval as ranking the equations in a database against a query that specifies the desired model.
  \begin{itemize}
    \item Query $q = (c, \mathcal{I}, \mathcal{O})$: a natural-language description $c$, input variables $\mathcal{I}$, and output variables $\mathcal{O}$.
    \item Database $\mathcal{E} = \{e_j\}$: each equation has its variable set $V(e_j)$, source document, and domain label.
    \item Ground truth: the equation set $\mathcal{R}_q \subset \mathcal{E}$ that composes the target model.
    \item Degrees of freedom of a selected set $S$: $\mathrm{DoF}(S) = |\mathrm{unknowns}(S)| - |S|$; the system is closed (solvable given the inputs) when $\mathrm{DoF}(S) = 0$.
    \item (wrap) The goal is to rank $\mathcal{R}_q$ at the top and, optionally, to return a closed set without knowing $|\mathcal{R}_q|$.
  \end{itemize}
\end{itemize}

\section{Stage 1: Candidate Retrieval with Query--Equation Features}
\begin{itemize}
  \item \textbf{P2.} The first stage scores each equation independently with seven query--equation features.
  \begin{itemize}
    % source: two_stage_query_conditioned.py compute_features L212-236（f0〜f6）
    \item (1) TF-IDF cosine similarity between the description and the equation text; (2) Jaccard similarity between the query I/O variables and the equation variables; (3) latent text similarity after singular-value decomposition (SVD); (4) input-variable coverage; (5) output-variable coverage; (6) fraction of the equation's variables that appear in the query I/O set; (7) binary domain match.
    \item The top-$k$ candidates by this score form the reference set for Stage 2.
    \item (wrap) Stage 1 is a strong classical-IR method on its own and serves as the baseline.
  \end{itemize}
\end{itemize}

\section{Stage 2: Set-aware Reranking}
\begin{itemize}
  \item \textbf{P3.} The second stage evaluates each candidate not in isolation but against a reference set, using three set-aware features.
  \begin{itemize}
    % source: set_aware_reranker.py docstring（gComp/gCoh/gDom の定義）
    \item Complementarity $g_{\mathrm{Comp}} = |V(e) \cap \mathcal{Q} \setminus \bigcup V(\mathrm{ref})| / |\mathcal{Q}|$: how many still-uncovered query variables the candidate supplies.
    \item Coherence $g_{\mathrm{Coh}} = |V(e) \cap \bigcup V(\mathrm{ref})| / |V(e)|$: how strongly the candidate shares variables with the reference set.
    \item Domain agreement $g_{\mathrm{Dom}}$: the fraction of the reference set that shares the candidate's domain.
    \item A multilayer perceptron (one hidden layer of 64 units, ReLU, dropout 0.1) scores the 10 features (7 + 3) and is trained with a pairwise ranking loss (margin 0.1).
    % source: experiments/strat_A.json config（hidden_dim 64, margin 0.1, loss pairwise）
    \item (wrap) We call this configuration reranker-10S.
  \end{itemize}
\end{itemize}

\section{Sequential Selection Strategies}
\begin{itemize}
  \item \textbf{P4.} Because set-aware features depend on what has already been selected, we compare three selection strategies.
  \begin{itemize}
    \item Static: set features are computed once against the fixed Stage-1 top-$k$ reference.
    \item Greedy at inference: after each selection, set features are recomputed against the already selected set; the model itself is trained statically.
    \item Learned greedy: the model is also \emph{trained} on sequential contexts with teacher forcing — during training the true partial set is given as the reference, so training matches inference.
    % source: set_aware_reranker.py --train-greedy, --greedy-train-cap 8
    \item (wrap) The comparison isolates the effect of matching the training condition to the inference condition.
  \end{itemize}
\end{itemize}

\section{Self-terminating Retrieval with Degrees of Freedom}
\begin{itemize}
  \item \textbf{P5.} Retrieval should stop by itself when the selected equations form a solvable system, rather than requiring the answer size $K$.
  \begin{itemize}
    \item DoF-stop keeps selecting greedily and stops when $\mathrm{DoF}(S) = 0$, i.e., when every non-input variable is determined by some equation.
    % source: set_aware_reranker.py --stop-dof（greedy_close）
    \item The returned set is closed by construction, which is exactly what a downstream simulator needs.
    \item (wrap) DoF-stop turns the ranker into a self-contained model builder that needs no oracle knowledge.
  \end{itemize}
\end{itemize}
% 手法パイプライン図（2段階＋greedy＋DoF停止）は Phase 2 で作成し §3.2 冒頭に挿入する。
```

- [ ] **Step 2: コンパイル確認 → Commit**（`feat(thesis): 第3章アウトライン`）

### Task 9: 第4章 データセットアウトライン

**Files:** Modify: `thesis/master_thesis/Dataset.tex`（全置換）

- [ ] **Step 1: 以下の内容で上書き**

```latex
% ============================================================
% Chapter 4: Equation Database and Test Case Construction — OUTLINE (Phase 1)
% ============================================================
\section{Equation Database Construction}
\begin{itemize}
  \item \textbf{P1.} We built an equation database of 11,146 equations from 361 sources across 32 engineering domains.
  \begin{itemize}
    % source: docs/progress_report_2026-06-14_seminar.tex §3
    \item A large language model (Gemini 2.5 Pro) read 329 paper PDFs and extracted the equations in the text.
    \item Basic equations of 32 domains (670 equations) were generated by the same model.
    \item Notation was normalized so that shared variables connect equations across sources.
    \item (wrap) The database grew from 3,244 to 11,146 equations during this work.
  \end{itemize}
\end{itemize}

\section{Test Case Construction}
\begin{itemize}
  \item \textbf{P2.} Each case pairs a query (description and I/O variables) with its ground-truth equation set; the 2,838 cases fall into four groups.
  \begin{itemize}
    % source: training_cases.json variant_type 集計（92+33+198+500+1000+拡張1,015）
    \item Original cases (92) come directly from single papers.
    \item Multi-source cases (731) are assembled rule-based from equations that share variables across documents — no LLM is used.
    \item DAE cases (1,000) are generated to be solvable by construction (Section 4.3).
    \item Augmentation cases (1,015: random I/O swap and paraphrase) are used only for training and excluded from the main evaluation.
    \item (wrap) 74.6\% of the cases need two or more equations, where set selection matters. (Figure: fig\_dataset)
  \end{itemize}
\end{itemize}

\section{Solvable DAE Case Generation}
\begin{itemize}
  \item \textbf{P3.} Earlier test cases were biased toward few and algebraic equations, so we regenerated hard cases as differential--algebraic systems that are solvable by construction.
  \begin{itemize}
    \item Procedure (proposed by Mr.\ Kato): start from an ordinary differential equation, take its differentiated variable as an output, repeatedly add equations that share non-output variables, and stop at the target size $X$; accept only square systems in which the number of unknowns equals the number of equations (zero degrees of freedom).
    \item $X = 1, \dots, 10$ with 100 cases each (1,000 in total); only the case description is written by an LLM (Anthropic Claude), everything else is rule-based.
    \item CSTR example: mass-balance ODE + reaction-rate equation $r_A = k C_A$ + Arrhenius equation $k = A \exp(-E/RT)$.
    \item (wrap) The construction guarantees that a correct answer exists and is solvable.
  \end{itemize}
\end{itemize}

\section{Stratified Source-disjoint Splitting}
\begin{itemize}
  \item \textbf{P4.} Sources are assigned wholly to train, validation, or test, and the assignment is chosen so that difficulty-related distributions match across splits.
  \begin{itemize}
    \item Source-disjoint assignment (roughly 6:2:2) prevents answer leakage through paraphrases of the same document.
    \item Among 400 random candidate assignments, we select the one minimizing the summed $L_1$ distance of four distributions: number of equations, input variables, output variables, and spanned sources.
    % source: docs/progress_report_2026-06-14_seminar.tex §4（L1 0.27→0.07 等。Phase 2 で一次ソースから再導出する）
    \item Evaluation repeats over 10 random seeds and reports mean and spread. (Figure: fig\_split\_balance)
    \item (wrap) Results therefore reflect generalization to unseen documents, not memorization.
  \end{itemize}
\end{itemize}

\begin{figure}[h]\centering
\includegraphics[width=0.95\linewidth]{figures/fig_dataset.pdf}
\caption{Composition of the 2,838 cases and the distribution of the number of correct equations. Setting A (1,823 cases) excludes augmentation.}
\label{fig:dataset}
\end{figure}

\begin{figure}[h]\centering
\includegraphics[width=0.95\linewidth]{figures/fig_split_balance.pdf}
\caption{Stratified source-disjoint split: four difficulty-related distributions match between train (blue) and test (orange); one seed shown.}
\label{fig:split}
\end{figure}
```

- [ ] **Step 2: コンパイル確認 → Commit**（`feat(thesis): 第4章アウトライン`）

### Task 10: 第5章 Experiments アウトライン

**Files:** Modify: `thesis/master_thesis/Experiment.tex`（全置換）

- [ ] **Step 1: 以下の内容で上書き**

```latex
% ============================================================
% Chapter 5: Experiments — OUTLINE (Phase 1)
% ============================================================
\section{Evaluation Settings}
\begin{itemize}
  \item \textbf{P1.} We evaluate on two settings that exclude training-only augmentation, because augmentation inflates the baseline and hides the true difficulty.
  \begin{itemize}
    \item Setting A: original + multi-source + DAE cases (1,823). Setting B: DAE only (1,000), the hardest setting.
    % source: experiments/strat_full.json vs strat_A.json baseline Recall@K_correct（0.6218 vs 0.5206）
    \item With augmentation included, the baseline rises from 0.521 to 0.622 (+0.101), shrinking the apparent gap of the proposed method from +0.222 to +0.156.
    \item (wrap) All results use the stratified source-disjoint split with 10 random seeds.
  \end{itemize}
\end{itemize}

\section{Compared Methods}
\begin{itemize}
  \item \textbf{P2.} We compare the proposed method against one internal and one external baseline.
  \begin{itemize}
    \item Baseline (classical IR): Stage-1 ranking with the seven query--equation features.
    \item Proposed (reranker-10S): Stage-2 set-aware reranking; greedy and DoF-stop variants are examined in Sections 6.4--6.5.
    \item LLM direct generation (external): the LLM generates the needed equations from the query alone; generated equations are matched to the database with LLM-based equivalence judgment and then ranked.
    % source: llm_set_*_equiv_results.json（n=48〜50、等価判定モデル claude-opus-4-8）
    \item (wrap) The LLM comparison uses a uniform sample of about 50 cases per setting because equivalence judgment is expensive.
  \end{itemize}
\end{itemize}

\section{Evaluation Metrics}
\begin{itemize}
  \item \textbf{P3.} The primary metric is Recall@$K$ with the window matched to the answer size $K$ of each case.
  \begin{itemize}
    \item Recall@$K$ measures whether the needed equations are placed in the top $K$, no more and no less; a fixed window (e.g., Recall@3) over- or under-rewards cases whose $K$ differs from 3.
    \item Secondary metrics: MAP (mean average precision) and Recall@20 (screening with human review in mind).
    \item For self-terminating retrieval: exact-match rate of the returned set, set F1, and closure rate (fraction of returned sets with zero degrees of freedom).
    \item (wrap) Together the metrics cover ranking quality, screening utility, and solvability.
  \end{itemize}
\end{itemize}

\section{Implementation and Reproducibility}
\begin{itemize}
  \item \textbf{P4.} All experiments run as deterministic scripts against pinned application programming interface (API) models, so that a third party can reproduce them.
  \begin{itemize}
    % source: docs/progress_report_2026-06-14_seminar.tex §6（claude-sonnet-4-5-20250929）
    \item LLM-dependent steps (DAE descriptions, LLM baseline) call date-pinned API models; the proposed method itself uses no LLM.
    \item Hyperparameters are listed in Appendix A.
    \item (wrap) Scripts, data, and random seeds are versioned in the repository.
  \end{itemize}
\end{itemize}

\section{Statistical Testing}
\begin{itemize}
  \item \textbf{P5.} Paired differences over the 10 seeds are tested with the Wilcoxon signed-rank test (with the paired $t$-test as reference).
  \begin{itemize}
    % source: experiments/significance_stats.json（Task 2）
    \item Baseline vs.\ proposed in Settings A and B; static vs.\ greedy variants in Section 6.4.
    \item We report effect size (Cohen's $d_z$) together with $p$-values; "significant" is used only after these tests.
    \item (wrap) With $n = 10$ pairs the smallest two-sided Wilcoxon $p$ is 0.00195.
  \end{itemize}
\end{itemize}
```

- [ ] **Step 2: コンパイル確認 → Commit**（`feat(thesis): 第5章アウトライン`）

### Task 11: 第6章 Results and Discussion アウトライン

**Files:** Modify: `thesis/master_thesis/ResultsAndDiscussion.tex`（全置換）

**Interfaces:**
- Consumes: `experiments/greedy_3way_A_stats.json`（Task 3完了が前提）と `experiments/significance_stats.json`（Task 2）。**下記テンプレの `⟨A-static⟩⟨A-infer⟩⟨A-train⟩⟨A-p⟩` は、書き込む前に次のコマンドで実値を取得して置き換える**（プレースホルダを残したままコミットしない）:
```bash
python3 -c "import json; d=json.load(open('experiments/greedy_3way_A_stats.json')); o=d['overall']; print('static',o['static']['mean'],'infer',o['infer']['mean'],'train',o['train']['mean']); print('train_vs_infer p_wilcoxon', d['tests']['train_vs_infer']['p_wilcoxon'])"
```

- [ ] **Step 1: 実値を取得し、以下の内容（⟨…⟩を実値に置換済み）で上書き**

```latex
% ============================================================
% Chapter 6: Results and Discussion — OUTLINE (Phase 1)
% 弧: 性能 → 機構 → 介入 → 自己停止 → 留意点
% ============================================================
\section{Overall Comparison}
\begin{itemize}
  \item \textbf{P1.} The proposed method improves Recall@K by +0.22 over the classical-IR baseline in both settings and clearly outperforms LLM direct generation.
  \begin{itemize}
    % source: experiments/strat_A.json / strat_B.json results.*.Recall@K_correct
    \item Setting A: 0.521 → 0.743 (+0.222); Setting B (DAE only): 0.466 → 0.689 (+0.223).
    % source: experiments/significance_stats.json A/B Recall@K_correct p_wilcoxon
    \item Both improvements are significant (Wilcoxon signed-rank, $p = 0.00195$, all 10 seeds improved).
    % source: experiments/strat_A.json MAP/Recall@20
    \item MAP rises from 0.581 to 0.863 (A) and 0.512 to 0.832 (B); Recall@20 reaches 0.909 (A) and 0.838 (B).
    \item (wrap) The gain is stable across seeds and metrics. (Figure: fig\_method\_comparison, Table 1)
  \end{itemize}
  \item \textbf{P2.} LLM direct generation can produce equivalent equations but cannot rank them into the top $K$.
  \begin{itemize}
    % source: llm_set_A_equiv_results.json / llm_set_B_equiv_results.json / llm_set_full_equiv_results.json
    \item Recall@K stays at 0.23--0.30 while rank-free coverage reaches 0.34--0.57; the gap means "can generate, cannot arrange."
    \item Database answers often contain source-specific empirical equations with fitted coefficients that are hard to generate verbatim.
    \item (wrap) Neither classical IR nor an LLM alone suffices; set-aware reranking over a database is needed.
  \end{itemize}
\end{itemize}

\section{Performance by Case Characteristics}
\begin{itemize}
  \item \textbf{P3.} The harder the case, the larger the advantage of the proposed method.
  \begin{itemize}
    % source: experiments/strat_A.json per_case（層別は fig_characteristic と同じバケット）
    \item By number of equations: the baseline falls from 0.85 (single equation) to about 0.28 (ten equations), while the proposed method falls only to about 0.54; the gap widens from +0.03 to +0.21--0.29.
    \item By number of input variables: the gap widens from +0.05 ($\le$5) to +0.28 ($\ge$21).
    \item By number of sources: the effect is roughly constant (+0.22--0.26) because source count is confounded with equation count.
    \item (wrap) The method leaves easy cases intact and helps exactly where selection is combinatorial. (Figure: fig\_characteristic)
  \end{itemize}
\end{itemize}

\section{Mechanism Analysis: Which Features Cause the Improvement}
\begin{itemize}
  \item \textbf{P4.} Feature-by-difficulty ablation shows that the structural set-aware features cause the improvement, while the domain feature contributes nothing.
  \begin{itemize}
    % source: experiments/feature_x_difficulty_stats.json overall_tests
    \item Adding complementarity alone: +0.052 ($p < 0.001$); coherence alone: +0.050 ($p < 0.001$); domain alone: $-0.0001$ ($p = 0.98$).
    \item All three together add +0.050 over the 7-feature reranker (0.693 → 0.743), so the pure set-feature contribution is about +0.05 of the total +0.22; the rest comes from the MLP learning the 7 base features.
    % source: experiments/feature_x_difficulty_stats.json lift_by_n_correct / lift_by_n_input
    \item The lift has a threshold-and-hump shape: none for single-equation cases ($-0.01$ at $X=1$), peak at $X=3$ (+0.12), and saturation at the hardest end (input variables $\ge 16$).
    \item (wrap) The saturation at the hardest end predicts that a sequential method should help there — tested next. (Figure: fig\_mechanism)
  \end{itemize}
\end{itemize}

\section{Intervention: Learned Greedy Selection}
\begin{itemize}
  \item \textbf{P5.} Making both inference and training sequential recovers the saturation, confirming the mechanism causally.
  \begin{itemize}
    % source: experiments/greedy_3way_stats.json overall
    \item Setting B (DAE): static 0.724 $<$ greedy at inference 0.756 $<$ learned greedy 0.765 (10 seeds).
    % source: experiments/greedy_3way_stats.json tests.train_vs_infer
    \item Learned greedy beats inference-only greedy by +0.009 (Wilcoxon $p = 0.037$) — training must match inference.
    % source: experiments/greedy_3way_stats.json tests_hard_X8plus.train_vs_static
    \item The gain concentrates at the predicted saturation: at $X \ge 8$, learned greedy beats static by +0.056 ($p < 0.001$).
    % source: experiments/greedy_3way_A_stats.json（Task 3。⟨…⟩は実値に置換すること）
    \item Setting A: static ⟨A-static⟩ $<$ inference greedy ⟨A-infer⟩ $<$ learned greedy ⟨A-train⟩ (train vs.\ infer Wilcoxon $p =$ ⟨A-p⟩), showing the effect generalizes beyond DAE.
    \item (wrap) Predict saturation → intervene → recover: the arc establishes causality, not mere correlation. (Figure: fig\_greedy\_3way)
  \end{itemize}
\end{itemize}

\section{Self-terminating Retrieval}
\begin{itemize}
  \item \textbf{P6.} DoF-stop returns closed, solvable equation sets without knowing the answer size, at almost no cost in accuracy.
  \begin{itemize}
    % source: experiments/dof_stop_stats.json set_exact_*
    \item Exact-match rate is identical to the oracle that knows $K$: 0.368 (DAE) and 0.526 (Setting A).
    % source: experiments/dof_stop_stats.json closed_*
    \item Closure rate exceeds the oracle: 0.883 vs.\ 0.781 (DAE) and 0.936 vs.\ 0.865 (A) — DoF-stop returns solvable systems by design.
    % source: experiments/dof_stop_stats.json dof_vs_oracle_f1
    \item The set-F1 cost is small: $-0.008$ (A, $p = 0.006$) and $-0.023$ (DAE).
    \item Caveat, stated honestly: in this dataset $K$ nearly equals the number of output variables, so predicting $K$ itself is easy; the claim is self-termination with a closure guarantee, not $K$ prediction.
    \item (wrap) The ranker becomes a self-contained builder: retrieve, close the system, stop. (Figure: fig\_dof\_stop)
  \end{itemize}
\end{itemize}

\section{Limitations}
\begin{itemize}
  \item \textbf{P7.} Four limitations qualify the results.
  \begin{itemize}
    \item DAE case descriptions are LLM-generated and contain stylistic variation.
    \item Source count is confounded with equation count, so its effect is not causal.
    \item The stratified split is one scheme chosen from 400 candidates; 10 seeds mitigate but do not remove this choice.
    \item Evaluation is database-internal: retrieval cannot find equations absent from the database, and the LLM comparison covers about 50 cases per setting.
  \end{itemize}
\end{itemize}

\begin{figure}[h]\centering
\includegraphics[width=0.95\linewidth]{figures/fig_method_comparison.pdf}
\caption{Recall@K of the three methods in the three settings. Boxes show 10 seeds (baseline and proposed); diamonds show the single-run LLM direct generation (n$\approx$50, equivalence-judged), with dashed lines for its rank-free coverage.}
\label{fig:comparison}
\end{figure}

\begin{figure}[h]\centering
\includegraphics[width=0.95\linewidth]{figures/fig_characteristic.pdf}
\caption{Recall@K stratified by case characteristics (Setting A): (a) number of equations, (b) number of sources, (c) number of input variables.}
\label{fig:characteristic}
\end{figure}

\begin{figure}[h]\centering
\includegraphics[width=0.95\linewidth]{figures/fig_mechanism.pdf}
\caption{Recall@K lift of each set-aware feature over the 7-feature reranker, stratified by difficulty (mean $\pm$ SEM over 10 seeds). Structural features (complementarity, coherence) carry the lift; the domain feature stays at zero.}
\label{fig:mechanism}
\end{figure}

\begin{figure}[h]\centering
\includegraphics[width=0.95\linewidth]{figures/fig_greedy_3way.pdf}
\caption{Three selection strategies (static, greedy at inference, learned greedy): (a) Setting B and (b) Setting A over 10 seeds; (c) by number of equations in Setting B.}
\label{fig:greedy}
\end{figure}

\begin{figure}[h]\centering
\includegraphics[width=0.95\linewidth]{figures/fig_dof_stop.pdf}
\caption{Self-terminating retrieval: (a) set F1 of oracle-K vs.\ DoF-stop (10 seeds); (b) set quality by number of equations in Setting B.}
\label{fig:dof}
\end{figure}
```

- [ ] **Step 2: 結果表（Table 1）を追加**

§6.1 の直後に、`experiments/strat_A.json` / `strat_B.json` の実値で全体結果表を置く:
```latex
\begin{table}[h]\centering
\caption{Recall@K, MAP, and Recall@20 over 10 stratified splits (mean $\pm$ SD for Recall@K).}
\label{tab:overall}
\begin{tabular}{llccc}
\toprule
Setting & Method & Recall@K & MAP & Recall@20 \\
\midrule
% source: experiments/strat_A.json results.baseline / results.reranker-10S
\multirow{2}{*}{A (1,823 cases)}
 & Baseline & $0.521 \pm 0.045$ & 0.581 & 0.836 \\
 & \textbf{Proposed (reranker-10S)} & $\mathbf{0.743 \pm 0.054}$ & \textbf{0.863} & \textbf{0.909} \\
\midrule
% source: experiments/strat_B.json results.baseline / results.reranker-10S
\multirow{2}{*}{B (DAE only, 1,000 cases)}
 & Baseline & $0.466 \pm 0.037$ & 0.512 & 0.721 \\
 & \textbf{Proposed (reranker-10S)} & $\mathbf{0.689 \pm 0.033}$ & \textbf{0.832} & \textbf{0.838} \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 3: コンパイル確認（図が全て解決されること）→ Commit**（`feat(thesis): 第6章アウトライン（図表込み）`）

### Task 12: 第7章 Conclusion・Appendix アウトライン

**Files:** Modify: `thesis/master_thesis/Conclusion.tex`, `thesis/master_thesis/Appendix.tex`（全置換）

- [ ] **Step 1: Conclusion.tex を上書き**

```latex
% ============================================================
% Chapter 7: Conclusion — OUTLINE (Phase 1)
% ============================================================
\begin{itemize}
  \item \textbf{P1.} This thesis proposed and validated set-aware retrieval of equation sets for automated physical model building.
  \begin{itemize}
    \item Contributions: solvable-by-construction dataset (11,146 equations, DAE cases, stratified split); set-aware two-stage retrieval (+0.22 Recall@K over classical IR, superior to LLM direct generation); mechanism analysis with causal confirmation via learned greedy; self-terminating DoF-stop with a closure guarantee.
  \end{itemize}
  \item \textbf{P2.} Future work follows three directions.
  \begin{itemize}
    \item Scheduled sampling to close the remaining gap between teacher forcing and inference-time behavior.
    \item Validation on real modeling tasks beyond the database, including equations absent from it.
    \item Integration into the AutoPMoB pipeline together with formula extraction and variable-equivalence judgment.
  \end{itemize}
\end{itemize}
```

- [ ] **Step 2: Appendix.tex を上書き**

```latex
% ============================================================
% Appendix — OUTLINE (Phase 1)
% ============================================================
\chapter{Hyperparameters and Reproducibility}
% source: experiments/strat_A.json config / set_aware_reranker.py argparse defaults
\begin{table}[h]\centering
\caption{Hyperparameters of the reranker (identical across settings unless noted).}
\begin{tabular}{ll}
\toprule
Item & Value \\
\midrule
MLP hidden units / dropout & 64 / 0.1 \\
Loss / margin & pairwise ranking / 0.1 \\
Epochs / learning rate / batch size & 15 / 0.001 / 16 \\
Negative samples / weight decay & 8 / 0.0001 \\
Stage-1 top-$k$ & 50 (Settings A, B); 200 (greedy and DoF-stop runs) \\
Greedy training cap & 8 \\
Random seeds & 42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999 \\
\bottomrule
\end{tabular}
\end{table}

\chapter{Example of a Generated DAE Case}
\begin{itemize}
  \item One DAE case ($X=3$, CSTR) with its description, I/O variables, and ground-truth equations.
  % Phase 2: dae_cases.json から実例を1件転記する（改変しない）。
\end{itemize}

\chapter{LLM Prompts}
\begin{itemize}
  \item Prompts used for DAE description generation and for the LLM direct-generation baseline, with pinned model identifiers.
  % Phase 2: generate_dae_cases.py / evaluate_llm_dae.py から実プロンプトを転記する（改変しない）。
\end{itemize}
```

- [ ] **Step 3: コンパイル確認 → Commit**（`feat(thesis): 第7章・Appendixアウトライン`）

---

### Task 13: 全体検証とアウトラインPDFの確定

**Files:** Modify: `docs/development_log.tex`（追記）

- [ ] **Step 1: フルコンパイルと通し確認**

Run: `cd thesis/master_thesis && latexmk -gg main.tex && pdftotext main.pdf - | head -100`
Expected: exitコード0。目次に7章＋Appendix A–C。未解決参照（`??`）と未定義引用（`[?]`）がゼロであること: `grep -c "??" <(pdftotext main.pdf -)` → 0。

- [ ] **Step 2: スペック照合チェックリスト**

`docs/superpowers/specs/2026-07-12-master-thesis-design.md` に対して:
- [ ] 成功基準1: 過去修論と同形式（英語本文・日本語表紙・jsbook）→ 表紙とAbstractで確認
- [ ] 成功基準2: 各アウトライン項目のトピックセンテンスだけ読んで要約になっている → 全章のP系列を通読して確認
- [ ] 成功基準3: 本文の全数値に `% source:` コメントがある → `grep -B1 "0\.[0-9][0-9]" thesis/master_thesis/*.tex | grep -c "source:"` で網羅を点検
- [ ] 成功基準5: 骨子PDFが生成できる → Step 1 で確認済み
- [ ] スペック§7: 追加実験1（significance_stats.json）と2（greedy_3way_A_stats.json）が存在しコミット済み
- [ ] Global Constraints: 「X≥8 +0.056 は train vs static」の帰属が第6章 §6.4 で正しいこと

- [ ] **Step 3: development_log.tex に追記**

内容: 修論Phase 1完了（TeX骨格・検定・設定Aのgreedy3者・英語図7点・全章アウトライン）。

- [ ] **Step 4: 最終 Commit**

```bash
git add thesis/master_thesis docs/development_log.tex
git commit -m "feat(thesis): Phase 1完了 — 全章アウトラインPDF・追加実験・英語図"
```

- [ ] **Step 5: ユーザーへの報告**

main.pdf の場所を伝え、先生方への骨子レビュー依頼（ゼミ提出）を提案して終了。Phase 2（Intro本文化＋Fig.1＋Method/Dataset章）は骨子レビューの結果を反映して別計画を作る。

---

## 実行順序の注意

- Task 3 Step 3（30ジョブ実行）は時間がかかるため、起動後に Task 4 を並行して進めてよい。ただし **Task 5 Step 0 と Task 11 の前に Task 3 完了が必須**。
- Task 1 と Task 2 は独立。Task 6–12 は Task 1（章ファイル存在）と Task 4–5（図ファイル存在。図を参照するのは Task 9・11 のみ）に依存する。
- 判断に迷う結果（例: 設定Aで train < infer になった等）が出たら、値を偽らずそのまま記録し、ユーザーに報告して指示を仰ぐ。
