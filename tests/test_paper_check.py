"""paper/cace/check_manuscript.py（投稿論文の原稿の点検）の単体テスト."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "cace"))
import check_manuscript as cm  # noqa: E402


# ---- 手で打った数値 --------------------------------------------------------------------------

def test_hand_typed_numbers_finds_decimals_and_counts():
    tex = r"Recall@20 rises from 0.906 to \RecAProp{} on 1{,}731 cases and 11,146 equations."
    assert cm.hand_typed_numbers(tex) == ["0.906", "1{,}731", "11,146"]


def test_hand_typed_numbers_accepts_macros_and_small_integers():
    tex = r"Recall@20 rises from \RecABase{} to \RecAProp{} in all 10 splits, with 20 equations presented in 2006."
    assert cm.hand_typed_numbers(tex) == []


def test_hand_typed_numbers_ignores_comments_definitions_lengths_and_model_names():
    tex = "\n".join([
        r"% source: 0.981 in main.json",
        r"\newcommand{\RecAProp}{0.981}",
        r"\includegraphics[width=0.95\linewidth]{figures/fig.pdf}",
        r"\begin{tabular}{p{0.44\linewidth}c}\setlength{\tabcolsep}{2.5pt}",
        r"Claude Opus~5.5, Claude Haiku 5.5, Gemini 2.5 Pro, Claude Opus 4.8, and Claude Sonnet~4.5",
        "A model name may break across lines in the source: Claude Sonnet\n4.5 wrote the descriptions.",
        r"a language model, Claude Fable~5.2, judged the cases",
        r"The share is 50\% (an escaped percent sign starts no comment), and 0.5 follows.",
    ])
    assert cm.hand_typed_numbers(tex) == ["0.5"]


def test_hand_typed_numbers_ignores_verbatim_blocks():
    tex = "\n".join([r"\begin{Verbatim}[fontsize=\scriptsize]", "Use k1 = 1.5 and 11,146 equations.",
                     r"\end{Verbatim}", "After 0.75."])
    assert cm.hand_typed_numbers(tex) == ["0.75"]


def test_hand_typed_numbers_finds_integers_that_depend_on_the_data():
    # 査読の指摘（2026-11-16）：区切りのない整数（36 cases、1731、90%）を見逃していた
    tex = r"Opus was higher in 36 cases, 28 of the 36, on 1731 cases and 5019 calls, 90\% of them, with seed 42."
    assert cm.hand_typed_numbers(tex) == ["36", "28", "36", "1731", "5019", "90", "42"]


def test_hand_typed_numbers_accepts_the_constants_of_the_design():
    tex = (r"in all 10 splits, the top 20, Recall@20, 100 cases for each $X$ from 1 to 10, $N_1 = 500$, "
           r"20\% and 60\%, 512 tokens, in 2006, \citep{Gemini25}, \ref{tab:n1}, \texttt{dae\_X3\_016}, "
           r"\texttt{intfloat/e5-base-v2}, NTCIR-12, \includegraphics[width=\linewidth]{figures/fig_greedy_3way.pdf}")
    assert cm.hand_typed_numbers(tex) == []


# ---- マクロの展開 ----------------------------------------------------------------------------

def test_read_macros_parses_newcommand_lines():
    text = "% generated\n\\newcommand{\\NCases}{1{,}731}\n\\newcommand{\\StagePairs}{$(0.7, 0.3)$, and $(0.0, 1.0)$}\n"
    assert cm.read_macros(text) == {"NCases": "1{,}731", "StagePairs": "$(0.7, 0.3)$, and $(0.0, 1.0)$"}


def test_expand_replaces_number_macros_only():
    macros = {"RecAProp": "0.981", "NPdf": "329"}
    tex = r"\NPdf{} documents reach \RecAProp, as \citet{CombineEq} showed; \NPdf\ files."
    assert cm.expand(tex, macros) == r"329 documents reach 0.981, as \citet{CombineEq} showed; 329 files."


def test_swallowed_spaces_finds_a_macro_followed_by_a_space():
    # TeX は制御語の後ろの空白を読み飛ばすので、\NPdf files は「329files」と印刷される
    macros = {"NPdf": "329"}
    assert cm.swallowed_spaces(r"The \NPdf files and the \NPdf{} files, \NPdf.", macros) == [r"\NPdf files"]


def test_swallowed_spaces_finds_a_macro_at_the_end_of_a_line():
    # 行末も空白と同じで、制御語の直後なら読み飛ばされる
    macros = {"NPdf": "329"}
    assert cm.swallowed_spaces("The \\NPdf\nfiles and \\NPdf{}\nfiles.", macros) == ["\\NPdf\nfiles"]


# ---- 語数 ------------------------------------------------------------------------------------

def test_word_count_counts_prose_words():
    assert cm.word_count("The method ranks the whole database.") == 6


def test_word_count_strips_commands_comments_math_and_floats():
    tex = "\n".join([
        r"\section{Results}\label{sec:results}",                                                    # 1
        r"Static reranking beat the baseline \citep{Manning} in all 10 splits. % a note of many words",  # 9
        r"The score $s_1(q, e_j)$ mixes two terms (Eq.~\eqref{eq:stage1}).",                        # 6
        r"\begin{table}[t]\caption{A caption is no prose.}\begin{tabular}{ll}a & b \\ \end{tabular}\end{table}",
        r"\begin{equation} s_1 = w_t \cos(c, t_j) \end{equation}",
        r"\todo{A note for the authors.}",
    ])
    assert cm.word_count(tex) == 16


def test_word_count_keeps_the_text_of_formatting_commands():
    assert cm.word_count(r"\paragraph{Proposed method} We call it \emph{set-aware} reranking, in 50\% of cases.") == 11


def test_word_count_treats_an_escaped_dollar_as_text():
    assert cm.word_count(r"It cost \$5 per query and \$7 per day in total.") == 11


# ---- 書かない言い回し ------------------------------------------------------------------------

def test_banned_phrases_finds_each_banned_wording():
    text = ("The 329 documents are split so that no document leaks; accuracy on other notations is untested. "
            "The method guarantees a solvable set and removes duplicated equations.")
    assert cm.banned_phrases(text) == ["329 documents", "no document leaks", "untested", "guarantee", "duplicate"]


def test_banned_phrases_accepts_the_agreed_wording():
    text = ("The 329 PDF files are split so that no source spans training and testing. The experiments cover only "
            "queries that share the notation of the ground truth, and the rule does not ensure full coverage.")
    assert cm.banned_phrases(text) == []


def test_banned_phrases_also_covers_what_the_paper_must_not_mention():
    text = "This thesis renamed the symbols of augmented cases and found a significant gain under a document-level split."
    assert cm.banned_phrases(text) == ["augmented cases", "thesis", "renamed symbols", "document-level", "significant"]


def test_banned_phrases_allows_augmented_in_other_senses():
    assert cm.banned_phrases("Retrieval-augmented generation is related work.") == []
    assert cm.banned_phrases("The augmented cases were dropped.") == ["augmented cases"]


# ---- 図の写し --------------------------------------------------------------------------------

def test_stale_figures_reports_copies_that_differ_from_the_thesis(tmp_path):
    thesis, paper = tmp_path / "thesis", tmp_path / "paper"
    thesis.mkdir()
    paper.mkdir()
    (thesis / "a.pdf").write_bytes(b"new")
    (paper / "a.pdf").write_bytes(b"old")
    (thesis / "b.pdf").write_bytes(b"same")
    (paper / "b.pdf").write_bytes(b"same")
    (thesis / "c.pdf").write_bytes(b"only in the thesis")
    assert cm.stale_figures(paper, thesis, ["a.pdf", "b.pdf", "c.pdf"]) == ["a.pdf", "c.pdf"]


# ---- Highlights ------------------------------------------------------------------------------

def test_long_highlights_reports_items_over_the_limit():
    tex = "\\item " + "a" * 85 + "\n\\item " + "b" * 86 + "\n"
    assert cm.long_highlights(tex, limit=85) == ["b" * 86]


# ---- 原稿そのもの ----------------------------------------------------------------------------

def test_manuscript_passes_the_checks():
    # 図の写しの照合は外す：tests/test_fig_overview.py が実行中に修論の図を書き直すので、テストの順で結果が変わる。
    # 図の照合そのものは下のテストと、python3 paper/cace/check_manuscript.py の実行で確かめる。
    assert cm.check(figures=False) == []


def test_check_reports_figures_that_differ_from_the_thesis(monkeypatch, tmp_path):
    (tmp_path / "fig_pfi.pdf").write_bytes(b"a newer figure in the thesis")
    monkeypatch.setattr(cm, "THESIS_FIGURES", tmp_path)
    failures = cm.check()
    assert len(failures) == 1 and "fig_pfi.pdf" in failures[0]


def test_appendix_prints_the_prompt_that_the_llm_comparison_used():
    rl = pytest.importorskip("run_llm_db_baseline")
    appendix = (ROOT / "paper" / "cace" / "appendix.tex").read_text(encoding="utf-8")
    assert rl.system_prompt("eqd", "v4") in appendix
