"""修士論文の点検.

決まり（2026-11-18、本人から）：本文は 40 ページ以下。付録は数えない。本文は論文と同じ書き方でよい。
この修論は、本文と付録 A〜J を、投稿論文 paper/cace/ の節のファイルから読み込む。

調べること
- 本文（1 章から結論の終わりまで。ページ番号は 1 から）が BODY_PAGE_LIMIT ページ以下であること
- 解決していない参照と引用がないこと
- 論文の本文の節（Abstract から Conclusion まで）を、同じ順にすべて読み込んでいること

使い方: 先に thesis/master_thesis で latexmk main.tex を実行してから、python3 thesis/master_thesis/check_thesis.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1] / "paper" / "cace"
BODY_PAGE_LIMIT = 40
END_LABEL = "thesis:endbody"
PAPER_ONLY = ("highlights", "keywords", "backmatter")   # 論文だけにある節（修論は読み込まない）


def body_pages(aux: str) -> int:
    """main.aux から、本文の最後（結論の終わりに置いたラベル）のページ番号を読む."""
    m = re.search(r"\\newlabel\{%s\}\{\{[^{}]*\}\{(\d+)\}" % re.escape(END_LABEL), aux)
    if not m:
        raise ValueError(f"main.aux has no label {END_LABEL}: build the thesis first (latexmk main.tex)")
    return int(m.group(1))


def unresolved(log: str) -> list:
    """main.log にある、解決していない参照と引用の名前."""
    return sorted(set(re.findall(r"Warning: (?:Reference|Citation) `([^']+)' on page \d+ undefined", log)))


def paper_sections(tex: str) -> list:
    """読み込んでいる論文の節（sections/ の下のファイル）の名前を、読み込む順に。コメントの行は数えない."""
    lines = [ln for ln in tex.splitlines() if not ln.lstrip().startswith("%")]
    return re.findall(r"\\input\{(?:\.\./\.\./paper/cace/)?sections/([a-z]+)\}", "\n".join(lines))


def check(thesis: Path = HERE, paper: Path = PAPER, report: list | None = None) -> list:
    report = [] if report is None else report
    failures = []
    n = body_pages((thesis / "main.aux").read_text(encoding="utf-8"))
    report.append(f"main text: {n} pages (limit {BODY_PAGE_LIMIT})")
    if n > BODY_PAGE_LIMIT:
        failures.append(f"the main text has {n} pages, over the limit of {BODY_PAGE_LIMIT}")
    if names := unresolved((thesis / "main.log").read_text(encoding="utf-8", errors="replace")):
        failures.append(f"unresolved references or citations: {', '.join(names)}")
    mine = paper_sections((thesis / "main.tex").read_text(encoding="utf-8"))
    theirs = [s for s in paper_sections((paper / "main.tex").read_text(encoding="utf-8")) if s not in PAPER_ONLY]
    if mine != theirs:
        failures.append(f"the thesis reads the sections {mine}, the paper has {theirs}")
    return failures


def main() -> int:
    report: list = []
    failures = check(report=report)
    print("\n".join(report))
    for line in failures:
        print("FAIL", line)
    print("all checks passed" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
