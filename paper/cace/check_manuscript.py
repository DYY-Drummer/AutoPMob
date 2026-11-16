"""Check the manuscript against the rules of the plan (docs/superpowers/plans/2026-11-16-cace-paper.md).

    python3 paper/cace/check_manuscript.py      # prints a report; exit 1 when a rule is broken

Rules
  - No number is typed by hand in sections/*.tex or appendix.tex: decimals (0.981) and counts (1{,}731)
    must be macros of numbers.tex. Lengths, model names (Opus 5.5), comments and verbatim prompts are exempt.
  - No number macro is followed by a bare space (TeX would swallow the space).
  - None of the wordings that the paper must not use (BANNED) appears, with the macros expanded.
  - The abstract has at most ABSTRACT_WORDS words, the main text MAIN_TEXT_WORDS words, and each
    highlight at most HIGHLIGHT_CHARS characters.
  - numbers.tex and the generated tables are current (make_numbers.py --check).
Word counts follow texcount's convention: text and headings count; mathematics, floats, citations,
and markers for the authors do not.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = ["abstract", "introduction", "related", "formulation", "method", "dataset", "setup", "results",
        "discussion", "conclusion"]
MAIN_TEXT_WORDS = (8500, 10000)   # Abstract to Conclusion (design document, success criterion 2)
ABSTRACT_WORDS = 250              # the author confirmed this limit in the guide for authors
HIGHLIGHT_CHARS = 85              # assumed from the publisher's general rule; the journal's guide is unconfirmed

VERBATIM = ("Verbatim", "verbatim")
NOT_PROSE = ("table", "figure", "equation", "align", "longtable", "tabular") + VERBATIM
MODEL_NAME = re.compile(r"(?:Opus|Haiku|Sonnet|Gemini)[~\s]+\d\.\d")
LENGTH = re.compile(r"\d+(?:\.\d+)?(?:\\linewidth|\\textwidth|\\columnwidth|pt|em|ex|cm|mm)\b")
NUMBER = re.compile(r"\d{1,3}\{,\}\d{3}|\d{1,3},\d{3}|\d+\.\d+")
NO_TEXT_COMMANDS = ("cite[a-z]*", "ref", "eqref", "label", "input", "includegraphics", "rowsfrom",
                    "bibliography", "bibliographystyle", "setlength", "vspace", "hspace")

# wording -> pattern (matched without regard to case, on text whose white space is collapsed)
BANNED = {
    "329 documents": r"329 documents",
    "no document leaks": r"no document leak",
    "untested": r"untested",
    "guarantee": r"guarantee",
    "duplicate": r"duplicat",
    "augmented": r"augmented",
    "thesis": r"\bthesis\b",
    "renamed symbols": r"\brenam",
    "document-level": r"document-level",
    "significant": r"significan",
}


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def strip_environments(tex: str, names) -> str:
    for name in names:
        tex = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (name, name), " ", tex, flags=re.S)
    return tex


def drop_command(tex: str, name: str) -> str:
    """Remove \\name{...} together with its argument (braces may nest)."""
    out, i, key = [], 0, "\\" + name + "{"
    while True:
        j = tex.find(key, i)
        if j < 0:
            return "".join(out) + tex[i:]
        out.append(tex[i:j])
        depth, k = 1, j + len(key)
        while k < len(tex) and depth:
            depth += {"{": 1, "}": -1}.get(tex[k], 0)
            k += 1
        i = k


def hand_typed_numbers(tex: str) -> list:
    """Decimals and counts with a thousands separator that are typed in the text."""
    tex = strip_comments(strip_environments(tex, VERBATIM))
    lines = [ln for ln in tex.splitlines() if "\\newcommand" not in ln and "\\renewcommand" not in ln]
    text = MODEL_NAME.sub(" ", LENGTH.sub(" ", "\n".join(lines)))
    return NUMBER.findall(text)


def read_macros(numbers_tex: str) -> dict:
    return dict(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", numbers_tex, flags=re.M))


def expand(tex: str, macros: dict) -> str:
    """Replace the number macros by their values; every other command stays."""
    def repl(m):
        if m.group(1) not in macros:
            return m.group(0)
        return macros[m.group(1)] + (" " if m.group(2) == "\\ " else "")
    return re.sub(r"\\([A-Za-z]+)(\{\}|\\ )?", repl, tex)


def swallowed_spaces(tex: str, macros: dict) -> list:
    """Number macros followed by a bare space and a word: TeX drops the space."""
    tex = strip_comments(strip_environments(tex, VERBATIM))
    return [m.group(0) for m in re.finditer(r"\\([A-Za-z]+) [A-Za-z0-9(]+", tex) if m.group(1) in macros]


def word_count(tex: str) -> int:
    tex = strip_comments(strip_environments(tex, NOT_PROSE))
    tex = drop_command(tex, "todo")
    tex = tex.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    tex = re.sub(r"\\\[.*?\\\]", " ", tex, flags=re.S)
    tex = re.sub(r"\$[^$]*\$", " ", tex)
    tex = re.sub(r"\\(?:%s)\*?(?:\[[^\]]*\])*(?:\{[^{}]*\})*" % "|".join(NO_TEXT_COMMANDS), " ", tex)
    tex = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", tex)
    tex = tex.replace("\\\\", " ").replace("~", " ").replace("{", "").replace("}", "")
    return sum(1 for w in tex.split() if re.search(r"[A-Za-z0-9]", w))


def banned_phrases(text: str) -> list:
    flat = re.sub(r"\s+", " ", text)
    return [name for name, pattern in BANNED.items() if re.search(pattern, flat, flags=re.I)]


def long_highlights(tex: str, limit: int = HIGHLIGHT_CHARS) -> list:
    items = [re.sub(r"\s+", " ", re.sub(r"[{}$]", "", it.replace("~", " "))).strip()
             for it in strip_comments(tex).split("\\item")[1:]]
    return [it for it in items if len(it) > limit]


def check(root: Path = HERE, report: list | None = None) -> list:
    """The broken rules, one line each. `report` collects the measured values."""
    report = [] if report is None else report
    macros = read_macros((root / "numbers.tex").read_text(encoding="utf-8"))
    failures = []
    files = sorted((root / "sections").glob("*.tex")) + [root / "appendix.tex"]
    for f in files:
        tex = f.read_text(encoding="utf-8")
        name = f.relative_to(root)
        if hand := hand_typed_numbers(tex):
            failures.append(f"{name}: numbers typed by hand: {', '.join(hand)}")
        if swallowed := swallowed_spaces(tex, macros):
            failures.append(f"{name}: a macro swallows the next space: {', '.join(swallowed)}")
        prose = expand(strip_comments(strip_environments(tex, VERBATIM)), macros)
        if banned := banned_phrases(prose):
            failures.append(f"{name}: wording the paper must not use: {', '.join(banned)}")
    words = {s: word_count(expand((root / "sections" / f"{s}.tex").read_text(encoding="utf-8"), macros)) for s in BODY}
    total = sum(words.values())
    report.append("words: " + ", ".join(f"{s} {n}" for s, n in words.items()) + f"; main text {total}")
    if words["abstract"] > ABSTRACT_WORDS:
        failures.append(f"abstract: {words['abstract']} words, over {ABSTRACT_WORDS}")
    if not MAIN_TEXT_WORDS[0] <= total <= MAIN_TEXT_WORDS[1]:
        failures.append(f"main text: {total} words, outside {MAIN_TEXT_WORDS[0]}-{MAIN_TEXT_WORDS[1]}")
    highlights = expand((root / "sections" / "highlights.tex").read_text(encoding="utf-8"), macros)
    for item in long_highlights(highlights):
        failures.append(f"highlight over {HIGHLIGHT_CHARS} characters ({len(item)}): {item}")
    n_todo = sum(strip_comments(f.read_text(encoding="utf-8")).count("\\todo{") for f in files + [root / "main.tex"])
    report.append(f"markers for the authors (\\todo): {n_todo}")
    sys.path.insert(0, str(root))
    import make_numbers
    if make_numbers.main(["--check"]) != 0:
        failures.append("numbers.tex or a generated table is stale: run python3 paper/cace/make_numbers.py")
    return failures


def main(argv=None) -> int:
    report: list = []
    failures = check(report=report)
    print("\n".join(report))
    for line in failures:
        print("FAIL", line)
    print("all checks passed" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
