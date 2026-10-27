"""8/31 報告の 3 回目の改訂版（10/27）の PDF に、10/21 版から変えた箇所のハイライトとコメント（PDF の注釈）を付ける.

入力: docs/progress_report_2026-08-31_methods_results.pdf（10/27 改訂版。ページの中身は変えない）
出力: docs/progress_report_2026-08-31_methods_results_revision3_comments.pdf

2 回目（annotate_revision_0831_r2.py）と同じ形にする（加藤先生の注釈と同じハイライト注釈と、ポップアップで開くコメント）。
- 比べる相手は、加藤先生が 3 回目のコメント（2026-10-25）を付けた 10/21 版（f1076cd）。
- コメントは「修正：…」などの見出しと内容だけ。加藤先生のコメントによる修正には理由を書かない（本人の指示、10/21）。
  応えたコメントは、各 Note の reasons に記録として残す（K1〜K13 は docs/加藤先生コメント対応_2026-08-31報告_3回目.md）。
- 色: 黄 = 10/21 版へのコメントを受けて書き直した箇所、青 = 10/21 版で予告した再計算の値に替えた箇所、
  橙 = 結果がまだで、次の版で示す箇所。1 ページ目に色の意味を書いた付箋を置く。
- 場所は改訂版 PDF の文字列で決める（NFKC にそろえ、空白を除いて照合。ヘッダーとページ番号は除く）。
  見つからないか 2 か所以上にあれば止まる。
実行: ~/.pyenv/versions/3.12.9/bin/python3 annotate_revision_0831_r3.py（PyMuPDF が要る）
"""
from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).parent
SRC = ROOT / "docs" / "progress_report_2026-08-31_methods_results.pdf"
OUT = ROOT / "docs" / "progress_report_2026-08-31_methods_results_revision3_comments.pdf"
AUTHOR = "一洋"
COLORS = {"y": (1.0, 0.82, 0.0), "b": (0.45, 0.70, 1.0), "o": (1.0, 0.68, 0.30)}
HEADS = {"y": "修正", "b": "再計算", "o": "次の版で示す"}
NOW = fitz.get_pdf_now()
HEADER_Y, FOOTER_Y = 60, 785

# 加藤先生の 10/21 版への 3 回目のコメント（2026-10-25。記録として残す。コメントには引かない）
KATO3 = {
    "K1": "p.2 記号が同じなら同じ変数とみなす → 影響を Discussion か Future work で述べる",
    "K2": "p.3 N1 → 影響を Discussion か Future work で述べる。N1 の決め方と、さらに大きくしたときは Future work",
    "K3": "p.5 teacher forcing で正解式を第 1 段の順位の順に足すのでよいか",
    "K4": "p.6 文献の集め方は、修論と論文では Appendix に書く",
    "K5": "p.6 初期の 9 報も Gemini 2.5 Pro で抽出し直して、方法をそろえられないか",
    "K6": "p.7 9 組の語で除いたのは、列挙してからか、列挙するときか",
    "K7": "p.7 微分方程式を含む組を優先した、の意味がわからない",
    "K8": "p.7 LLM が作ったモデルや式を人手で確かめる（33 件と、LLM が作った数式の抜き取り）",
    "K9": "p.8 拡張ケースの段落は論文には入れない",
    "K10": "p.8〜9 分割の方法がわからない。400 通りの候補はどう作ったのか",
    "K11": "p.9 LLM のモデルは費用に応じて 2 つに減らしてよい",
    "K12": "p.15 重複は除かないといけないのか。重複の定義と判定は難しいはず",
    "K13": "p.15 「多くは言い換え」と言えれば多くの場合に使える。定義文の検討は今後の課題に",
}


@dataclass
class Note:
    start: str
    end: str | None
    cat: str            # y / b / o
    fix: str
    reasons: list[str] = field(default_factory=list)  # 応えた先生のコメント（K）。コメントには書かない
    why: str = ""       # それ以外の理由。あるときだけ「理由：」として書く
    pre: str = ""

    def content(self) -> str:
        if self.why:
            return f"{HEADS[self.cat]}：{self.fix}\n理由：{self.why}"
        return f"{HEADS[self.cat]}：{self.fix}"


N = Note
NOTES: list[Note] = [
    # ---------------- §1 ----------------
    N("また、出典横断のケース 113 件の説明文を入出力変数に合うように書き直し", "この 1,731 件で計算し直した。", "b",
      "10/21 版で予告した再計算が終わったので、§3.4 以降と付録 A〜C の数値を、説明文を書き直し、1 件を除いた 1,731 件の値に替えた。"
      "提案手法がどの指標でも 10 通りすべてで baseline を上回ることは変わらない。上回った分割の数が変わった主張は、各節のコメントに書いた。"),
    # ---------------- §3.2 ----------------
    N("分類の組ごとに、データベースの順に 2 式の組を調べ", "2 式とも微分方程式である組から作ったものは 277 件）。", "y",
      "ケースを作ったプログラムのとおりに書き直した。9 組の語の規則は、2 式の組を列挙するときに当てていた。"
      "微分方程式を含む組だけで 500 件に達したので、使った 101 組はすべて微分方程式を含む。", ["K6", "K7"]),
    # ---------------- §3.3 ----------------
    N("そこで、割り当ての候補を 400 通り無作為に作り", "10 通りの異なる分割で評価することになる。", "y",
      "候補の作り方を書いた。乱数の種ごとに、乱数生成器から引いた 400 個の数で出典の並びを 400 通り作り、"
      "L1 距離の和が最小のものを選ぶ。候補は種ごとに別のもので、L1 距離は訓練とテストの間だけで比べる。", ["K10"]),
    # ---------------- §3.4 ----------------
    N("（20 通りのすべてで hard negative）", None, "b",
      "計算し直すと、不正解の選び方は 20 通りのすべてで hard negative になった（10/21 版では 19 通り）。"),
    # ---------------- §4 ----------------
    N("一括の並べ直しは、設定 A で Recall@20 を 0.906 から 0.981 に", None, "b",
      "§4 と表 3 の数値を替えた。設定 A の MAP は 0.690 → 0.693（baseline）、0.860 → 0.862（一括）、0.874 → 0.877（訓練も逐次）。"
      "Recall@20 と設定 B の値は変わらない。"),
    # ---------------- §5 ----------------
    N("Recall@20 は 0.907 から 0.904 に、MAP は 0.693 から 0.691 に下がる", "10 通り中 3 通り）。", "b",
      "x1 と x2 だけの MLP が baseline を上回った分割は、Recall@20 で 4 通りから 3 通りになった。重みを学び直しても精度が上がらないことは変わらない。"),
    N("Recall@20 は 0.904 から 0.975 に", "この段階の向上はほとんど式の特化度による。", "b",
      "向上は Recall@20 で +0.083 から +0.071 に小さくなった。出発点の x1 と x2 だけの MLP が上がったためである"
      "（10/21 版では、種 3141 の分割だけ 0.800 と低かった）。式の特化度による向上であることは変わらない。"),
    N("（10 通りのうち、Recall@20 では 9 通り、MAP ではすべてで上回った）。", None, "b",
      "Recall@20 で上回った分割が 10 通りから 9 通りになったので、「どちらも 10 通りすべて」を直した。"),
    N("（どちらも 10 通りすべて）、分野ラベルの一致度（集合）", "上がった。", "b",
      "x10 を加えたときに MAP で上回った分割が 10 通りから 9 通りになったので、「どれも 10 通りすべて」を直した。"),
    N("Recall@20 は 0.981 から 0.018 に", None, "b",
      "PFI の値と図 2 を、計算し直したものに替えた。結論は変わらない。"),
    # ---------------- §6 ----------------
    N("（10 通り中 7 通りで上回った）", None, "b",
      "設定 A の (3) と (2) の差（+0.002）で上回った分割が、9 通りから 7 通りになった。表 4 の値も替えた。"),
    # ---------------- §7 ----------------
    N("（10 通り中 5 通りで上回った）、潜在意味類似度", "（9 通り）であった。", "b",
      "x7 を加えたときの差は +0.004 から −0.008 に、x3 は −0.012 から −0.021 になった。x10 の MAP で上回った分割は 9 通りになった。"
      "話題の特徴量を加えても精度をほとんど上げないことは変わらない。"),
    N("（どれも上回ったのは 10 通り中 3〜5 通り）。", "どの場合も 10 通り中 5 通り以下である。", "b",
      "分野ラベルを合併したときの差を替えた。特徴量が 2〜3 個の MLP では 1 つの分割だけ大きく下がることがあり、"
      "差の平均がその分割に左右されることを書き足した。"),
    N("ほとんど変わらなかった（変化は 0.001 未満）。", None, "b",
      "表 5 の AUC と、x10 の PFI をケースで分けた値を替えた。2 つの群で効く向きが逆になる様子は、さらに小さくなった。"),
    # ---------------- §8 ----------------
    N("テストのケース 407 件、正解式は延べ 1,622 本。", None, "b",
      "1 件を除いたので、種 42 の分割の値を替えた。上位 20 件に入らない正解式がすべて第 2 段の並べ方で生まれることは変わらない。"),
    # ---------------- §9 ----------------
    N("同じ式が上位 20 件の枠を占めると", "見つけるのは難しい。", "y",
      "「重複を除く必要がある」を、重複で起きることと、同じ式を見つける難しさの記述に改めた。", ["K12"]),
    # ---------------- §10 ----------------
    N("LLM が作った評価ケースと式（基礎式集）が妥当かを、人手で確かめる。", "別の出典にある同じ式の扱いを検討する。", "y",
      "今後の課題に、LLM が作ったデータを人手で確かめること、定義の文章も使う照合、N1 の決め方と同じ式の扱いを加えた。",
      ["K1", "K2", "K8", "K12", "K13"]),
    # ---------------- 付録 ----------------
    N("BM25 で 0.905（設定 A）", None, "b", "表 6 と表 7 の設定 A の値を替えた。"),
    N("延べ 4,229 件のケース平均", None, "b", "表 8 の値を替えた。"),
    N("（設定 A の Recall@20 は、500 でも 2,000 でも 0.981）。", "（設定 A で 66 秒と 161 秒）。", "b",
      "表 9 と図 3 の設定 A の値を替えた。時間の比は約 3 倍から約 2.4 倍になった（時間は計算機の負荷で変わる）。"),
    # ---------------- 次の版 ----------------
    N("LLM との比較は、§3.4 の方法で準備しており、結果は次の版で示す。", None, "o",
      "LLM との比較は、2 つのモデルの組み合わせと、データベースの扱い（初期の 9 報）が決まってから実行し、次の版で示す。", ["K5", "K11"]),
]

LEGEND = ("10 月 21 日版から変えた箇所に、ハイライトとコメントを付けた。\n"
          "黄：10/21 版へのコメントを受けて書き直した箇所\n"
          "青：10/21 版で予告した再計算（説明文を書き直し、1 件を除いた 1,731 件）の値に替えた箇所\n"
          "橙：結果がまだで、次の版で示す箇所\n"
          "先生のコメントによる修正には、理由を書いていない。")


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", s) if not c.isspace())


def char_stream(doc):
    chars = []
    for pno, page in enumerate(doc):
        for bi, b in enumerate(page.get_text("rawdict")["blocks"]):
            for li, l in enumerate(b.get("lines", [])):
                if l["bbox"][3] <= HEADER_Y or l["bbox"][1] >= FOOTER_Y:
                    continue
                for sp in l["spans"]:
                    for ch in sp["chars"]:
                        for c in norm(ch["c"]):
                            chars.append((c, pno, (bi, li), fitz.Rect(ch["bbox"])))
    return chars, "".join(c[0] for c in chars)


def locate(text: str, n: Note) -> tuple[int, int]:
    key = norm(n.pre + n.start)
    cnt = text.count(key)
    if cnt != 1:
        sys.exit(f"開始の文字列が {cnt} か所にある: {n.start[:50]!r}")
    i = text.index(key) + len(norm(n.pre))
    j = i + len(norm(n.start))
    if n.end is not None:
        k = text.find(norm(n.end), j)
        if k < 0:
            sys.exit(f"終わりの文字列がない: {n.end[:50]!r}（開始 {n.start[:40]!r}）")
        j = k + len(norm(n.end))
    if j - i > 1400:
        sys.exit(f"ハイライトが長すぎる（{j - i} 文字）: {n.start[:40]!r}")
    return i, j


def main():
    for n in NOTES:
        if bad := [k for k in n.reasons if k not in KATO3]:
            sys.exit(f"記録にない理由の記号 {bad}: {n.start[:30]!r}")
    doc = fitz.open(SRC)
    chars, text = char_stream(doc)
    spans = [(n, *locate(text, n)) for n in NOTES]
    spans.sort(key=lambda s: s[1])
    for (a, i1, j1), (b, i2, j2) in zip(spans, spans[1:]):
        if i2 < j1:
            sys.exit(f"ハイライトが重なる: {a.start[:30]!r} と {b.start[:30]!r}")
    for n, i, j in spans:
        by_page: dict[int, dict] = {}
        for k in range(i, j):
            c, pno, line, r = chars[k]
            by_page.setdefault(pno, {}).setdefault(line, fitz.Rect(r)).include_rect(r)
        first = True
        for pno, lines in by_page.items():
            page = doc[pno]
            rects = sorted(lines.values(), key=lambda r: (round(r.y0), r.x0))
            annot = page.add_highlight_annot(quads=[r.quad for r in rects])
            annot.set_colors(stroke=COLORS[n.cat])
            annot.set_opacity(0.9)
            body = n.content() if first else "前のページから続くハイライトです（コメントは前のページ）。"
            annot.set_info(title=AUTHOR, content=body, subject="修正箇所", creationDate=NOW, modDate=NOW)
            y = rects[0].y0
            annot.set_popup(fitz.Rect(page.rect.width + 2, y, page.rect.width + 222, y + 140))
            annot.update()
            first = False
    page = doc[0]
    note = page.add_text_annot(fitz.Point(page.rect.width - 40, 66), LEGEND, icon="Comment")
    note.set_info(title=AUTHOR, subject="この PDF の見方", creationDate=NOW, modDate=NOW)
    note.set_popup(fitz.Rect(page.rect.width + 2, 60, page.rect.width + 262, 200))
    note.update()
    doc.save(OUT, garbage=3, deflate=True)
    counts = {c: sum(1 for n in NOTES if n.cat == c) for c in COLORS}
    longest = max(len(n.content()) for n in NOTES)
    print(f"コメント {len(NOTES)} 件（黄 {counts['y']}、青 {counts['b']}、橙 {counts['o']}。"
          f"最長 {longest} 字）＋見方の付箋 1 件 → {OUT.name}")


if __name__ == "__main__":
    main()
