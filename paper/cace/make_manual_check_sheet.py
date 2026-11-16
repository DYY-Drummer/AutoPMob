"""人手の確認のための標本とチェックシートを作る.

評価用のケース（設定 A、1,731 件）から、3 つの種類（単一出典、出典横断、合成）の件数の割合に合わせて 50 件を
無作為に選ぶ（種は固定）。1 件ごとに、説明文、入出力変数、正解式（表記、出典、分野、文脈、変数の定義）、
2 本以上の正解式に現れる記号の定義を並べた HTML を作る。確認する人は、基準 C1〜C4 のそれぞれに
「はい／いいえ」（C3 は「対象外」も）を付け、結果を CSV にして返す。

基準は、確認の前に決めてこのファイルに書く（CRITERIA）。論文には、実際にこの基準で確かめた結果だけを書く。

出力:
  paper/cace/data/manual_check_sample.json   標本（種、割り当て、ケースの ID）。git で追跡する
  docs/build/manual_check/check_sheet.html   チェックシート。式データベースの式を含むので git に入れない
結果の置き場所: paper/cace/data/manual_check_results.csv（列は RESULT_COLUMNS。確認した人が作る）

使い方: python3 paper/cace/make_manual_check_sheet.py
"""
from __future__ import annotations

import ast
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze_split_overlap as so  # noqa: E402

CASES_FILE = ROOT / "training_cases.json"
DB_FILE = ROOT / "unified_equations.json"
FLAGS_FILE = ROOT / "experiments" / "var_descriptions.json"   # 記号の意味の違いを Claude Opus 5.5 が判定した結果
SAMPLE_JSON = HERE / "data" / "manual_check_sample.json"
SHEET_HTML = ROOT / "docs" / "build" / "manual_check" / "check_sheet.html"
RESULTS_CSV = HERE / "data" / "manual_check_results.csv"

SEED = 20261116
N_SAMPLE = 50
KINDS = {"single": "単一出典", "cross": "出典横断", "synth": "合成"}
# 基準：(記号, 見出し, 「はい」と答える条件)
CRITERIA = (
    ("C1", "説明文と正解式", "説明文が求めているモデル（装置や現象）を、正解式が表している。"),
    ("C2", "出力変数", "出力変数は、説明文が求めている量である。"),
    ("C3", "記号の意味", "2 本以上の正解式に現れる記号は、どの式でも同じ量を指している。共通の記号がないケースは「対象外」。"),
    ("C4", "入力から出力が求まる", "入力変数の値を与えると、正解式を組み合わせて出力変数が求まる。"),
)
RESULT_COLUMNS = ["case_id"] + [c[0] for c in CRITERIA] + ["note"]


# ---- 標本 ---------------------------------------------------------------------------------------

def allocate(sizes: dict, n: int) -> dict:
    """n 件を、各種類の件数の割合に合わせて割り当てる（端数は、切り捨てた分の大きい種類から 1 件ずつ足す）."""
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"the sample ({n}) is larger than the population ({total})")
    quota = {k: n * v / total for k, v in sizes.items()}
    out = {k: int(q) for k, q in quota.items()}
    for k in sorted(sizes, key=lambda k: (-(quota[k] - out[k]), k))[:n - sum(out.values())]:
        out[k] += 1
    return out


def draw_sample(population: dict, n: int, seed: int) -> list:
    """{種類: ケースの ID} から n 件を、種類の件数の割合に合わせて無作為に選ぶ。同じ種からは同じ標本が出る
    （ID の並び順や種類の並び順に左右されない）。返す並びは、種類（KINDS の順）ごとに ID の昇順."""
    alloc = allocate({k: len(v) for k, v in population.items()}, n)
    out = []
    for k in [k for k in KINDS if k in population] + sorted(k for k in population if k not in KINDS):
        rng = random.Random(f"{seed}:{k}")
        out += sorted(rng.sample(sorted(population[k]), alloc[k]))
    return out


def load_cases() -> dict:
    return {c["case_id"]: c for c in json.loads(CASES_FILE.read_text(encoding="utf-8"))}


def load_population(cases: dict) -> dict:
    """設定 A の評価用のケース（除外したケースを含まない）を、種類ごとに分ける."""
    split = next(iter(json.loads(so.SPLITS.read_text(encoding="utf-8"))["splits"]["A"].values()))
    ids = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    out = {k: [] for k in KINDS}
    for c in ids:
        out[so.case_kind(cases[c])].append(c)
    return out


def build_sample() -> dict:
    population = load_population(load_cases())
    sizes = {k: len(v) for k, v in population.items()}
    return {"rule": "設定 A の評価用のケースから、種類の件数の割合に合わせて無作為に選ぶ（make_manual_check_sheet.draw_sample）",
            "seed": SEED, "n": N_SAMPLE, "population": sizes, "allocation": allocate(sizes, N_SAMPLE),
            "cases": draw_sample(population, N_SAMPLE, SEED)}


# ---- 1 件分の材料 -------------------------------------------------------------------------------

def _literal(x, default):
    if isinstance(x, str):
        return ast.literal_eval(x) if x.strip() else default
    return x if x is not None else default


def load_equations() -> dict:
    """式データベースを {「出典__式番号」: 式} にする（git で追跡していないので、手元にあるときだけ使える）."""
    if not DB_FILE.exists():
        raise FileNotFoundError(f"{DB_FILE} がない。チェックシートは式データベースのある環境で作る")
    out = {}
    for e in json.loads(DB_FILE.read_text(encoding="utf-8")):
        eid = (e.get("eq_id") or "").strip()
        key = eid if "__" in eid else f"{(e.get('source_id') or 'unknown').strip()}__{eid}"
        out[key] = {"id": key, "latex": e.get("equation") or "", "source": e.get("source_id") or "",
                    "domain": e.get("domain") or "", "context": e.get("context_text") or "",
                    "variables": {str(k).strip(): str(v) for k, v in _literal(e.get("variables"), {}).items()}}
    return out


def shared_symbols(eqs: list) -> dict:
    """2 本以上の式に現れる記号 → [(式の ID, その式での定義)]."""
    seen = {}
    for e in eqs:
        for sym, definition in e["variables"].items():
            seen.setdefault(sym, []).append((e["id"], definition))
    return {s: v for s, v in seen.items() if len(v) >= 2}


def other_variables(eqs: list, inputs: list, outputs: list) -> list:
    """入力変数でも出力変数でもない変数（式に現れる順）."""
    known = set(inputs) | set(outputs)
    return list(dict.fromkeys(s for e in eqs for s in e["variables"] if s not in known))


def case_record(case: dict, eq_index: dict, flags: dict) -> dict:
    eqs = [eq_index[i] for i in so.ground_truth_ids(case)]
    inputs = [s.strip() for s in _literal(case.get("input_variables"), [])]
    outputs = [s.strip() for s in _literal(case.get("output_variables"), [])]
    flagged = [s for s, v in (flags.get(case["case_id"], {}).get("variables") or {}).items() if v.get("meaning_conflict")]
    return {"case_id": case["case_id"], "kind": so.case_kind(case), "description": case.get("context") or "",
            "inputs": inputs, "outputs": outputs, "equations": eqs, "shared": shared_symbols(eqs), "flagged": flagged,
            "others": other_variables(eqs, inputs, outputs)}


# ---- チェックシート（HTML） ----------------------------------------------------------------------

def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _tex(s, display: bool = False) -> str:
    """数式や記号を置く場所。KaTeX が読み込めたときはスクリプトが組み、読み込めないときは LaTeX のまま見せる."""
    return f'<span class="tex{" display" if display else ""}" data-tex="{_e(s)}">{_e(s)}</span>'


def _symbols(syms: list) -> str:
    return "、".join(_tex(s) for s in syms) if syms else "（なし）"


def _case_html(i: int, n: int, r: dict) -> str:
    num = {e["id"]: k + 1 for k, e in enumerate(r["equations"])}
    role = {**{s: "入力" for s in r["inputs"]}, **{s: "出力" for s in r["outputs"]}}
    parts = [f'<section class="case" data-case="{_e(r["case_id"])}" data-idx="{i}">',
             f'<h2><span class="num">{i + 1} / {n}</span> {_e(r["case_id"])} '
             f'<span class="kind">{_e(KINDS.get(r["kind"], r["kind"]))}</span> '
             f'<span class="kind">正解式 {len(r["equations"])} 本</span></h2>',
             f'<p class="desc">{_e(r["description"])}</p>',
             f'<p><b>入力変数</b>：{_symbols(r["inputs"])}</p>',
             f'<p><b>出力変数</b>：{_symbols(r["outputs"])}</p>']
    if r["others"]:
        parts.append(f'<p><b>入力でも出力でもない変数</b>：{_symbols(r["others"])}</p>')
    for e in r["equations"]:
        rows = "".join(f'<tr><td>{_tex(s)}</td><td class="role">{_e(role.get(s, ""))}</td><td>{_e(d)}</td></tr>'
                       for s, d in e["variables"].items())
        parts.append(f'<div class="eq"><div class="src">[{num[e["id"]]}] {_e(e["id"])}（分野：{_e(e["domain"])}）</div>'
                     f'<div class="math">{_tex(e["latex"], display=True)}</div>'
                     f'<details><summary>LaTeX</summary><code>{_e(e["latex"])}</code></details>'
                     f'<div class="ctx">{_e(e["context"])}</div>'
                     f'<table class="vars">{rows}</table></div>')
    if r["shared"]:
        items = []
        for s, defs in r["shared"].items():
            same = len({d.strip().lower() for _, d in defs}) == 1
            flag = ' <span class="flag">言語モデルが「式によって別の量」と判定</span>' if s in r["flagged"] else ""
            body = "".join(f'<li>[{num[eid]}] {_e(d)}</li>' for eid, d in defs)
            items.append(f'<div class="sym{" same" if same else ""}">{_tex(s)}'
                         f'{"（定義の文言は同じ）" if same else ""}{flag}<ul>{body}</ul></div>')
        parts.append('<div class="shared"><b>2 本以上の正解式に現れる記号と、式ごとの定義</b>' + "".join(items) + '</div>')
    else:
        parts.append('<div class="shared none">2 本以上の正解式に現れる記号はない（C3 は対象外）。</div>')
    judge = []
    for code, title, _ in CRITERIA:
        options = [("Y", "はい"), ("N", "いいえ")] + ([("NA", "対象外")] if code == "C3" else [])
        radios = "".join(
            f'<label><input type="radio" name="case{i}_{code}" value="{v}"'
            f'{" checked" if code == "C3" and v == "NA" and not r["shared"] else ""}> {label}</label>'
            for v, label in options)
        judge.append(f'<div class="crit"><span class="code">{code}</span> {_e(title)}：{radios}</div>')
    judge.append(f'<div class="crit"><span class="code">メモ</span> <input type="text" name="case{i}_note" size="70" '
                 f'placeholder="「いいえ」の理由など"></div>')
    parts.append('<div class="judge">' + "".join(judge) + '</div></section>')
    return "\n".join(parts)


_PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>人手の確認のチェックシート</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css" crossorigin="anonymous">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js" crossorigin="anonymous"></script>
<style>
body { font-family: -apple-system, "Hiragino Sans", sans-serif; margin: 0; color: #1c1c1e; background: #f6f6f4; line-height: 1.6; }
main { max-width: 920px; margin: 0 auto; padding: 16px; }
#bar { position: sticky; top: 0; z-index: 5; background: #1f2937; color: #fff; padding: 8px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
#bar button { font-size: 14px; padding: 4px 12px; border-radius: 6px; border: 0; background: #e5e7eb; cursor: pointer; }
#msg { font-size: 13px; color: #d1fae5; }
.intro, .case { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; margin: 16px 0; }
.case.done { border-left: 6px solid #2f855a; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin: 4px 0 8px; }
.num { color: #6b7280; margin-right: 8px; } .kind { font-size: 12px; background: #e5e7eb; border-radius: 10px; padding: 1px 8px; margin-left: 4px; font-weight: normal; }
.desc { background: #fef9e7; border-left: 4px solid #d4a017; padding: 6px 10px; margin: 6px 0; }
.eq { border-top: 1px solid #e5e7eb; margin-top: 10px; padding-top: 8px; }
.src { font-size: 12px; color: #4b5563; word-break: break-all; } .ctx { font-size: 13px; color: #374151; margin: 4px 0; }
.math { overflow-x: auto; padding: 4px 0; } details { font-size: 12px; color: #6b7280; } code { word-break: break-all; }
table.vars { border-collapse: collapse; font-size: 13px; margin-top: 4px; } table.vars td { border-top: 1px solid #f0f0f0; padding: 1px 10px 1px 0; vertical-align: top; }
td.role { color: #b45309; white-space: nowrap; }
.shared { background: #eef4fb; border-radius: 6px; padding: 8px 12px; margin-top: 10px; font-size: 14px; } .shared.none { color: #6b7280; background: #f3f4f6; }
.sym { margin-top: 6px; } .sym.same { color: #6b7280; } .sym ul { margin: 2px 0 2px 18px; padding: 0; }
.flag { background: #fde2e2; color: #9b1c1c; font-size: 12px; border-radius: 4px; padding: 1px 6px; margin-left: 6px; }
.judge { border-top: 2px solid #1f2937; margin-top: 12px; padding-top: 8px; } .crit { margin: 3px 0; } .crit label { margin-left: 10px; }
.code { display: inline-block; min-width: 2.6em; font-weight: bold; }
ol.criteria li { margin: 4px 0; } textarea { width: 100%; height: 140px; font-family: ui-monospace, monospace; font-size: 12px; }
</style></head><body>
<div id="bar"><span id="progress"></span><button id="copy">結果をコピー</button><button id="save">CSV を保存</button><span id="msg"></span></div>
<main>
<div class="intro">
<h1>人手の確認のチェックシート（評価用のケース %%N%% 件）</h1>
<p>設定 A の評価用のケース %%POP%% 件から、種類の件数の割合に合わせて無作為に選んだ %%N%% 件です（%%ALLOC%%。乱数の種 %%SEED%%）。
1 件ごとに、下の 4 つの基準に「はい」か「いいえ」を付けてください。「いいえ」のときは、メモに理由を書いてください。
途中経過はこのブラウザに保存されます。終わったら「結果をコピー」を押して貼り付けるか、「CSV を保存」で保存したファイルを
<code>paper/cace/data/manual_check_results.csv</code> に置いてください。</p>
<ol class="criteria">%%CRITERIA%%</ol>
<p>確かめないこと：式が元の文献のとおりに抽出されているかどうか（この確認は、ケースの作り方だけを対象にします）。</p>
<p id="nomath" hidden>数式を組む部品（KaTeX）を読み込めなかったので、数式は LaTeX のまま表示しています。</p>
</div>
%%CASES%%
<div class="intro"><b>結果</b>（「結果をコピー」と同じ内容）<textarea id="out" readonly></textarea></div>
</main>
<script>
(function () {
  var KEY = "manual_check_%%SEED%%";
  var CODES = %%CODES%%;
  var cases = Array.prototype.slice.call(document.querySelectorAll("section.case"));
  var texs = Array.prototype.slice.call(document.querySelectorAll("[data-tex]"));
  if (window.katex) {
    texs.forEach(function (el) {
      try {
        katex.render(el.getAttribute("data-tex"), el, {throwOnError: false, displayMode: el.classList.contains("display")});
      } catch (err) { el.textContent = el.getAttribute("data-tex"); }
    });
  } else {
    document.getElementById("nomath").hidden = false;
  }
  function read(sec) {
    var i = sec.getAttribute("data-idx"), row = {case_id: sec.getAttribute("data-case")};
    CODES.forEach(function (c) {
      var el = sec.querySelector('input[name="case' + i + "_" + c + '"]:checked');
      row[c] = el ? el.value : "";
    });
    row.note = sec.querySelector('input[name="case' + i + '_note"]').value;
    return row;
  }
  function cell(s) { return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
  function csv() {
    var lines = [["case_id"].concat(CODES, ["note"]).join(",")];
    cases.forEach(function (sec) {
      var r = read(sec);
      lines.push([r.case_id].concat(CODES.map(function (c) { return r[c]; }), [r.note]).map(cell).join(","));
    });
    return lines.join("\\n") + "\\n";
  }
  function refresh() {
    var done = 0, state = {};
    cases.forEach(function (sec) {
      var r = read(sec), ok = CODES.every(function (c) { return r[c] !== ""; });
      sec.classList.toggle("done", ok);
      done += ok ? 1 : 0;
      state[r.case_id] = r;
    });
    document.getElementById("progress").textContent = "済み " + done + " / " + cases.length;
    document.getElementById("out").value = csv();
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (err) {}
  }
  try {
    var saved = JSON.parse(localStorage.getItem(KEY) || "{}");
    cases.forEach(function (sec) {
      var i = sec.getAttribute("data-idx"), r = saved[sec.getAttribute("data-case")];
      if (!r) { return; }
      CODES.forEach(function (c) {
        var el = r[c] && sec.querySelector('input[name="case' + i + "_" + c + '"][value="' + r[c] + '"]');
        if (el) { el.checked = true; }
      });
      sec.querySelector('input[name="case' + i + '_note"]').value = r.note || "";
    });
  } catch (err) {}
  document.addEventListener("change", refresh);
  document.addEventListener("input", refresh);
  function say(t) { document.getElementById("msg").textContent = t; }
  document.getElementById("copy").addEventListener("click", function () {
    var text = csv(), out = document.getElementById("out");
    function fallback() { out.focus(); out.select(); document.execCommand("copy"); say("コピーしました"); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { say("コピーしました"); }, fallback);
    } else { fallback(); }
  });
  document.getElementById("save").addEventListener("click", function () {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv()], {type: "text/csv"}));
    a.download = "manual_check_results.csv";
    document.body.appendChild(a); a.click(); a.remove();
    say("manual_check_results.csv を保存しました");
  });
  refresh();
})();
</script>
</body></html>
"""


def render_html(records: list, seed: int, sample: dict | None = None) -> str:
    n = len(records)
    alloc = "、".join(f"{KINDS[k]} {v} 件" for k, v in (sample or {}).get("allocation", {}).items())
    pop = sum((sample or {}).get("population", {}).values())
    criteria = "".join(f"<li><b>{c} {_e(t)}</b>：{_e(d)}</li>" for c, t, d in CRITERIA)
    page = _PAGE
    for key, value in (("%%CASES%%", "\n".join(_case_html(i, n, r) for i, r in enumerate(records))),
                       ("%%CRITERIA%%", criteria), ("%%CODES%%", json.dumps([c[0] for c in CRITERIA])),
                       ("%%SEED%%", str(seed)), ("%%N%%", str(n)), ("%%POP%%", f"{pop:,}" if pop else "—"),
                       ("%%ALLOC%%", alloc or "—")):
        page = page.replace(key, value)
    return page


def main() -> None:
    sample = build_sample()
    SAMPLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_JSON.write_text(json.dumps(sample, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cases, eq_index = load_cases(), load_equations()
    flags = json.loads(FLAGS_FILE.read_text(encoding="utf-8"))["cases"] if FLAGS_FILE.exists() else {}
    records = [case_record(cases[c], eq_index, flags) for c in sample["cases"]]
    SHEET_HTML.parent.mkdir(parents=True, exist_ok=True)
    SHEET_HTML.write_text(render_html(records, SEED, sample), encoding="utf-8")
    n_eq = sum(len(r["equations"]) for r in records)
    print(f"sample: {sample['allocation']} of {sample['population']} (seed {SEED}) -> {SAMPLE_JSON}")
    print(f"sheet: {len(records)} cases, {n_eq} ground-truth equations, "
          f"{sum(1 for r in records if r['shared'])} cases with a symbol in two or more equations, "
          f"{sum(1 for r in records if r['flagged'])} cases with a symbol that the language model flagged -> {SHEET_HTML}")


if __name__ == "__main__":
    main()
