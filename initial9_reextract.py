"""初期の 9 報を Gemini 2.5 Pro で抽出し直す準備と、古い抽出との対応づけ（2026-11-04）.

加藤先生の 3 回目のコメント K5（docs/加藤先生コメント対応_2026-08-31報告_3回目.md の「1. 初期の 9 報の抽出し直し」）の
判断材料を作る。unified_equations.json と training_cases.json は読むだけで、書き換えない。

サブコマンド:
  facts     9 報の式の数と、正解集合に 9 報の式を含むケース（設定 A：外す一覧 v3 を除いた 1,731 件）を数える。
  extract   9 報の PDF から式を抽出し直す。既定は見積もり（ページ数と費用）だけ。--run を付けたときだけ Gemini を呼ぶ
            （有料。本人の承認を取ってから）。抽出は extract_equations.extract_equations()（320 報と同じプロンプト・
            モデル・設定）をそのまま呼び、応答のモデルの版（model_version）とトークン数だけを横から記録する。
  match     古い抽出（DB の式）と新しい抽出を出典ごとに対応づけ、146 件のケースが新しい式に対応づけられるかを判定する。
  baseline  同じ PDF（バイト単位で同一）を Gemini 2.5 Pro で別々に抽出した DB の出典の組（重複して取得した文献）に、
            同じ対応づけを当てる。同じモデルで抽出し直しても生じる違いの目安にする。

式の対応づけ（match と baseline で共通）:
  1. 同じ式：set_aware_reranker.latex_key（空白と \\left・\\right・\\,・\\;・\\!・\\quad を除いた LaTeX）が一致する。
  2. 同じ式（表記の細部だけ違う）：さらに波括弧、\\cdot、\\times、\\mathrm などを除くと一致する。
  3. 残りの式は、LaTeX の文字列の近さ・変数記号の集合の重なり・式周辺の文章の近さの重み付き和で 1 対 1 に割り当て
     （ハンガリー法）、和が閾値 MATCH_THRESHOLD 以上の組だけを対応とみなす。変数記号の集合が同じなら
     「記号は同じで書き方が違う式」、違えば「記号の違う式」。
  4. 割り当てられなかった式は「古い抽出にだけある式」「新しい抽出にだけある式」。

ケースの判定（match）:
  9 報の正解式がすべて新しい式に対応づき、新しい式の変数で要件 1〜3（kato1021_counts.py と同じ定義）が成り立ち、
  入出力変数の記号がそのまま現れる → A「ID を付け替えるだけ」。記号の書き方（波括弧・空白など）をそろえれば成り立つ、
  または変数と式のつながりを保ったまま記号を 1 対 1 に付け替えられる（C_{TG} → c_{TG} など。find_relabel）→ B
  「入出力の記号も付け替える」。対応はつくが要件が崩れる → C「作り直しが要る」。対応のつかない正解式がある → D。
  PDF がなく抽出していない → E。

出力: experiments/initial9_reextract/（facts.json、cases_146.csv、raw/、match_*.csv、case_mapping.csv、*_summary.json）
使い方:
  python initial9_reextract.py facts
  python initial9_reextract.py extract --pdf-dir input_pdfs/initial9           # 見積もりだけ（API を呼ばない）
  python initial9_reextract.py extract --pdf-dir input_pdfs/initial9 --run     # Gemini を呼ぶ（有料）
  python initial9_reextract.py match
  python initial9_reextract.py baseline
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "experiments" / "initial9_reextract"
EXCLUDE = ROOT / "experiments" / "kato1020" / "excluded_cases_v3.json"

# 書誌は 2026-11-04 に Crossref・OpenAlex の API と配布元のページで確かめた。bytes は配布元の Content-Length。
# availability: on_hand（手元に PDF がある）、open（オープンアクセス。curl で取れる）、
#               open_browser（オープンアクセスだがサイトが自動取得を拒むのでブラウザで取る）、university（大学の契約が要る）
SOURCES = [
    {"source_id": "038.pdf", "availability": "open", "pages": 6,
     "authors": "Laszczyk, P., Niedźwiedź, M., Skupin, P., Metzger, M.", "year": 2017,
     "title": "Modelling of hybrid CSTR plant: Heat transfer considerations",
     "venue": "2017 21st International Conference on Process Control (PC), pp. 228-233",
     "doi": "10.1109/PC.2017.7976218",
     "url": "https://skoge.folk.ntnu.no/prost/proceedings/process-control-slovakia-2017/data/papers/038.pdf",
     "bytes": 693414},
    {"source_id": "1-s2.0-0009250974800898-main.pdf", "availability": "university", "pages": 19,
     "authors": "Uppal, A., Ray, W. H., Poore, A. B.", "year": 1974,
     "title": "On the dynamic behavior of continuous stirred tank reactors",
     "venue": "Chemical Engineering Science 29(4), 967-985", "doi": "10.1016/0009-2509(74)80089-8"},
    {"source_id": "1-s2.0-S1369703X09000059-main.pdf", "availability": "university", "pages": 7,
     "authors": "Al-Zuhair, S., Dowaidar, A., Kamal, H.", "year": 2009,
     "title": "Dynamic modeling of biodiesel production from simulated waste cooking oil using immobilized lipase",
     "venue": "Biochemical Engineering Journal 44(2-3), 256-262", "doi": "10.1016/j.bej.2009.01.003"},
    {"source_id": "Chemical Process Dynamics.pdf", "availability": "on_hand", "pages": 4,
     "authors": "Virtual Labs, IIT Kharagpur", "year": None,
     "title": "Chemical Process Dynamics: Continuous Stirred Tank Reactor (theory page)",
     "venue": "Web page printed to PDF on 2026-03-15", "doi": None,
     "url": "https://cpd-iitkgp.vlabs.ac.in/exp/continuous-stirred-tank-reactor/theory.html"},
    {"source_id": "WJET_2015040715405408.pdf", "availability": "open", "pages": 16,
     "authors": "Igbokwe, P. K., Nwabanne, J. T., Gadzama, S. W.", "year": 2015,
     "title": "Characterization of a 5 Litre Continuous Stirred Tank Reactor",
     "venue": "World Journal of Engineering and Technology 3(2), 25-40", "doi": "10.4236/wjet.2015.32003",
     "url": "https://www.scirp.org/pdf/WJET_2015040715405408.pdf", "bytes": 2858346},
    {"source_id": "applsci-10-00992-v2.pdf", "availability": "open", "pages": None,
     "authors": "Castillo Gónzalez, J. P., Álvarez Gutiérrez, P. E., Adam Medina, M., López Zapata, B. Y., "
                "Ramírez Guerrero, G. V., Vela Valdés, L. G.", "year": 2020,
     "title": "Effects on Biodiesel Production Caused by Feed Oil Changes in a Continuous Stirred-Tank Reactor",
     "venue": "Applied Sciences 10(3), 992", "doi": "10.3390/app10030992",
     "url": "https://mdpi-res.com/d_attachment/applsci/applsci-10-00992/article_deploy/applsci-10-00992-v2.pdf",
     "bytes": 1076366},
    {"source_id": "energies-17-01344.pdf", "availability": "open", "pages": None,
     "authors": "Răducanu, C. E., Dobre, T., Mihăiescu, D. E., Moroşan, A., Jidveian, R., Cioroiu Tîrpan, D. R., "
                "Vasiliu, A. D., Gogoaşă, C. I., Pârvulescu, O. C., Trică, B.", "year": 2024,
     "title": "Synthesis of Guanidine and Its Deposition on Bacterial Cellulose as Green Heterogeneous Catalyst "
              "for Transesterification to Methyl Esters",
     "venue": "Energies 17(6), 1344", "doi": "10.3390/en17061344",
     "url": "https://mdpi-res.com/d_attachment/energies/energies-17-01344/article_deploy/energies-17-01344.pdf",
     "bytes": 7087846},
    {"source_id": "kinetics_soybean_oil.pdf", "availability": "on_hand", "pages": 8,
     "authors": "Noureddini, H., Zhu, D.", "year": 1997,
     "title": "Kinetics of transesterification of soybean oil",
     "venue": "Journal of the American Oil Chemists' Society 74(11), 1457-1463", "doi": "10.1007/s11746-997-0254-2"},
    {"source_id": "matecconf_cscc2019_01027.pdf", "availability": "open_browser", "pages": None,
     "authors": "Šánek, L., Pecha, J., Husár, J., Kolomazník, K.", "year": 2019,
     "title": "Mathematical modeling of transesterification process kinetics of triglycerides catalyzed by TMAH",
     "venue": "MATEC Web of Conferences 292, 01027", "doi": "10.1051/matecconf/201929201027",
     "url": "https://www.matec-conferences.org/articles/matecconf/pdf/2019/41/matecconf_cscc2019_01027.pdf",
     "bytes": 1638377},
]
S9 = [s["source_id"] for s in SOURCES]
BY_ID = {s["source_id"]: s for s in SOURCES}

# Gemini 2.5 Pro の料金（2026-11-04 に https://ai.google.dev/gemini-api/docs/pricing で確認。プロンプト 20 万トークン以下）
PRICE_IN, PRICE_OUT = 1.25 / 1e6, 10.00 / 1e6  # USD／トークン。出力は思考のトークンを含む
TOKENS_PER_PAGE = 258      # PDF の 1 ページ（Gemini の文書入力の目安）
TOKENS_PER_PAGE_HIGH = 1000  # 本文の文字も数えられた場合の上限の目安
THINKING_MAX = 32768       # 2.5 Pro の思考の予算の上限
CHARS_PER_TOKEN = 3.0      # LaTeX を含む JSON の目安
PAGES_UNKNOWN = 30         # ページ数が分からない PDF の見積もりに使う値

MATCH_THRESHOLD = 0.45  # 同じ PDF の 2 回の抽出で確かめた（0.45〜0.5 で足される 4 組はすべて正しい対応、0.40〜0.45 は 3 組中 2 組が誤り）
W_TEX, W_VAR, W_CTX = 0.5, 0.35, 0.15


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------

LATEX_KEY_RE = re.compile(r"\\left|\\right|\\,|\\;|\\!|\\quad|\s+")
# 表記の細部：書体の命令、掛け算の記号、波括弧。同じ記号の書き方の違いとみなす
COSMETIC_RE = re.compile(r"\\displaystyle|\\mathrm|\\textrm|\\text|\\operatorname|\\mathit|\\boldsymbol|\\mathbf"
                         r"|\\bm|\\vec|\\rm|\\mathcal|\\cdot|\\times|[{}]")
# Unicode の文字を LaTeX の命令にそろえる（同じ PDF でも抽出ごとに ρ と \rho のように揺れる）
_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon", "ǫ": "epsilon",
          "ζ": "zeta", "η": "eta", "θ": "theta", "ϑ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
          "µ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "ϱ": "rho", "σ": "sigma", "ς": "sigma",
          "τ": "tau", "υ": "upsilon", "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
          "Γ": "Gamma", "Δ": "Delta", "∆": "Delta", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi", "Π": "Pi",
          "Σ": "Sigma", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega", "∂": "partial", "∇": "nabla",
          "∞": "infty", "∑": "sum", "∫": "int", "√": "sqrt", "≡": "equiv", "≈": "approx", "≤": "leq",
          "≥": "geq", "≠": "neq", "∝": "propto", "→": "rightarrow", "ℓ": "ell", "ħ": "hbar"}
_UNI = {k: "\\" + v + " " for k, v in _GREEK.items()}
_UNI.update({"−": "-", "–": "-", "·": "", "×": "", "′": "'", "²": "^2", "³": "^3", "¹": "^1",
             **{chr(0x2080 + d): f"_{d}" for d in range(10)}})
_VAR_CMD_RE = re.compile(r"\\var(epsilon|phi|theta|rho|sigma)")


def unicode_to_latex(s):
    s = unicodedata.normalize("NFC", s or "")
    s = "".join(_UNI.get(ch, ch) for ch in s)
    return _VAR_CMD_RE.sub(lambda m: "\\" + m.group(1), s)


def latex_key(s):
    """set_aware_reranker.latex_key と同じ定義（torch を読み込まないよう写した）。"""
    return LATEX_KEY_RE.sub("", s or "")


def cosmetic_key(s):
    return COSMETIC_RE.sub("", latex_key(unicode_to_latex(s)))


_FUNC_ARG_RE = re.compile(r"(?<=.)\([^()]*\)$")


def sym_key(s):
    """変数記号の表記の揺れ（Unicode と LaTeX、太字、波括弧、空白、末尾の関数の引数 \\Phi(\\xi)→\\Phi）をそろえた記号。"""
    return _FUNC_ARG_RE.sub("", cosmetic_key(s))


def P(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def family(v):
    """kato1021_counts.family と同じ。"""
    return ("単一出典" if v == "original" else "出典横断" if v.startswith("multisource")
            else "合成" if v.startswith("dae_") else None)


def load_db():
    eqs = json.load(open(ROOT / "unified_equations.json", encoding="utf-8"))
    return eqs, {f"{e['source_id']}__{e['eq_id']}": e for e in eqs}


def load_setting_a():
    ex = set(json.load(open(EXCLUDE, encoding="utf-8"))["excluded_case_ids"])
    return [c for c in json.load(open(ROOT / "training_cases.json", encoding="utf-8"))
            if family(c["variant_type"]) and c["case_id"] not in ex]


def requirements(I, O, var_sets):
    """要件 1〜3（kato1021_counts.counts と同じ定義）。var_sets は正解式ごとの変数記号の集合。"""
    allv = set().union(*var_sets) if var_sets else set()
    occ = Counter(v for vs in var_sets for v in vs)
    inter = allv - (I | O)
    return {"req1": (I | O) <= allv, "req2": len(allv - I) == len(var_sets),
            "req3": all(occ[v] >= 2 for v in inter), "missing_io": sorted((I | O) - allv),
            "n_unknown": len(allv - I), "n_eq": len(var_sets)}


def ok(r):
    return r["req1"] and r["req2"] and r["req3"]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------

def gt_9(c, db):
    return [r for r in P(c["correct_model_ids"]) if db[r]["source_id"] in BY_ID]


def cmd_facts(args):
    eqs, db = load_db()
    cases = load_setting_a()
    hit = [c for c in cases if gt_9(c, db)]
    gt_ids = {r for c in hit for r in gt_9(c, db)}
    per = []
    for s in SOURCES:
        sid = s["source_id"]
        mine = [e for e in eqs if e["source_id"] == sid]
        ks = [c for c in hit if any(db[r]["source_id"] == sid for r in gt_9(c, db))]
        per.append({
            "source_id": sid, "availability": s["availability"], "n_equations": len(mine),
            "n_gt_equations": sum(1 for r in gt_ids if db[r]["source_id"] == sid),
            "n_cases": len(ks), "cases_by_family": dict(Counter(family(c["variant_type"]) for c in ks)),
            # fix_dataset_quality.py（2026-03-30）が規則で書き足した s・t（320 報には当てていない）
            "n_equations_with_rule_s_t": sum(1 for e in mine if e["variables"].get("s") ==
                                             "Laplace variable (complex frequency domain)"
                                             or e["variables"].get("t") == "Time variable"),
            "old_output_chars": len(json.dumps(mine, ensure_ascii=False)),
        })
    groups = {"on_hand": {"on_hand"}, "open": {"open", "open_browser"}, "university": {"university"}}
    by_group = {}
    for g, av in groups.items():
        srcs = {s["source_id"] for s in SOURCES if s["availability"] in av}
        touch = [c for c in hit if any(db[r]["source_id"] in srcs for r in gt_9(c, db))]
        by_group[g] = {"sources": sorted(srcs), "n_cases_touching": len(touch)}
    # PDF がそろう段階ごとに、9 報の正解式をすべて抽出し直せるケースの数
    stages = [("on_hand", {"on_hand"}), ("on_hand+open", {"on_hand", "open", "open_browser"}),
              ("all", {"on_hand", "open", "open_browser", "university"})]
    reach = {}
    for name, av in stages:
        srcs = {s["source_id"] for s in SOURCES if s["availability"] in av}
        reach[name] = sum(1 for c in hit if all(db[r]["source_id"] in srcs for r in gt_9(c, db)))
    multi = [c["case_id"] for c in hit if len({db[r]["source_id"] for r in gt_9(c, db)}) > 1]
    facts = {
        "made": dt.date.today().isoformat(),
        "n_setting_a": len(cases), "n_equations_9": sum(p["n_equations"] for p in per),
        "n_db_equations": len(eqs), "n_gt_equations_9": len(gt_ids), "n_cases_9": len(hit),
        "cases_by_family": dict(Counter(family(c["variant_type"]) for c in hit)),
        "cases_by_variant": dict(Counter(c["variant_type"] if not c["variant_type"].startswith("dae_") else "dae"
                                         for c in hit)),
        "cases_with_two_or_more_of_9": multi,
        "per_source": per, "by_availability": by_group, "cases_fully_reextractable": reach,
    }
    dump(OUT / "facts.json", facts)
    rows = []
    for c in hit:
        R = P(c["correct_model_ids"])
        rows.append({"case_id": c["case_id"], "family": family(c["variant_type"]), "variant": c["variant_type"],
                     "n_gt": len(R), "gt_9": " | ".join(gt_9(c, db)),
                     "sources_9": " | ".join(sorted({db[r]["source_id"] for r in gt_9(c, db)})),
                     "gt_other": " | ".join(r for r in R if db[r]["source_id"] not in BY_ID)})
    write_csv(OUT / "cases_146.csv", rows, ["case_id", "family", "variant", "n_gt", "sources_9", "gt_9", "gt_other"])
    print(json.dumps({k: v for k, v in facts.items() if k != "per_source"}, ensure_ascii=False, indent=1))
    for p in per:
        print(p)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def find_pdfs(pdf_dir):
    found = {}
    for p in sorted(Path(pdf_dir).rglob("*.pdf")):
        if p.name in BY_ID:
            found.setdefault(p.name, p)
    return found


def pdf_pages(path):
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def estimate(sid, pages, eqs):
    old_chars = len(json.dumps([e for e in eqs if e["source_id"] == sid], ensure_ascii=False))
    out_json = old_chars / CHARS_PER_TOKEN
    pg = pages or PAGES_UNKNOWN
    lo = (pg * TOKENS_PER_PAGE + 600) * PRICE_IN + out_json * PRICE_OUT
    hi = (pg * TOKENS_PER_PAGE_HIGH + 600) * PRICE_IN + (2 * out_json + THINKING_MAX) * PRICE_OUT
    return {"pages": pg, "pages_known": pages is not None, "out_tokens_est": round(out_json),
            "usd_low": round(lo, 3), "usd_high": round(hi, 3)}


def cmd_extract(args):
    eqs, _ = load_db()
    found = find_pdfs(args.pdf_dir) if args.pdf_dir else {}
    targets = [s for s in S9 if (not args.only or s in args.only)]
    rows, tot_lo, tot_hi = [], 0.0, 0.0
    for sid in targets:
        p = found.get(sid)
        pages = pdf_pages(p) if p else BY_ID[sid]["pages"]
        e = estimate(sid, pages, eqs)
        rows.append({"source_id": sid, "pdf": str(p) if p else None, **e})
        if p:
            tot_lo += e["usd_low"]
            tot_hi += e["usd_high"]
    print(f"{'source_id':38s} {'PDF':4s} {'pages':>5s} {'USD 下限':>9s} {'USD 上限':>9s}")
    for r in rows:
        print(f"{r['source_id'][:38]:38s} {'あり' if r['pdf'] else 'なし':4s} {r['pages']:5d} "
              f"{r['usd_low']:9.3f} {r['usd_high']:9.3f}")
    n_avail = sum(1 for r in rows if r["pdf"])
    print(f"PDF のある {n_avail} 報の見積もり：{tot_lo:.2f}〜{tot_hi:.2f} USD（1 回ずつ。失敗して呼び直すと増える）")
    if not args.run:
        print("見積もりだけ（Gemini は呼んでいない）。抽出するには --run を付ける。")
        return

    import google.genai
    import google.genai.models as gm
    import extract_equations as ee

    captured = {}
    orig = gm.Models.generate_content

    def wrapped(self, *a, **kw):  # 応答を横から記録するだけ。呼び方は extract_equations のまま
        resp = orig(self, *a, **kw)
        captured["resp"] = resp
        captured["model_requested"] = kw.get("model")
        return resp

    gm.Models.generate_content = wrapped
    raw = OUT / args.raw_subdir
    raw.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if not r["pdf"]:
            continue
        sid = r["source_id"]
        out_json, out_meta = raw / f"{sid}.json", raw / f"{sid}.meta.json"
        if out_json.exists() and not args.force:
            print(f"skip（出力あり）: {sid}")
            continue
        p = Path(r["pdf"])
        captured.clear()
        t0 = dt.datetime.now().astimezone()
        print(f"抽出: {sid}", flush=True)
        meta = {"source_id": sid, "pdf_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "pdf_bytes": p.stat().st_size, "pages": r["pages"], "started": t0.isoformat(timespec="seconds"),
                "script": "extract_equations.extract_equations", "google_genai": google.genai.__version__,
                "prompt_sha256": hashlib.sha256(ee.EXTRACTION_PROMPT.encode()).hexdigest()}
        try:
            got = ee.extract_equations(p)
        except Exception as ex:  # noqa: BLE001
            meta.update({"error": f"{type(ex).__name__}: {str(ex)[:500]}"})
            if captured.get("resp") is not None:  # 応答が返ってから失敗した（課金済み）。使った量と本文を残す
                meta.update(_response_meta(captured))
                (raw / f"{sid}.error_response.txt").write_text(_response_text(captured["resp"]), encoding="utf-8")
            dump(raw / f"{sid}.error.json", meta)
            print(f"  失敗: {meta['error']}", flush=True)
            continue
        data = [g.model_dump() for g in got]
        meta.update({"finished": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                     **_response_meta(captured), "n_equations": len(data),
                     "returned_source_ids": dict(Counter(d["source_id"] for d in data))})
        (raw / f"{sid}.response.txt").write_text(_response_text(captured["resp"]), encoding="utf-8")
        dump(out_json, data)
        dump(out_meta, meta)
        flag = "" if meta["model_ok"] else "  ※ 応答のモデルが gemini-2.5-pro ではない"
        print(f"  {len(data)} 式、model_version={meta['model_version']}、{meta['usage']}、"
              f"約 {meta['usd']:.3f} USD{flag}", flush=True)


def _response_text(resp):
    try:
        return resp.text or ""
    except Exception:  # noqa: BLE001
        return ""


def _response_meta(captured):
    """応答から、モデルの版・トークン数・費用（USD）を取り出す。"""
    resp = captured.get("resp")
    um = getattr(resp, "usage_metadata", None)
    usage = {k: getattr(um, k, None) for k in ("prompt_token_count", "candidates_token_count",
                                               "thoughts_token_count", "total_token_count")} if um else {}
    model_version = getattr(resp, "model_version", None)
    cost = ((usage.get("prompt_token_count") or 0) * PRICE_IN
            + ((usage.get("candidates_token_count") or 0) + (usage.get("thoughts_token_count") or 0)) * PRICE_OUT)
    return {"model_requested": captured.get("model_requested"), "model_version": model_version,
            "model_ok": bool(model_version) and str(model_version).startswith("gemini-2.5-pro"),
            "usage": usage, "usd": round(cost, 4)}


# ---------------------------------------------------------------------------
# 対応づけ
# ---------------------------------------------------------------------------

def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def match_equations(A, B):
    """A（古い）と B（新しい）の式を 1 対 1 に対応づける。戻り値は (pairs, only_a, only_b, best)。
    pairs の要素は (i, j, category, score, s_tex, s_var, s_ctx)。best は対応のない式の、相手側で最も近い式と近さ
    （("a", i) -> (j, score)、("b", j) -> (i, score)。閾値で落とした本当の対応がないかを確かめるため）。"""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    k0a, k0b = [latex_key(e["equation"]) for e in A], [latex_key(e["equation"]) for e in B]
    k1a, k1b = [cosmetic_key(e["equation"]) for e in A], [cosmetic_key(e["equation"]) for e in B]
    va = [{sym_key(v) for v in (e.get("variables") or {})} for e in A]
    vb = [{sym_key(v) for v in (e.get("variables") or {})} for e in B]
    ca = [(e.get("context_text") or "").lower()[:400] for e in A]
    cb = [(e.get("context_text") or "").lower()[:400] for e in B]
    S = np.zeros((len(A), len(B)))
    parts = {}
    for i in range(len(A)):
        for j in range(len(B)):
            st = SequenceMatcher(None, k1a[i], k1b[j], autojunk=False).ratio()
            sv = _jaccard(va[i], vb[j])
            sc = SequenceMatcher(None, ca[i], cb[j], autojunk=False).ratio()
            S[i, j] = W_TEX * st + W_VAR * sv + W_CTX * sc
            parts[(i, j)] = (st, sv, sc)
    used_a, used_b, pairs = set(), set(), []
    for keys_a, keys_b, cat in ((k0a, k0b, "same"), (k1a, k1b, "same_cosmetic")):
        idx_b = defaultdict(list)
        for j, k in enumerate(keys_b):
            if j not in used_b:
                idx_b[k].append(j)
        for i, k in enumerate(keys_a):
            if i in used_a or not idx_b.get(k):
                continue
            j = idx_b[k].pop(0)
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j, cat, 1.0, 1.0, _jaccard(va[i], vb[j]), None))
    ra = [i for i in range(len(A)) if i not in used_a]
    rb = [j for j in range(len(B)) if j not in used_b]
    if ra and rb:
        sub = S[np.ix_(ra, rb)]
        rows_, cols_ = linear_sum_assignment(-sub)
        for x, y in zip(rows_, cols_):
            if sub[x, y] < MATCH_THRESHOLD:
                continue
            i, j = ra[x], rb[y]
            used_a.add(i)
            used_b.add(j)
            st, sv, sc = parts[(i, j)]
            cat = "same_symbols" if va[i] == vb[j] else "diff_symbols"
            pairs.append((i, j, cat, float(S[i, j]), st, sv, sc))
    only_a = [i for i in range(len(A)) if i not in used_a]
    only_b = [j for j in range(len(B)) if j not in used_b]
    best = {}
    for i in only_a:
        if len(B):
            j = int(S[i].argmax())
            best[("a", i)] = (j, float(S[i, j]))
    for j in only_b:
        if len(A):
            i = int(S[:, j].argmax())
            best[("b", j)] = (i, float(S[i, j]))
    return pairs, only_a, only_b, best


CAT_JA = {"same": "同じ式", "same_cosmetic": "同じ式（表記の細部だけ違う）",
          "same_symbols": "記号は同じで書き方が違う式", "diff_symbols": "記号の違う式",
          "old_only": "古い抽出にだけある式", "new_only": "新しい抽出にだけある式"}


def table_rows(sid, A, B, pairs, only_a, only_b, gt_use=None, best=None):
    gt_use = gt_use or {}
    best = best or {}
    rows = []

    def syms(e):
        return " | ".join(sorted((e.get("variables") or {}).keys()))

    def nsyms(e):
        return {sym_key(v) for v in (e.get("variables") or {})}

    for i, j, cat, score, st, sv, sc in sorted(pairs, key=lambda p: p[0]):
        a, b = A[i], B[j]
        rows.append({"source_id": sid, "category": cat, "category_ja": CAT_JA[cat],
                     "old_eq_id": a["eq_id"], "new_eq_id": b["eq_id"], "score": round(score, 3),
                     "s_tex": round(st, 3), "s_var": round(sv, 3), "s_ctx": None if sc is None else round(sc, 3),
                     "old_gt_cases": gt_use.get(f"{sid}__{a['eq_id']}", 0),
                     "old_equation": a["equation"], "new_equation": b["equation"],
                     "old_symbols": syms(a), "new_symbols": syms(b),
                     "symbols_raw_equal": set((a.get("variables") or {})) == set((b.get("variables") or {})),
                     "symbols_only_old": " | ".join(sorted(nsyms(a) - nsyms(b))),
                     "symbols_only_new": " | ".join(sorted(nsyms(b) - nsyms(a)))})
    for i in only_a:
        a = A[i]
        rows.append({"source_id": sid, "category": "old_only", "category_ja": CAT_JA["old_only"],
                     "old_eq_id": a["eq_id"], "old_gt_cases": gt_use.get(f"{sid}__{a['eq_id']}", 0),
                     "old_equation": a["equation"], "old_symbols": syms(a),
                     "nearest_other": B[best[("a", i)][0]]["eq_id"] if ("a", i) in best else None,
                     "nearest_score": round(best[("a", i)][1], 3) if ("a", i) in best else None})
    for j in only_b:
        b = B[j]
        rows.append({"source_id": sid, "category": "new_only", "category_ja": CAT_JA["new_only"],
                     "new_eq_id": b["eq_id"], "new_equation": b["equation"], "new_symbols": syms(b),
                     "nearest_other": A[best[("b", j)][0]]["eq_id"] if ("b", j) in best else None,
                     "nearest_score": round(best[("b", j)][1], 3) if ("b", j) in best else None})
    return rows


TABLE_FIELDS = ["source_id", "category", "category_ja", "old_eq_id", "new_eq_id", "score", "s_tex", "s_var", "s_ctx",
                "old_gt_cases", "old_equation", "new_equation", "old_symbols", "new_symbols", "symbols_raw_equal",
                "symbols_only_old", "symbols_only_new", "nearest_other", "nearest_score"]


def counts_of(rows):
    c = Counter(r["category"] for r in rows)
    return {k: c.get(k, 0) for k in CAT_JA}


STATUS_JA = {"A": "ID を付け替えるだけで使える", "B": "記号の書き方をそろえれば使える",
             "C": "対応はつくが要件が崩れる（作り直しが要る）", "D": "対応のつかない正解式がある",
             "E": "PDF がなく抽出し直していない"}


def map_case(c, db, target, new_map, no_pdf_sources=frozenset()):
    """ケース c の正解式のうち target（古い式の ID の並び）を、new_map（古い式の ID → 新しい式）で置き換えたときの判定。
    戻り値は (状態 A〜E, 理由, 古い式で要件 1〜3 が成り立っていたか, 置き換えの記録)。"""
    R = P(c["correct_model_ids"])
    I, O = set(P(c["input_variables"])), set(P(c["output_variables"]))
    old_ok = ok(requirements(I, O, [set(db[r]["variables"]) for r in R]))
    no_pdf = [r for r in target if db[r]["source_id"] in no_pdf_sources]
    unmatched = [r for r in target if r not in new_map and r not in no_pdf]
    if no_pdf:
        return "E", "PDF がない: " + " | ".join(sorted({db[r]["source_id"] for r in no_pdf})), old_ok, None
    if unmatched:
        return "D", "対応のない正解式: " + " | ".join(unmatched), old_ok, None
    tset = set(target)
    vs = [set((new_map[r] if r in tset else db[r])["variables"]) for r in R]
    mapped = " | ".join(f"{r}->{new_map[r]['eq_id']}" for r in target)
    new = requirements(I, O, vs)
    if ok(new):
        return "A", "", old_ok, mapped
    norm = requirements({sym_key(v) for v in I}, {sym_key(v) for v in O}, [{sym_key(v) for v in s} for s in vs])
    if ok(norm):
        return "B", "記号の書き方をそろえれば成り立つ（そのままでは欠ける記号: " + " | ".join(new["missing_io"]) + "）", \
            old_ok, mapped
    phi = find_relabel(R, target, db, new_map)
    if phi is not None and ok(requirements({phi.get(v, v) for v in I}, {phi.get(v, v) for v in O}, vs)):
        io = [f"{v}→{phi[v]}" for v in sorted(I | O) if v in phi]
        return "B", "記号を 1 対 1 に付け替えれば成り立つ（入出力の付け替え: " + " | ".join(io) + "）", old_ok, mapped
    bad = [k for k in ("req1", "req2", "req3") if not norm[k]]
    grew = [f"{db[r]['eq_id']} {len(db[r]['variables'])}→{len(new_map[r]['variables'])}"
            for r in target if len(db[r]["variables"]) != len(new_map[r]["variables"])]
    cause = ("式の変数の数が変わる: " + "、".join(grew)) if grew else "変数の数は同じだが、式どうしのつながりが変わる"
    return "C", ("要件 " + ",".join(bad) + " が崩れる（" + cause + "。欠ける入出力: " + " | ".join(norm["missing_io"])
                 + f"、未知数 {norm['n_unknown']}／式 {norm['n_eq']}）"), old_ok, mapped


def _name_sim(a, b):
    if a == b:
        return 2.0
    ka, kb = sym_key(a).lower(), sym_key(b).lower()
    return 1.5 if ka == kb else SequenceMatcher(None, ka, kb).ratio()


def find_relabel(R, target, db, new_map):
    """正解式のどの式にどの変数が現れるか（変数と式のつながり）を保ったまま、古い記号を新しい記号へ 1 対 1 に付け替え
    られるかを調べる。付け替えられれば {古い記号: 新しい記号}（名前の変わるものだけ）を、だめなら None を返す。
    記号を「現れる正解式の集合」でまとめ、まとまりごとの記号の数が古い抽出と新しい抽出で同じなら付け替えられる。
    9 報以外の正解式は変わらないので、そこに現れる記号は、つながりが同じなら自分自身に対応する。
    まとまりの中に記号が複数あるときは、名前の近いものどうしを組にする（どう組んでもつながりは同じ）。"""
    from scipy.optimize import linear_sum_assignment
    tset = set(target)
    old = {r: set(db[r]["variables"]) for r in R}
    new = {r: set((new_map[r] if r in tset else db[r])["variables"]) for r in R}
    if any(len(old[r]) != len(new[r]) for r in R):
        return None

    def groups(sets):
        sig = defaultdict(set)
        for r, vs in sets.items():
            for v in vs:
                sig[v].add(r)
        g = defaultdict(list)
        for v, s in sig.items():
            g[frozenset(s)].append(v)
        return g

    go, gn = groups(old), groups(new)
    if {k: len(v) for k, v in go.items()} != {k: len(v) for k, v in gn.items()}:
        return None
    phi = {}
    for k, olds in go.items():
        olds, news = sorted(olds), sorted(gn[k])
        ri, ci = linear_sum_assignment([[-_name_sim(a, b) for b in news] for a in olds])
        phi.update({olds[i]: news[j] for i, j in zip(ri, ci)})
    return {a: b for a, b in phi.items() if a != b}


def cmd_match(args):
    eqs, db = load_db()
    cases = load_setting_a()
    hit = [c for c in cases if gt_9(c, db)]
    gt_use = Counter(r for c in hit for r in gt_9(c, db))
    raw = OUT / args.raw_subdir
    new_map, rows, per_source = {}, [], {}
    for sid in S9:
        f = raw / f"{sid}.json"
        if not f.exists():
            continue
        meta_f = raw / f"{sid}.meta.json"
        meta = json.load(open(meta_f, encoding="utf-8")) if meta_f.exists() else {}
        A = [e for e in eqs if e["source_id"] == sid]
        B = json.load(open(f, encoding="utf-8"))
        pairs, only_a, only_b, best = match_equations(A, B)
        rs = table_rows(sid, A, B, pairs, only_a, only_b, gt_use, best)
        rows += rs
        per_source[sid] = {"n_old": len(A), "n_new": len(B), **counts_of(rs),
                           "model_version": meta.get("model_version"), "model_ok": meta.get("model_ok")}
        for i, j, *_ in pairs:
            new_map[f"{sid}__{A[i]['eq_id']}"] = B[j]
    write_csv(OUT / f"match_table_{args.raw_subdir}.csv", rows, TABLE_FIELDS)

    done = {sid for sid in S9 if (raw / f"{sid}.json").exists()}
    case_rows, status = [], Counter()
    for c in hit:
        st, why, old_ok, mapped = map_case(c, db, gt_9(c, db), new_map, set(S9) - done)
        row = {"case_id": c["case_id"], "family": family(c["variant_type"]), "variant": c["variant_type"],
               "n_gt": len(P(c["correct_model_ids"])), "gt_9": " | ".join(gt_9(c, db)), "old_ok": old_ok,
               "status": st, "why": why, "new_map": mapped}
        status[(st, row["family"])] += 1
        case_rows.append(row)
    write_csv(OUT / f"case_mapping_{args.raw_subdir}.csv", case_rows,
              ["case_id", "family", "variant", "n_gt", "status", "why", "old_ok", "gt_9", "new_map"])
    summary = {"made": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               "raw_subdir": args.raw_subdir, "match_threshold": MATCH_THRESHOLD,
               "weights": {"tex": W_TEX, "var": W_VAR, "ctx": W_CTX},
               "per_source": per_source, "equations_total": counts_of(rows),
               "cases_by_status": dict(Counter(r["status"] for r in case_rows)),
               "cases_by_status_family": {f"{k[0]}:{k[1]}": v for k, v in sorted(status.items())},
               "cases_old_not_ok": [r["case_id"] for r in case_rows if not r["old_ok"]]}
    dump(OUT / f"match_summary_{args.raw_subdir}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------
# baseline（同じ PDF を 2.5 Pro で 2 回抽出した組）
# ---------------------------------------------------------------------------

def cmd_baseline(args):
    eqs, _ = load_db()
    by_src = defaultdict(list)
    for e in eqs:
        by_src[e["source_id"]].append(e)
    dup = list(csv.DictReader(open(ROOT / "docs" / "appendix_materials" / "duplicate_works.csv",
                                   encoding="utf-8-sig")))
    files = {p.name: p for p in (ROOT / "pdfs").rglob("*.pdf")}
    works = defaultdict(list)
    for r in dup:
        if r["source_id"] in files and r["source_id"] in by_src:
            works[r["openalex_id"]].append(r["source_id"])
    _, db = load_db()
    cases = load_setting_a()
    rows, pair_rows, case_rows = [], [], []
    for wid, srcs in works.items():
        h = defaultdict(list)
        for s in srcs:
            h[hashlib.sha256(files[s].read_bytes()).hexdigest()].append(s)
        for same in h.values():
            for s1, s2 in combinations(sorted(same), 2):
                A, B = by_src[s1], by_src[s2]
                pairs, only_a, only_b, best = match_equations(A, B)
                rs = table_rows(f"{s1} | {s2}", A, B, pairs, only_a, only_b, best=best)
                rows += rs
                c = counts_of(rs)
                pair_rows.append({"openalex_id": wid, "source_a": s1, "source_b": s2, "n_a": len(A), "n_b": len(B),
                                  **c, "same_share_a": round((c["same"] + c["same_cosmetic"]) / len(A), 3)})
                # 片方を「古い抽出」、もう片方を「抽出し直した結果」とみなし、古い側の式を正解に含むケースを判定する（両向き）
                for s_old, s_new, mp in ((s1, s2, {f"{s1}__{A[i]['eq_id']}": B[j] for i, j, *_ in pairs}),
                                         (s2, s1, {f"{s2}__{B[j]['eq_id']}": A[i] for i, j, *_ in pairs})):
                    for cs in cases:
                        target = [r for r in P(cs["correct_model_ids"]) if db[r]["source_id"] == s_old]
                        if not target:
                            continue
                        st, why, old_ok, _ = map_case(cs, db, target, mp)
                        case_rows.append({"openalex_id": wid, "source_old": s_old, "source_new": s_new,
                                          "case_id": cs["case_id"], "family": family(cs["variant_type"]),
                                          "n_target": len(target), "status": st, "why": why, "old_ok": old_ok})
    write_csv(OUT / "baseline_table.csv", rows, TABLE_FIELDS)
    write_csv(OUT / "baseline_pairs.csv", pair_rows,
              ["openalex_id", "source_a", "source_b", "n_a", "n_b", *CAT_JA.keys(), "same_share_a"])
    tot = counts_of(rows)
    n_a = sum(p["n_a"] for p in pair_rows)
    n_b = sum(p["n_b"] for p in pair_rows)
    summary = {"made": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               "n_works": len({p["openalex_id"] for p in pair_rows}), "n_pairs": len(pair_rows),
               "n_equations_a": n_a, "n_equations_b": n_b, "equations_total": tot,
               "share_of_a_same_or_cosmetic": round((tot["same"] + tot["same_cosmetic"]) / n_a, 3),
               "share_of_a_matched": round((n_a - tot["old_only"]) / n_a, 3),
               "pairs_equal_count": sum(1 for p in pair_rows if p["n_a"] == p["n_b"]),
               "pairs_all_same": sum(1 for p in pair_rows if p["same"] + p["same_cosmetic"] == p["n_a"] == p["n_b"]),
               "match_threshold": MATCH_THRESHOLD,
               # 古い側の式を正解に含む設定 A のケースを、抽出し直した側の式に置き換えたときの判定（単位は 出典の向き×ケース）
               "case_units": len(case_rows), "case_distinct": len({r["case_id"] for r in case_rows}),
               "case_status": dict(sorted(Counter(r["status"] for r in case_rows).items())),
               "case_status_by_family": {f"{k[0]}:{k[1]}": v for k, v in
                                         sorted(Counter((r["status"], r["family"]) for r in case_rows).items())},
               "case_old_not_ok": sum(1 for r in case_rows if not r["old_ok"])}
    # 9 報（4〜42 式）に近い大きさの出典（どちらの抽出も 50 式以下）に絞った集計
    small_pairs = [p for p in pair_rows if p["n_a"] <= 50 and p["n_b"] <= 50]
    size = {**{p["source_a"]: p["n_a"] for p in pair_rows}, **{p["source_b"]: p["n_b"] for p in pair_rows}}
    small_cases = [r for r in case_rows if size[r["source_old"]] <= 50]
    sa = sum(p["n_a"] for p in small_pairs)
    summary["small_sources_le50"] = {
        "n_pairs": len(small_pairs), "n_equations_a": sa,
        "same_or_cosmetic": sum(p["same"] + p["same_cosmetic"] for p in small_pairs),
        "matched": sum(p["n_a"] - p["old_only"] for p in small_pairs),
        "pairs_all_same": sum(1 for p in small_pairs if p["same"] + p["same_cosmetic"] == p["n_a"] == p["n_b"]),
        "case_units": len(small_cases), "case_distinct": len({r["case_id"] for r in small_cases}),
        "case_status": dict(sorted(Counter(r["status"] for r in small_cases).items())),
        "case_status_by_family": {f"{k[0]}:{k[1]}": v for k, v in
                                  sorted(Counter((r["status"], r["family"]) for r in small_cases).items())}}
    write_csv(OUT / "baseline_cases.csv", case_rows,
              ["openalex_id", "source_old", "source_new", "case_id", "family", "n_target", "status", "why", "old_ok"])
    dump(OUT / "baseline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("facts")
    e = sub.add_parser("extract")
    e.add_argument("--pdf-dir", default=str(ROOT / "input_pdfs" / "initial9"))
    e.add_argument("--only", nargs="*", help="この source_id だけ")
    e.add_argument("--run", action="store_true", help="Gemini を呼ぶ（有料）")
    e.add_argument("--force", action="store_true", help="出力があっても抽出し直す")
    e.add_argument("--raw-subdir", default="raw")
    m = sub.add_parser("match")
    m.add_argument("--raw-subdir", default="raw")
    sub.add_parser("baseline")
    args = ap.parse_args()
    {"facts": cmd_facts, "extract": cmd_extract, "match": cmd_match, "baseline": cmd_baseline}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
