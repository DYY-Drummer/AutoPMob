"""
物理モデル数式グラフ構築用データセット作成パイプライン（統合版）。

- 要件1: input_pdfs/ 内の「未抽出」PDF のみ一括処理（処理済みは processed_pdfs.json で管理、1ファイルごとに 15 秒待機で 5 RPM 対策）
- 要件2: LLM の知識から化学工学基礎式を 20 個生成（Handbook Generation）
- 要件3: 抽出・生成結果を単一の unified_equations.json に統合（既存時は追記、source_id+eq_id で重複排除）

実行例:
  python build_equation_graph.py --mode all
  python build_equation_graph.py --mode pdf
  python build_equation_graph.py --mode handbook
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# 既存モジュールの Equation とスキーマ・抽出関数を利用
from extract_equations import (
    Equation,
    _get_response_schema,
    extract_equations,
    ExtractionError,
    ParseError,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path(__file__).parent / "input_pdfs"
DEFAULT_OUTPUT_FILE = Path(__file__).parent / "unified_equations.json"
PROCESSED_PDFS_FILE = Path(__file__).parent / "processed_pdfs.json"
HANDBOOK_SOURCE_ID = "handbook_chemeng"
SLEEP_BETWEEN_PDF_FILES = 15  # 1ファイル処理ごとに 15 秒（5 RPM 対策）

# 既に抽出済みとして扱う PDF ファイル名（未処理のみ処理するための初期リスト）
INITIALLY_PROCESSED_PDFS = [
    "038.pdf",
    "1-s2.0-0009250974800898-main.pdf",
    "1-s2.0-S1369703X09000059-main.pdf",
    "Chemical Process Dynamics.pdf",
    "Mcgraw-Hill-Process-Modelling-Simulation-And-Control-For-Chemical-Engineers-Re-Distilled.pdf",
    "WJET_2015040715405408.pdf",
    "applsci-10-00992-v2.pdf",
    "energies-17-01344.pdf",
    "kinetics_soybean_oil.pdf",
    "matecconf_cscc2019_01027.pdf",
]

# ---------------------------------------------------------------------------
# Handbook 用プロンプト（化学工学基礎式 20 個を LLM 知識から生成）
# ---------------------------------------------------------------------------

HANDBOOK_PROMPT = """You are an expert in chemical engineering and process modeling.
Generate exactly 20 fundamental equations that are essential for process modeling, in LaTeX format.
Include equations from the following areas:
- CSTR (continuous stirred-tank reactor): mass balance, energy balance, design equation
- Biodiesel production: transesterification kinetics, yield relations
- Mass balance and heat balance (general)
- Reaction kinetics: rate laws, Arrhenius equation, activation energy
- Dimensionless numbers: Reynolds, Nusselt, Prandtl, Damköhler, Peclet, etc.
- Heat transfer: convection, overall heat transfer coefficient
- Fluid mechanics: pressure drop, flow relations

For each equation provide:
- source_id: use exactly "handbook_chemeng"
- eq_id: "eq_1" through "eq_20"
- equation: LaTeX form of the equation
- variables: object with each variable symbol as key and its definition (string) as value
- context_text: brief physical meaning or context (one or two sentences)
- domain: short domain name (e.g. "CSTR", "Reaction Kinetics", "Heat Transfer")

Output a single JSON array of exactly 20 equation objects. Use the exact field names above.
Do not include any markdown or code fence, only the JSON array."""


def _call_gemini_text_only(prompt: str) -> str:
    """テキストのみで Gemini を呼び、応答テキストを返す。5 RPM 用に extract_equations の最終呼び出し時刻を更新する。"""
    from google import genai
    from google.genai.errors import ClientError

    import extract_equations as ex_mod

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or "AIzaSyD5GuIuXVUSeLavZc7_Q0J_muL8Mycwp70"
    )
    client = genai.Client(api_key=api_key)
    min_interval = 12.0  # 5 RPM
    last_time = getattr(ex_mod, "_last_generate_content_time", 0.0)
    now = time.monotonic()
    if last_time > 0 and (now - last_time) < min_interval:
        time.sleep(min_interval - (now - last_time))
    ex_mod._last_generate_content_time = time.monotonic()

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": _get_response_schema(),
        },
    )
    if hasattr(response, "text") and response.text:
        text = response.text.strip()
    elif response.candidates and response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text.strip()
    else:
        raise ParseError("Empty response from the model.")
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    ex_mod._last_generate_content_time = time.monotonic()
    return text


def generate_handbook_equations() -> list[Equation]:
    """LLM の知識から化学工学の基礎式を 20 個生成する。"""
    text = _call_gemini_text_only(HANDBOOK_PROMPT)
    raw_list = json.loads(text)
    if not isinstance(raw_list, list):
        raw_list = [raw_list]
    equations = [Equation.model_validate(item) for item in raw_list]
    for eq in equations:
        if not eq.source_id:
            eq.source_id = HANDBOOK_SOURCE_ID
    return equations


def load_unified(path: Path) -> list[dict]:
    """unified_equations.json が存在すれば読み込み、なければ空リスト。"""
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def load_processed_pdfs(path: Path) -> set[str]:
    """処理済み PDF のファイル名一覧を読み込む。ファイルが無い場合は初期リストで新規作成する。"""
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        names = data if isinstance(data, list) else data.get("processed", [])
        return set(str(n) for n in names)
    processed = set(INITIALLY_PROCESSED_PDFS)
    save_processed_pdfs(path, processed)
    return processed


def save_processed_pdfs(path: Path, processed: set[str]) -> None:
    """処理済み PDF 一覧を保存する。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(processed), f, ensure_ascii=False, indent=2)


def merge_and_dedupe(existing: list[dict], new_equations: list[Equation]) -> list[dict]:
    """既存リストに新規 Equation を追記。source_id + eq_id の重複は追加しない。"""
    keys = {(d["source_id"], d["eq_id"]) for d in existing}
    for eq in new_equations:
        k = (eq.source_id, eq.eq_id)
        if k not in keys:
            existing.append(eq.model_dump())
            keys.add(k)
    return existing


def save_unified(data: list[dict], path: Path) -> None:
    """単一 JSON ファイルに保存。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} equations to {path}")


def run_batch_pdfs(
    input_dir: Path,
    unified_path: Path,
    data: list[dict],
    processed_path: Path,
    processed: set[str],
) -> list[dict]:
    """input_dir 内の「未処理」PDF のみ順に処理し、結果を data にマージ。1ファイルごとに 15 秒待機。"""
    all_pdfs = sorted(input_dir.glob("*.pdf"))
    pdfs = [p for p in all_pdfs if p.name not in processed]
    skipped = len(all_pdfs) - len(pdfs)
    if skipped:
        print(f"Skipping {skipped} already processed PDF(s).", file=sys.stderr)
    if not pdfs:
        print(f"No unprocessed PDFs in {input_dir} (total {len(all_pdfs)} PDFs).", file=sys.stderr)
        return data
    print(f"Processing {len(pdfs)} unprocessed PDF(s)...", file=sys.stderr)
    for i, pdf_path in enumerate(pdfs):
        print(f"  ({i+1}/{len(pdfs)}): {pdf_path.name}", file=sys.stderr)
        try:
            equations = extract_equations(pdf_path)
            data = merge_and_dedupe(data, equations)
            save_unified(data, unified_path)
            processed.add(pdf_path.name)
            save_processed_pdfs(processed_path, processed)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}", file=sys.stderr)
        if i < len(pdfs) - 1:
            print(f"Sleeping {SLEEP_BETWEEN_PDF_FILES}s (5 RPM)...", file=sys.stderr)
            time.sleep(SLEEP_BETWEEN_PDF_FILES)
    return data


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="数式グラフ用データセット作成: PDF一括抽出とHandbook生成を統合"
    )
    parser.add_argument(
        "--mode",
        choices=["pdf", "handbook", "all"],
        default="all",
        help="pdf: PDFのみ, handbook: 基礎式生成のみ, all: 両方（handbook→PDF、5 RPM 対策で 15s 間隔）",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"PDF を検索するフォルダ（既定: {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"統合出力 JSON（既定: {DEFAULT_OUTPUT_FILE}）",
    )
    parser.add_argument(
        "--processed-list",
        type=Path,
        default=PROCESSED_PDFS_FILE,
        help="処理済み PDF 一覧 JSON（既定: processed_pdfs.json）。未抽出の PDF のみ処理する。",
    )
    args = parser.parse_args()

    unified_path = args.output
    processed_path = args.processed_list
    data = load_unified(unified_path)
    if data:
        print(f"Loaded {len(data)} existing equations from {unified_path}", file=sys.stderr)

    if args.mode in ("handbook", "all"):
        print("Generating handbook equations (20) from LLM...", file=sys.stderr)
        try:
            handbook_eqs = generate_handbook_equations()
            data = merge_and_dedupe(data, handbook_eqs)
            save_unified(data, unified_path)
        except Exception as e:
            print(f"Handbook generation failed: {e}", file=sys.stderr)
            if args.mode == "handbook":
                sys.exit(6)
        if args.mode == "all":
            print(f"Sleeping {SLEEP_BETWEEN_PDF_FILES}s before PDF batch...", file=sys.stderr)
            time.sleep(SLEEP_BETWEEN_PDF_FILES)

    if args.mode in ("pdf", "all"):
        if not args.input_dir.is_dir():
            print(f"Input dir not found: {args.input_dir}", file=sys.stderr)
            sys.exit(2)
        processed = load_processed_pdfs(processed_path)
        data = run_batch_pdfs(
            args.input_dir, unified_path, data, processed_path, processed
        )

    print(f"Done. Total equations in {unified_path}: {len(data)}", file=sys.stderr)


if __name__ == "__main__":
    main()
