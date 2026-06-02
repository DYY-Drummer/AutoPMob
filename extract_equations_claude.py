"""Claude Opus 4.7 で PDF から数式を抽出する（Gemini 版の代替）.

extract_equations.py の Gemini 版と同じスキーマで出力するため、
unified_equations.json への統合パイプラインは完全に互換。

【セットアップ】
  pip install anthropic
  export ANTHROPIC_API_KEY='sk-ant-xxx'

【使い方】
  python extract_equations_claude.py /path/to/paper.pdf
  python extract_equations_claude.py /path/to/paper.pdf --out custom.json

【特徴】
  - Claude Opus 4.7 (1M context) で PDF を直接処理
  - レート制限：通常 50 RPM 以上（プラン依存）
  - PDF サイズ上限：32 MB
  - 出力スキーマ：extract_equations.py（Gemini 版）と完全同一
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# 出力スキーマ（extract_equations.py と完全同一）
# ---------------------------------------------------------------------------

class Equation(BaseModel):
    source_id: str = Field(default="", description="PDFのファイル名など、出典を識別するID")
    eq_id: str = Field(description="数式の通しID（例: eq_1）")
    equation: str = Field(description="LaTeX形式の数式")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="変数記号をキー、その定義（文字列）を値とする辞書",
    )
    context_text: str = Field(default="", description="数式の物理的意味（1-3 文の英語）")
    domain: str = Field(default="", description="分野ラベル")


class EquationList(BaseModel):
    equations: list[Equation]


# ---------------------------------------------------------------------------
# 抽出プロンプト（Gemini 版と完全同一）
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract all numbered equations and equations referenced in the main body
and appendices of this document in LaTeX format.

For each equation, provide:
- source_id: use exactly the source_id given below
- eq_id: a unique identifier (e.g., "eq_1", "eq_2.5")
- equation: the LaTeX-formatted equation
- variables: a dict mapping each variable symbol to its definition (extracted from surrounding text or symbol table)
- context_text: 1-3 sentences describing the physical meaning of the equation, based on its surrounding context
- domain: a concise domain label (e.g., "CSTR", "Heat Transfer: Conduction", "Polymer Reactor")

CONSTRAINTS:
- Extract the content exactly as it appears in the literature. Do not make unauthorized corrections.
- Variables may include subscripts, superscripts, or other marks (hat, dot, bar, etc.).
- Skip equations that are purely numerical examples or single-step substitutions.
- Skip equations from exercise/problem sections.

OUTPUT FORMAT: Respond with valid JSON ONLY (no markdown fences), matching this structure:

{
  "equations": [
    {
      "source_id": "<source_id>",
      "eq_id": "eq_1",
      "equation": "...",
      "variables": {"x": "definition", "y": "definition"},
      "context_text": "...",
      "domain": "..."
    },
    ...
  ]
}
"""


# ---------------------------------------------------------------------------
# 例外
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    pass


class ParseError(ExtractionError):
    pass


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def extract_from_pdf(
    pdf_path: str | Path,
    *,
    model: str = "claude-opus-4-5-20250805",
    api_key: str | None = None,
    max_retries: int = 3,
) -> list[Equation]:
    """PDF を Claude に直接渡して数式を抽出する。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is required")

    # PDF サイズチェック（Anthropic API: 32MB 上限）
    pdf_size = pdf_path.stat().st_size
    if pdf_size > 32 * 1024 * 1024:
        raise ExtractionError(f"PDF too large ({pdf_size/1024/1024:.1f} MB > 32 MB)")

    # PDF を Base64 エンコード
    with pdf_path.open("rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

    source_id = pdf_path.name

    client = anthropic.Anthropic(api_key=api_key)
    prompt_with_source = (
        EXTRACTION_PROMPT
        + f'\n\nUse source_id = "{source_id}" for all equations from this document.'
    )

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt_with_source,
                            },
                        ],
                    }
                ],
            )
            break
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 503, 529):
                wait = 30 * (attempt + 1)
                print(f"  {e.status_code} retry {attempt+1}/{max_retries}, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            print(f"  Unexpected error: {e}")
            raise
    else:
        raise ExtractionError(f"All {max_retries} retries failed")

    # レスポンスからテキスト抽出
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    # マークダウンの ```json ``` がある場合は除去
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # JSON パース
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON parse failed: {e}\nResponse preview: {text[:500]}")

    try:
        parsed = EquationList.model_validate(raw)
    except ValidationError as e:
        raise ParseError(f"Schema validation failed: {e}")

    # source_id を強制（一貫性確保）
    for eq in parsed.equations:
        eq.source_id = source_id

    return parsed.equations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="PDF ファイルのパス")
    ap.add_argument("--out", default="extracted_equations.json", help="出力 JSON パス")
    ap.add_argument("--model", default="claude-opus-4-5-20250805",
                    help="使用する Claude モデル（例: claude-opus-4-5-20250805）")
    args = ap.parse_args()

    print(f"Extracting from: {args.pdf}")
    print(f"Using model: {args.model}")

    try:
        equations = extract_from_pdf(args.pdf, model=args.model)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(equations)} equations.")

    # JSON 出力（extract_equations.py と同じ形式：list[Equation]）
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            [eq.model_dump() for eq in equations],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
