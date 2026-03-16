"""
科学文献PDFから数式と変数定義を抽出し、JSONで保存するパイプライン（Gemini API + Pydantic）。

【セットアップ】
  pip install google-genai pydantic

【環境変数】
  - APIキー: GEMINI_API_KEY または GOOGLE_API_KEY（未設定時はデフォルトキーを使用）
  - レートリミット時: EXTRACT_EQUATIONS_MAX_RETRIES, EXTRACT_EQUATIONS_BASE_DELAY, EXTRACT_EQUATIONS_MAX_DELAY
  - 5 RPM用: EXTRACT_EQUATIONS_MIN_INTERVAL（既定12秒＝1分あたり5回を超えない）

【実行例】
  python extract_equations.py /path/to/paper.pdf
  出力: extracted_equations.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 出力スキーマ（Pydantic）
# ---------------------------------------------------------------------------


class Equation(BaseModel):
    """抽出された1つの数式を表すモデル。"""

    source_id: str = Field(
        default="",
        description="PDFのファイル名など、出典を識別するID"
    )
    eq_id: str = Field(
        description="数式の通しID（例: eq_1）"
    )
    equation: str = Field(
        description="LaTeX形式の数式"
    )
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="変数記号をキー、その定義（文字列）を値とする辞書"
    )
    context_text: str = Field(
        default="",
        description="数式の物理的意味を説明する周辺テキスト"
    )
    domain: str = Field(
        default="",
        description="プロセスや分野名（例: CSTR Kinetics）"
    )


# Structured Output用に「配列」として返すスキーマを用意する。
# Geminiに「配列」を返させるため、配列のJSONスキーマを渡す。
def _get_response_schema() -> dict:
    eq_schema = Equation.model_json_schema()
    return {
        "type": "array",
        "items": eq_schema,
        "description": "List of extracted equations with variable definitions and context.",
    }


# ---------------------------------------------------------------------------
# プロンプト
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract all numbered equations and equations found in the appendices from this textbook in LaTeX format. For each equation, identify the "variable definitions" and the "physical meaning (context)" from the surrounding text. Output the result in the JSON format specified below.

Constraints:
- Extract the content EXACTLY as it appears in the literature. Do not make unauthorized corrections or modifications.
- Preserve every variable symbol and subscript exactly as printed: e.g. if the source writes c_w (specific heat of water), do NOT change it to c_p; if it writes C_p or Cp, do not change to c_w. Never "normalize" or "standardize" notation.

Output a single JSON array. Each element must have these fields:
- source_id: string (e.g. PDF filename)
- eq_id: string (e.g. "eq_1")
- equation: string (LaTeX form)
- variables: object with variable symbols as keys and their definitions (strings) as values
- context_text: string (surrounding text explaining physical meaning)
- domain: string (process or domain name, e.g. "CSTR Kinetics")
"""


# 5 RPM（1分間に5回）を超えないための最小呼び出し間隔（秒）
_MIN_INTERVAL_FOR_5_RPM = 12  # 60 / 5
_last_generate_content_time: float = 0.0


# ---------------------------------------------------------------------------
# 例外
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """抽出パイプライン用の基底例外。"""
    pass


class RateLimitError(ExtractionError):
    """APIレートリミットに達した場合。"""
    pass


class ParseError(ExtractionError):
    """API応答のJSONパースに失敗した場合。"""
    pass


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------


def extract_equations(pdf_path: str | Path) -> list[Equation]:
    """
    指定PDFをGeminiにアップロードし、数式リストを抽出して返す。

    Args:
        pdf_path: PDFファイルのパス

    Returns:
        抽出されたEquationのリスト

    Raises:
        FileNotFoundError: PDFが存在しない
        RateLimitError: レートリミット
        ParseError: 応答のパース失敗
    """
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or "AIzaSyD5GuIuXVUSeLavZc7_Q0J_muL8Mycwp70"
    )
    client = genai.Client(api_key=api_key)
    source_id = pdf_path.name

    # 1. PDFを1回だけアップロード（リトライは generate_content に集中する）
    uploaded = client.files.upload(file=str(pdf_path))
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise ExtractionError(
            f"File upload failed or not ready: state={uploaded.state.name}"
        )

    prompt_with_source = (
        EXTRACTION_PROMPT
        + f"\n\nUse source_id = \"{source_id}\" for all equations from this document."
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded.uri,
                    mime_type=uploaded.mime_type or "application/pdf",
                ),
                types.Part.from_text(text=prompt_with_source),
            ],
        ),
    ]

    # 2. generate_content のみリトライ（待機は短めに・回数も少なめ）
    # 環境変数で上書き可: EXTRACT_EQUATIONS_MAX_RETRIES, EXTRACT_EQUATIONS_BASE_DELAY, EXTRACT_EQUATIONS_MAX_DELAY
    max_retries = int(os.environ.get("EXTRACT_EQUATIONS_MAX_RETRIES", "5"))
    base_delay = int(os.environ.get("EXTRACT_EQUATIONS_BASE_DELAY", "15"))  # 秒
    max_delay = int(os.environ.get("EXTRACT_EQUATIONS_MAX_DELAY", "60"))  # 1回の待機は最大60秒

    min_interval = float(os.environ.get("EXTRACT_EQUATIONS_MIN_INTERVAL", str(_MIN_INTERVAL_FOR_5_RPM)))

    for attempt in range(max_retries):
        try:
            # 5 RPM を超えないよう、前回呼び出しから min_interval 秒以上あけてから呼ぶ
            global _last_generate_content_time
            now = time.monotonic()
            elapsed = now - _last_generate_content_time
            if elapsed < min_interval and _last_generate_content_time > 0:
                wait = min_interval - elapsed
                print(f"5 RPM 制限: あと {wait:.1f}s 待機してから API を呼びます...", file=sys.stderr)
                time.sleep(wait)
            _last_generate_content_time = time.monotonic()

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": _get_response_schema(),
                },
            )

            # 応答テキスト取得（SDKの違いに備えて .text と candidates の両方に対応）
            if hasattr(response, "text") and response.text:
                text = response.text.strip()
            elif response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text.strip()
            else:
                raise ParseError("Empty response from the model.")
            # マークダウンのコードブロックで囲まれている場合は除去
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            # 5. JSON をパースして Pydantic で検証
            raw_list = json.loads(text)
            if not isinstance(raw_list, list):
                raw_list = [raw_list]
            equations = [Equation.model_validate(item) for item in raw_list]
            # source_id が未設定ならPDFファイル名で統一
            for eq in equations:
                if not eq.source_id:
                    eq.source_id = source_id
            return equations

        except ClientError as e:
            err_str = str(e).lower()
            status = getattr(e, "status_code", None)
            if status is None and e.args and isinstance(e.args[0], int):
                status = e.args[0]
            is_rate_limit = (
                status in (429, 503, 529)
                or "429" in err_str
                or "resource exhausted" in err_str
                or "quota" in err_str
                or "rate limit" in err_str
            )
            if is_rate_limit and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                # Retry-After ヘッダがあれば利用（SDKが公開していれば）
                retry_after = getattr(e, "retry_after", None)
                if isinstance(retry_after, (int, float)) and retry_after > 0:
                    delay = min(int(retry_after), max_delay)
                print(f"Rate limit (status={status}). Retry in {delay}s ({attempt + 1}/{max_retries})...", file=sys.stderr)
                time.sleep(delay)
                continue
            if is_rate_limit:
                raise RateLimitError("Rate limit exceeded after retries.") from e
            raise
        except Exception as e:
            err_msg = str(e).lower()
            is_rate_limit = (
                "429" in err_msg
                or "resource exhausted" in err_msg
                or "quota" in err_msg
                or "rate limit" in err_msg
            )
            if is_rate_limit and attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"Rate limit detected. Retry in {delay}s ({attempt + 1}/{max_retries})...", file=sys.stderr)
                time.sleep(delay)
                continue
            if is_rate_limit:
                raise RateLimitError("Rate limit exceeded after retries.") from e
            if "json" in err_msg or "parse" in err_msg or isinstance(e, json.JSONDecodeError):
                raise ParseError(f"Failed to parse model output as JSON: {e}") from e
            raise


def save_equations(equations: list[Equation], out_path: str | Path = "extracted_equations.json") -> None:
    """EquationのリストをJSONファイルに保存する。"""
    out_path = Path(out_path)
    data = [eq.model_dump() for eq in equations]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(equations)} equations to {out_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="PDF から数式・変数定義を抽出し JSON で保存")
    parser.add_argument("pdf_path", help="入力 PDF のパス")
    parser.add_argument("output_json", nargs="?", default="extracted_equations.json", help="出力 JSON パス（省略時: extracted_equations.json）")
    parser.add_argument("--normalize", action="store_true", help="抽出後に変数表記を正規化し _normalized 付きで保存（訓練用）")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    out_path = args.output_json

    try:
        equations = extract_equations(pdf_path)
        save_equations(equations, out_path)
        if args.normalize:
            from normalize_notation import normalize_equations_file
            from pathlib import Path
            out_p = Path(out_path)
            normalized_path = out_p.parent / f"{out_p.stem}_normalized{out_p.suffix}"
            normalize_equations_file(out_p, normalized_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except RateLimitError as e:
        print(f"Rate limit: {e}", file=sys.stderr)
        sys.exit(3)
    except ParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(4)
    except ExtractionError as e:
        print(f"Extraction error: {e}", file=sys.stderr)
        sys.exit(5)


if __name__ == "__main__":
    main()
