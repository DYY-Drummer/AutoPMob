"""
抽出済み数式JSONに対して、同じ物理意味の変数表記を正規化する。
物理モデル自動生成モデルの訓練用に、表記揺れ（c_w/c_p、ko/k_0 等）を統一する。

使い方:
  python normalize_notation.py extracted_equations.json -o extracted_equations_normalized.json
  python normalize_notation.py extracted_equations.json   # 上書きしない。標準出力先は -o で指定
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# デフォルトの設定パス
DEFAULT_CONFIG = Path(__file__).parent / "config" / "notation_map.json"


def load_notation_map(config_path: Path) -> list[tuple[str, str]]:
    """config の replacements_order を [(from, to), ...] で返す。"""
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    order = data.get("replacements_order", [])
    return [tuple(pair) for pair in order]


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    """文字列に対して、指定順で置換を適用する（LaTeX 内で安全に置換）。"""
    for from_str, to_str in replacements:
        # エスケープが必要な場合は re.escape は使わず、そのまま置換（LaTeX なので）
        text = text.replace(from_str, to_str)
    return text


def normalize_equations_file(
    input_path: Path,
    output_path: Path | None,
    config_path: Path = DEFAULT_CONFIG,
) -> list[dict]:
    """JSON を読み、数式・変数キー・変数説明文に正規化を適用して返す。output_path があれば保存。"""
    with open(input_path, encoding="utf-8") as f:
        equations = json.load(f)

    if not isinstance(equations, list):
        equations = [equations]

    replacements = load_notation_map(config_path)

    for eq in equations:
        # 数式 LaTeX
        if "equation" in eq and eq["equation"]:
            eq["equation"] = apply_replacements(eq["equation"], replacements)
        # 変数辞書: キーと定義文の両方を正規化
        if "variables" in eq and isinstance(eq["variables"], dict):
            new_vars = {}
            for k, v in eq["variables"].items():
                new_key = apply_replacements(k, replacements)
                new_val = apply_replacements(v, replacements) if isinstance(v, str) else v
                new_vars[new_key] = new_val
            eq["variables"] = new_vars
        # context_text も統一したい場合は有効化
        # if "context_text" in eq and eq["context_text"]:
        #     eq["context_text"] = apply_replacements(eq["context_text"], replacements)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(equations, f, ensure_ascii=False, indent=2)
        print(f"Normalized {len(equations)} equations -> {output_path}")

    return equations


def main() -> None:
    parser = argparse.ArgumentParser(description="Extracted equations JSON の変数表記を正規化")
    parser.add_argument("input", type=Path, help="入力 JSON パス（例: extracted_equations.json）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="出力 JSON パス（省略時は入力に _normalized を付与）")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG, help="notation_map.json のパス")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    out = args.output
    if out is None:
        stem = args.input.stem
        out = args.input.parent / f"{stem}_normalized.json"

    normalize_equations_file(args.input, out, args.config)


if __name__ == "__main__":
    main()
