"""
unified_equations.json から GNN 訓練用の多様な訓練ケースを自動生成するスクリプト。

- ステップ1: Gemini (gemini-2.5-pro) による「コアケース」生成（約150件）
- ステップ2: Python によるデータ拡張（入出力スワップ・中間変数予測・context 言い換え）
- 最終的に 900 件以上のケースをシャッフルして training_cases.json に保存する。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

from pydantic import BaseModel, Field

from extract_equations import Equation, _get_response_schema


ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
OUTPUT_JSON = ROOT / "training_cases.json"

CORE_CASES_TARGET = 150


# ---------------------------------------------------------------------------
# Pydantic スキーマ（Structured Output 用）
# ---------------------------------------------------------------------------


class CoreCase(BaseModel):
    case_id: str = Field(description="Unique ID for the core case, e.g., core_001.")
    context: str = Field(description="Textual description of the physical model and scenario.")
    input_variables: List[str] = Field(
        description="List of variable symbols that are considered known inputs/conditions."
    )
    output_variables: List[str] = Field(
        description="List of variable symbols that are target outputs to be predicted."
    )
    correct_model_ids: List[str] = Field(
        description=(
            "List of model IDs that form a correct mathematical model for this case. "
            "Each ID must be of the form '<source_id>__<eq_id>' and MUST refer to an "
            "existing equation in the provided equation list."
        )
    )


class CoreCaseList(BaseModel):
    cases: List[CoreCase] = Field(
        default_factory=list,
        description="List of core training cases.",
    )


def _get_core_schema() -> Dict:
    schema = CoreCaseList.model_json_schema()
    return schema


# ---------------------------------------------------------------------------
# ユーティリティ: unified_equations.json の読み込みとモデル要約の構築
# ---------------------------------------------------------------------------


def load_equations(path: Path) -> List[Equation]:
    if not path.is_file():
        raise FileNotFoundError(f"unified_equations.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        eq_list = raw
    elif isinstance(raw, dict) and "equations" in raw:
        eq_list = raw["equations"]
    else:
        raise ValueError("unified_equations.json has unexpected format.")
    return [Equation.model_validate(e) for e in eq_list]


def build_model_catalog(equations: List[Equation]) -> Tuple[List[Dict], Dict[str, Equation]]:
    """
    Equation のリストから、LLM に渡すための軽量なカタログと、
    model_id -> Equation のマップを作成する。
    model_id は '<source_id>__<eq_id>' 形式。
    """
    catalog: List[Dict] = []
    id_to_eq: Dict[str, Equation] = {}
    for eq in equations:
        model_id = f"{eq.source_id}__{eq.eq_id}"
        id_to_eq[model_id] = eq
        catalog.append(
            {
                "model_id": model_id,
                "source_id": eq.source_id,
                "eq_id": eq.eq_id,
                "domain": eq.domain,
                "variable_symbols": list(eq.variables.keys()),
            }
        )
    return catalog, id_to_eq


# ---------------------------------------------------------------------------
# ステップ1: LLM によるコアケース生成
# ---------------------------------------------------------------------------


def generate_core_cases(equations: List[Equation]) -> List[CoreCase]:
    """
    Gemini (gemini-2.5-pro) に unified_equations のカタログを渡し、
    物理的に意味のあるコアケースを約150件生成してもらう。
    """
    from google import genai

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or "AIzaSyD5GuIuXVUSeLavZc7_Q0J_muL8Mycwp70"
    )
    client = genai.Client(api_key=api_key)

    catalog, _ = build_model_catalog(equations)

    # カタログはそのまま JSON 文字列として渡す（必要なら先頭数百件に制限）
    # サイズが気になる場合は domain ごとにサンプリングするなど拡張余地あり。
    import textwrap

    catalog_text = json.dumps(catalog, ensure_ascii=False)

    prompt = textwrap.dedent(
        f"""
        You are helping to create training data for a graph neural network that selects
        relevant equations for physical process modeling.

        You are given a catalog of equations extracted from chemical engineering and process
        modeling textbooks and papers. Each equation has:
        - model_id: a unique ID in the form "<source_id>__<eq_id>"
        - source_id: the document identifier
        - eq_id: equation identifier within the document
        - domain: a short domain label (e.g., "CSTR", "Heat Transfer", "Reaction Kinetics")
        - variable_symbols: list of variable symbols that appear in the equation

        CATALOG:
        {catalog_text}

        TASK:
        Using ONLY the models listed in the catalog above, create approximately {CORE_CASES_TARGET}
        realistic and physically meaningful core training cases for process modeling.

        For each core case:
        - Pick one or more model_ids from the catalog that together represent a coherent physical model.
          (For example: mass balance + energy balance for a CSTR, or a rate law + Arrhenius equation.)
        - Define a context string that describes the physical scenario in natural language
          (e.g., CSTR temperature control, biodiesel transesterification, etc.).
        - Choose input_variables: variable symbols that would be considered known inputs/conditions.
        - Choose output_variables: variable symbols that the model should predict.
        - correct_model_ids: list of model_ids (exact strings from the catalog) that form a correct
          mathematical model for this case.

        VERY IMPORTANT CONDITIONS FOR correct_model_ids:
        (1) Given numerical values for all input_variables and the equations in correct_model_ids,
            it MUST be possible (in principle) to solve for all output_variables.
        (2) The set of equations in correct_model_ids MUST be sufficient and minimal:
            - If you remove any single equation from correct_model_ids, then it should NO LONGER
              be possible to uniquely solve for all output_variables from the remaining equations
              and input_variables (i.e., no redundant equations).

        REQUIREMENTS:
        - All correct_model_ids MUST exactly match model_id entries from the catalog above.
        - input_variables and output_variables MUST be subsets of the variable_symbols from the
          selected correct_model_ids.
        - Each core case MUST satisfy conditions (1) and (2) above.
        - Make the cases diverse across domains (CSTR, biodiesel, heat transfer, reaction kinetics,
          dimensionless numbers, etc.).
        - Use English for context and variable names exactly as given in the catalog.
        - Generate at least {CORE_CASES_TARGET} cases; you may generate a few more if helpful.

        OUTPUT:
        Return a JSON object that matches the following schema:
        - cases: array of objects with fields:
          - case_id: string (e.g., "core_001")
          - context: string
          - input_variables: array of strings
          - output_variables: array of strings
          - correct_model_ids: array of strings (each "<source_id>__<eq_id>")
        """
    ).strip()

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": _get_core_schema(),
        },
    )

    if hasattr(response, "text") and response.text:
        text = response.text.strip()
    else:
        # フォールバック（candidates 経由）
        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("Empty response from Gemini when generating core cases.")
        text = response.candidates[0].content.parts[0].text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    raw = json.loads(text)
    core_list = CoreCaseList.model_validate(raw)
    cases = core_list.cases
    print(f"Generated {len(cases)} core cases from LLM.")
    return cases


# ---------------------------------------------------------------------------
# ステップ2: Python によるデータ拡張
# ---------------------------------------------------------------------------


@dataclass
class TrainingCase:
    case_id: str
    original_core_id: str
    variant_type: str
    context: str
    input_variables: List[str]
    output_variables: List[str]
    correct_model_ids: List[str]


def collect_variables_from_models(
    core: CoreCase, id_to_eq: Dict[str, Equation]
) -> List[str]:
    """correct_model_ids に含まれる全 Equation から variables.keys() を統合。"""
    vars_set = set()
    for mid in core.correct_model_ids:
        eq = id_to_eq.get(mid)
        if not eq:
            continue
        vars_set.update(eq.variables.keys())
    return sorted(vars_set)


def paraphrase_context_variants(context: str) -> List[str]:
    """
    LLM を使わずに簡単な言い換えテンプレートを適用して 3 パターンの context を生成。
    """
    return [
        context,
        f"This case describes the following physical model: {context}",
        f"In this scenario, the underlying process can be summarized as: {context}",
        f"From the perspective of process modeling, this case focuses on: {context}",
    ]


def augment_core_case(
    core: CoreCase, id_to_eq: Dict[str, Equation], idx: int
) -> List[TrainingCase]:
    """
    1つのコアケースから複数のバリアントを生成する。
    - V1: オリジナル
    - V2: 入出力完全スワップ（可能なら）
    - V3〜: correct_model_ids に含まれる変数集合からランダムに出力・入力を割り当て
    - context は簡単なテンプレートで言い換え
    """
    variants: List[TrainingCase] = []
    base_id = f"core_{idx:03d}"

    # context の言い換えパターンを事前生成
    context_variants = paraphrase_context_variants(core.context)

    # V1: オリジナル
    variants.append(
        TrainingCase(
            case_id=f"{base_id}_v1",
            original_core_id=core.case_id,
            variant_type="original",
            context=context_variants[0],
            input_variables=list(core.input_variables),
            output_variables=list(core.output_variables),
            correct_model_ids=list(core.correct_model_ids),
        )
    )

    # V1 の言い換え: 入出力・正解モデルは同じで、context のみ言い換えたケース（context_paraphrased）
    for para_idx, ctx in enumerate(context_variants[1:4], start=1):  # テンプレート 2, 3, 4 を使用
        variants.append(
            TrainingCase(
                case_id=f"{base_id}_v1_para{para_idx}",
                original_core_id=core.case_id,
                variant_type="context_paraphrased",
                context=ctx,
                input_variables=list(core.input_variables),
                output_variables=list(core.output_variables),
                correct_model_ids=list(core.correct_model_ids),
            )
        )

    # V2: 入出力の完全スワップ（両方非空なら）
    if core.input_variables and core.output_variables:
        variants.append(
            TrainingCase(
                case_id=f"{base_id}_v2",
                original_core_id=core.case_id,
                variant_type="swap_io",
                context=context_variants[1],
                input_variables=list(core.output_variables),
                output_variables=list(core.input_variables),
                correct_model_ids=list(core.correct_model_ids),
            )
        )

    # V3〜: correct_model_ids に含まれる全変数からランダムに出力を選び、残りを入力にする
    all_vars = collect_variables_from_models(core, id_to_eq)
    if not all_vars:
        all_vars = sorted(set(core.input_variables) | set(core.output_variables))

    if all_vars:
        for v_idx in range(3, 7):  # v3..v6
            k_out = random.randint(1, min(2, len(all_vars)))
            outs = sorted(random.sample(all_vars, k_out))
            ins = sorted([v for v in all_vars if v not in outs])
            variants.append(
                TrainingCase(
                    case_id=f"{base_id}_v{v_idx}",
                    original_core_id=core.case_id,
                    variant_type="random_io_from_models",
                    context=context_variants[min(v_idx - 1, len(context_variants) - 1)],
                    input_variables=ins,
                    output_variables=outs,
                    correct_model_ids=list(core.correct_model_ids),
                )
            )

    return variants


def main() -> None:
    # 1. Equation とカタログの読み込み
    equations = load_equations(UNIFIED_JSON)
    _, id_to_eq = build_model_catalog(equations)

    # 2. コアケース生成（LLM）
    core_cases = generate_core_cases(equations)
    if not core_cases:
        raise RuntimeError("No core cases generated from LLM.")

    # 3. データ拡張
    all_variants: List[TrainingCase] = []
    for idx, core in enumerate(core_cases, start=1):
        variants = augment_core_case(core, id_to_eq, idx)
        all_variants.extend(variants)

    # 4. シャッフルして JSON 保存
    random.shuffle(all_variants)
    out = [
        {
            "case_id": tc.case_id,
            "original_core_id": tc.original_core_id,
            "variant_type": tc.variant_type,
            "context": tc.context,
            "input_variables": tc.input_variables,
            "output_variables": tc.output_variables,
            "correct_model_ids": tc.correct_model_ids,
        }
        for tc in all_variants
    ]

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(all_variants)} training cases -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

