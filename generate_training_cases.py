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


def _group_by_domain(catalog: List[Dict], max_per_batch: int = 300) -> List[List[Dict]]:
    """カタログをドメインでグループ化し、各バッチが max_per_batch 以下になるよう分割。"""
    from collections import defaultdict
    by_domain: Dict[str, List[Dict]] = defaultdict(list)
    for entry in catalog:
        by_domain[entry["domain"]].append(entry)

    batches: List[List[Dict]] = []
    current_batch: List[Dict] = []
    for domain in sorted(by_domain.keys()):
        entries = by_domain[domain]
        if len(current_batch) + len(entries) > max_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
        current_batch.extend(entries)
    if current_batch:
        batches.append(current_batch)
    return batches


def _generate_core_cases_batch(
    client, batch_catalog: List[Dict], n_cases: int, batch_idx: int
) -> List[CoreCase]:
    """1バッチ分のカタログからコアケースを生成。"""
    import textwrap, time

    catalog_text = json.dumps(batch_catalog, ensure_ascii=False)
    domains = sorted(set(e["domain"] for e in batch_catalog))
    domain_list = ", ".join(domains[:20])
    if len(domains) > 20:
        domain_list += f", ... ({len(domains)} domains total)"

    prompt = textwrap.dedent(
        f"""
        You are helping to create training data for a graph neural network that selects
        relevant equations for physical process modeling.

        You are given a catalog of equations. Each equation has:
        - model_id: a unique ID in the form "<source_id>__<eq_id>"
        - domain: a short domain label
        - variable_symbols: list of variable symbols in the equation

        CATALOG ({len(batch_catalog)} equations, domains: {domain_list}):
        {catalog_text}

        TASK:
        Using ONLY the models listed above, create approximately {n_cases}
        realistic and physically meaningful core training cases.

        For each core case:
        - Pick one or more model_ids that together represent a coherent physical model.
        - Define a context string describing the physical scenario in natural language.
        - Choose input_variables and output_variables from variable_symbols of selected models.
        - correct_model_ids: list of model_ids forming a correct mathematical model.

        CONDITIONS FOR correct_model_ids:
        (1) Given input_variables values and the equations, it MUST be possible to solve for all output_variables.
        (2) The equation set MUST be sufficient and minimal (no redundant equations).

        REQUIREMENTS:
        - All correct_model_ids MUST exactly match entries from the catalog.
        - input/output variables MUST be subsets of variable_symbols from selected models.
        - Make cases diverse across the available domains.
        - Generate at least {n_cases} cases.

        OUTPUT: JSON object with "cases" array, each having:
        case_id, context, input_variables, output_variables, correct_model_ids
        """
    ).strip()

    # リトライ付きAPI呼び出し（503対策）
    import time as _time
    for attempt in range(4):
        _time.sleep(13)  # 5 RPM 対策
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": _get_core_schema(),
                },
            )
            break
        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 30 * (attempt + 1)
                print(f"    503 retry {attempt+1}/3, waiting {wait}s...")
                _time.sleep(wait)
                continue
            raise
    else:
        print(f"    WARNING: All retries failed for batch {batch_idx}")
        return []

    if hasattr(response, "text") and response.text:
        text = response.text.strip()
    elif response.candidates and response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text.strip()
    else:
        print(f"    WARNING: Empty response for batch {batch_idx}")
        return []

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    raw = json.loads(text)
    core_list = CoreCaseList.model_validate(raw)
    return core_list.cases


def generate_core_cases(equations: List[Equation]) -> List[CoreCase]:
    """
    Gemini にカタログをバッチ分割で渡し、コアケースを生成する。
    大規模データセット対応: 300式ずつのバッチに分割し、各バッチから
    均等にケースを生成。
    """
    from google import genai

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")
    client = genai.Client(api_key=api_key)

    catalog, _ = build_model_catalog(equations)

    # カタログサイズに応じて分割
    catalog_json_size = len(json.dumps(catalog, ensure_ascii=False))
    print(f"Catalog: {len(catalog)} equations, {catalog_json_size/1000:.0f}K chars")

    if catalog_json_size < 200_000:
        # 小規模: 1回で全部
        batches = [catalog]
    else:
        # 大規模: ドメイン別に分割
        batches = _group_by_domain(catalog, max_per_batch=300)

    cases_per_batch = max(10, CORE_CASES_TARGET // len(batches))
    print(f"Split into {len(batches)} batches, ~{cases_per_batch} cases each")

    all_cases: List[CoreCase] = []
    for i, batch in enumerate(batches):
        domains = sorted(set(e["domain"] for e in batch))
        print(f"  Batch {i+1}/{len(batches)}: {len(batch)} equations, "
              f"{len(domains)} domains, target {cases_per_batch} cases...")
        try:
            cases = _generate_core_cases_batch(client, batch, cases_per_batch, i + 1)
            # case_id にバッチ番号を付与して衝突回避
            for c in cases:
                c.case_id = f"b{i+1}_{c.case_id}"
            all_cases.extend(cases)
            print(f"    → {len(cases)} cases generated")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"Generated {len(all_cases)} core cases from LLM ({len(batches)} batches).")
    return all_cases


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

