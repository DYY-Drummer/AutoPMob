"""文献横断ケース自動生成 v3（加藤先生フィードバック反映版）.

【v2 からの改善】
1. ドメイン適合性チェック：化学工学定番組合せのホワイトリスト
2. 変数の意味整合性チェック：同記号でも定義が異なれば除外
3. 共有変数 2 個以上を必須化（物理的に強く結合）
4. 自然な context 生成（カテゴリ別テンプレート）
5. 微分方程式 + 代数式の組合せを優先（物理モデル感を強化）
6. 物理量＋単位を context に組込み（加藤先生 E3 指摘）

加藤先生方針準拠：
- 用語の意味整合を保証（A4, A5）
- 物理モデル例は微分方程式 + 代数式（E1）
- CSTR・反応器など定番例を含む（E2）
- 数学的可解性 + 物理的妥当性の両立
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent
EQS_JSON = ROOT / "unified_equations.json"
OUT_JSON = ROOT / "multisource_cases_v3.json"


# ============================================================
# ドメインカテゴリ（10 大分類）
# ============================================================

DOMAIN_CATEGORIES = {
    "reactor": [
        # CSTR / PFR / Batch / 反応工学 全般
        "CSTR", "PFR", "Batch Reactor", "Reaction Engineering",
        "Reactor", "Reaction Kinetics", "Catalysis",
        "Reaction Engineering: CSTR", "Reaction Engineering: PFR",
        "Reaction Engineering: Batch Reactors",
        "Stirred-Tank Heating Process", "Blending Process",
        "Polymerization", "Polymer Reactor",
        "Isothermal CSTR", "CSTR Kinetics", "CSTR Modeling",
        "Biodiesel Transesterification",
    ],
    "heat_transfer": [
        "Heat Transfer", "Heat Exchanger", "Conduction", "Convection",
        "Radiation", "Heat Transfer: Conduction", "Heat Transfer: Convection",
        "Heat Transfer: Combined Modes", "Heat Exchanger Dynamics",
        "Double-Pipe Heat Exchanger", "Helium Core Heat Exchanger Modeling",
    ],
    "mass_transfer": [
        "Mass Transfer", "Diffusion", "Absorption",
        "Mass Transfer: Molecular Diffusion", "Mass Transfer: Convective",
        "Distillation", "Extraction",
    ],
    "fluid": [
        "Fluid Dynamics", "Navier-Stokes", "Pipe Flow",
        "Fluid Dynamics: Viscous Flow", "Fluid Dynamics: Turbulent",
        "Fluid Dynamics: Inviscid Flow",
    ],
    "thermo": [
        "Thermodynamics", "Phase Equilibrium", "Equation of State",
        "Thermodynamics: Phase", "Thermodynamics: First Law",
    ],
    "control": [
        "Process Control", "PID Control", "Control Theory",
        "Process Control: PID Control", "Process Control: Process Modeling",
        "State-Space Models", "Laplace Transform",
        "Routh Stability Analysis", "Frequency Response",
        "System Identification", "Analysis of Digital Control Systems",
    ],
    "bioprocess": [
        "Bioprocess Engineering", "Bioprocess Kinetics",
        "Bacterial chemostat", "Fermentation",
        "Bioprocess Engineering: Microbial Kinetics",
    ],
    "electrochem": [
        "Electrochemistry", "Butler-Volmer", "Nernst",
    ],
    "combustion": [
        "Combustion", "Flame Theory", "Flame Propagation",
    ],
    "separation": [
        "Separation", "Distillation", "Absorption", "Extraction",
    ],
}


def categorize_domain(domain_label: str) -> str | None:
    """ドメインラベルをカテゴリに分類"""
    if not domain_label:
        return None
    label_lower = domain_label.lower()
    for cat, keywords in DOMAIN_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in label_lower or label_lower in kw.lower():
                return cat
    return None


# ============================================================
# 自然な組合せのホワイトリスト（化学工学の定番）
# ============================================================

COMPATIBLE_PAIRS = {
    # 反応器 + 熱（非等温反応器）
    ("reactor", "heat_transfer"): {
        "scenario": "non-isothermal reactor with heat transfer",
        "context_template": (
            "Design of a {reactor_type} with simultaneous heat transfer "
            "to a coolant. The mass balance from the reactor kinetics couples "
            "with the heat removal equation through shared variables "
            "{shared_vars}, determining the steady-state concentration and "
            "heat removal rate."
        ),
    },
    # 反応器 + 物質移動（拡散律速反応器）
    ("reactor", "mass_transfer"): {
        "scenario": "diffusion-limited reactor",
        "context_template": (
            "A reactor with diffusion-limited substrate transfer at the "
            "catalyst surface. The reaction rate from the kinetic equation "
            "is coupled with the diffusion flux through shared variables "
            "{shared_vars}, both required to predict the conversion."
        ),
    },
    # 反応器 + プロセス制御（制御付き反応器）
    ("reactor", "control"): {
        "scenario": "feedback-controlled reactor",
        "context_template": (
            "A continuously operated reactor with feedback control on key "
            "process variables {shared_vars}. The reactor model and the "
            "controller transfer function together define the closed-loop "
            "behavior."
        ),
    },
    # 熱交換 + 流体（圧力損失込み）
    ("heat_transfer", "fluid"): {
        "scenario": "heat exchanger with flow dynamics",
        "context_template": (
            "A heat exchanger in which heat transfer and fluid flow are "
            "coupled through shared variables {shared_vars}, requiring "
            "simultaneous solution of the heat balance and the momentum "
            "balance."
        ),
    },
    # 物質移動 + 熱移動（蒸発・凝縮）
    ("mass_transfer", "heat_transfer"): {
        "scenario": "simultaneous heat and mass transfer",
        "context_template": (
            "A unit operation with simultaneous heat and mass transfer "
            "(e.g., evaporator or absorber). Variables {shared_vars} couple "
            "the two transport processes."
        ),
    },
    # バイオプロセス + 反応器
    ("bioprocess", "reactor"): {
        "scenario": "bioreactor with substrate kinetics",
        "context_template": (
            "A bioreactor combining microbial growth kinetics with reactor "
            "design equations. The growth rate and reactor balance share "
            "variables {shared_vars}, both needed to predict steady-state "
            "biomass and substrate concentrations."
        ),
    },
    # バイオプロセス + 物質移動
    ("bioprocess", "mass_transfer"): {
        "scenario": "diffusion-limited bioreactor",
        "context_template": (
            "An immobilized-cell bioreactor where substrate diffusion limits "
            "microbial growth. The Monod kinetics and the diffusion equation "
            "share variables {shared_vars}."
        ),
    },
    # 電気化学 + 物質移動
    ("electrochem", "mass_transfer"): {
        "scenario": "electrochemical cell with diffusion",
        "context_template": (
            "An electrochemical cell with diffusion-limited mass transport "
            "to the electrode. The electrode kinetics and the diffusion flux "
            "share variables {shared_vars}."
        ),
    },
    # 反応器 + 熱力学（状態方程式）
    ("reactor", "thermo"): {
        "scenario": "reactor with phase equilibrium",
        "context_template": (
            "A reactor in which the reaction system is coupled with "
            "thermodynamic phase equilibrium. The reactor balance and the "
            "equation of state share variables {shared_vars}."
        ),
    },
    # 燃焼 + 熱移動
    ("combustion", "heat_transfer"): {
        "scenario": "combustion with heat transfer",
        "context_template": (
            "A combustion process in which heat release from the chemical "
            "reaction couples with heat transfer to the surroundings via "
            "shared variables {shared_vars}."
        ),
    },
    # 燃焼 + 流体（火炎伝播）
    ("combustion", "fluid"): {
        "scenario": "turbulent flame propagation",
        "context_template": (
            "Turbulent flame propagation in which combustion kinetics couples "
            "with fluid flow dynamics through shared variables {shared_vars}."
        ),
    },
    # 分離プロセス + 熱力学（VLE）
    ("separation", "thermo"): {
        "scenario": "separation with phase equilibrium",
        "context_template": (
            "A separation unit (distillation/absorption) in which the stage "
            "balance is coupled with phase equilibrium relations through "
            "shared variables {shared_vars}."
        ),
    },
    # 分離プロセス + 物質移動
    ("separation", "mass_transfer"): {
        "scenario": "separation with mass transfer",
        "context_template": (
            "A separation unit in which the stage balance is coupled with "
            "mass transfer at the interface through shared variables "
            "{shared_vars}."
        ),
    },
    # 制御 + 流体（流量制御）
    ("control", "fluid"): {
        "scenario": "flow control system",
        "context_template": (
            "A flow control system in which the process model and the "
            "controller share variables {shared_vars}."
        ),
    },
    # 制御 + 熱移動（温度制御）
    ("control", "heat_transfer"): {
        "scenario": "temperature control loop",
        "context_template": (
            "A temperature control loop on a thermal process. The heat "
            "transfer equation and the controller transfer function share "
            "variables {shared_vars}."
        ),
    },
}


def is_compatible_pair(cat_a: str, cat_b: str) -> dict | None:
    """カテゴリペアが互換か？ → context テンプレート返す or None"""
    if (cat_a, cat_b) in COMPATIBLE_PAIRS:
        return COMPATIBLE_PAIRS[(cat_a, cat_b)]
    if (cat_b, cat_a) in COMPATIBLE_PAIRS:
        return COMPATIBLE_PAIRS[(cat_b, cat_a)]
    return None


# ============================================================
# 変数意味整合性チェック
# ============================================================

# 同名変数で「異なる物理量」を区別するためのキーワード辞書
INCOMPATIBLE_DEFINITIONS = [
    # 温度 vs 時間定数 vs 透過率
    ("temperature", "time constant"),
    ("temperature", "transmission"),
    ("temperature", "torque"),
    # 圧力 vs 確率
    ("pressure", "probability"),
    # 反応速度 vs 抵抗
    ("rate of reaction", "resistance"),
    # 流量 vs 力
    ("flow rate", "force"),
    ("flow rate", "frequency"),
    # 質量 vs モーメント
    ("mass", "moment"),
    # 角度 vs その他
    ("angle", "concentration"),
]


def vars_semantically_compatible(eq_a: dict, eq_b: dict, shared_vars: set) -> bool:
    """共有変数が両式で同じ物理量を指しているか粗くチェック"""
    vars_a_def = {k: (v or "").lower() for k, v in eq_a.get("variables", {}).items()}
    vars_b_def = {k: (v or "").lower() for k, v in eq_b.get("variables", {}).items()}

    for var in shared_vars:
        def_a = vars_a_def.get(var, "")
        def_b = vars_b_def.get(var, "")
        if not def_a or not def_b:
            continue
        # 不整合キーワードペアに該当するか
        for kw_a, kw_b in INCOMPATIBLE_DEFINITIONS:
            if (kw_a in def_a and kw_b in def_b) or (kw_b in def_a and kw_a in def_b):
                return False
    return True


# ============================================================
# 微分方程式判定（物理モデル感を強化）
# ============================================================

def is_ode_or_pde(eq_text: str) -> bool:
    """微分方程式・偏微分方程式を含むか"""
    if not eq_text:
        return False
    return any(p in eq_text for p in ["frac{d", "\\partial", "\\dot{", "\\nabla", "\\mathrm{d}"])


# ============================================================
# メインアルゴリズム
# ============================================================

def main():
    eqs = json.load(open(EQS_JSON))
    print(f"DB: {len(eqs)} 式")

    # 既存ケース取り込み（重複避け）
    cases = json.load(open(ROOT / "training_cases.json"))
    existing_keys = set()
    for c in cases:
        key = tuple(sorted(c.get("correct_model_ids", [])))
        existing_keys.add(key)
    print(f"既存ケース正解集合: {len(existing_keys)} 通り")

    # 各式のカテゴリ判定
    cat_to_eqs = defaultdict(list)
    for e in eqs:
        cat = categorize_domain(e.get("domain", ""))
        if cat:
            cat_to_eqs[cat].append(e)
    print(f"\n=== カテゴリ別の式数 ===")
    for cat, lst in sorted(cat_to_eqs.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:20s}: {len(lst):4d} 式")

    # 候補ペア探索（互換カテゴリのみ）
    print(f"\n=== 物理的に妥当なペア探索（共有変数 ≥ 2、意味整合あり）===")
    candidates = []
    seen_pairs = set()

    for (cat_a, cat_b), template_info in COMPATIBLE_PAIRS.items():
        eqs_a = cat_to_eqs.get(cat_a, [])
        eqs_b = cat_to_eqs.get(cat_b, [])
        if not eqs_a or not eqs_b:
            continue

        n_found = 0
        for ea in eqs_a:
            vars_a = set(ea.get("variables", {}).keys())
            if not vars_a:
                continue
            for eb in eqs_b:
                if ea["source_id"] == eb["source_id"]:
                    continue  # 同じソースは除外
                vars_b = set(eb.get("variables", {}).keys())
                if not vars_b:
                    continue
                shared = vars_a & vars_b
                if len(shared) < 2:
                    continue  # 共有変数 2 個以上必須

                # 意味整合性チェック
                if not vars_semantically_compatible(ea, eb, shared):
                    continue

                eq_key_a = f"{ea['source_id']}__{ea['eq_id']}"
                eq_key_b = f"{eb['source_id']}__{eb['eq_id']}"
                pair = tuple(sorted([eq_key_a, eq_key_b]))
                if pair in seen_pairs:
                    continue
                if pair in existing_keys:
                    continue
                seen_pairs.add(pair)

                # ODE 含むペアを優先（priority タグ）
                has_ode = is_ode_or_pde(ea.get("equation", "")) or is_ode_or_pde(eb.get("equation", ""))
                priority = 1 if has_ode else 0

                candidates.append({
                    "eq_a": ea, "eq_b": eb,
                    "cat_a": cat_a, "cat_b": cat_b,
                    "shared": shared,
                    "template": template_info,
                    "priority": priority,
                })
                n_found += 1
                if n_found >= 200:  # カテゴリペアごとに上限
                    break
            if n_found >= 200:
                break
        print(f"  {cat_a:15s} + {cat_b:15s}: {n_found:4d} 候補")

    print(f"\n候補ペア総数: {len(candidates)}")

    # 優先度・多様性を考慮してサンプリング
    # ODE 含むペア優先 + ランダム化
    candidates.sort(key=lambda x: (-x["priority"], random.Random(42).random()))
    rng = random.Random(42)
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: -x["priority"])  # ODE 優先で安定ソート

    # ケース構築
    print(f"\n=== ケース構築（DoF=0 構成的に保証）===")
    new_cases = []
    seen_io = set()

    for i, cand in enumerate(candidates):
        if len(new_cases) >= 500:
            break

        ea, eb = cand["eq_a"], cand["eq_b"]
        eq_key_a = f"{ea['source_id']}__{ea['eq_id']}"
        eq_key_b = f"{eb['source_id']}__{eb['eq_id']}"
        eq_keys = [eq_key_a, eq_key_b]

        vars_a = set(ea.get("variables", {}).keys())
        vars_b = set(eb.get("variables", {}).keys())
        all_vars = vars_a | vars_b
        n_eq = 2

        if len(all_vars) < 4:  # 入力 2 + 出力 2 = 最低 4 変数欲しい
            continue

        # 出力選び：各式から 1 個ずつ「独自変数」を選ぶ（強い結合）
        unique_a = vars_a - vars_b
        unique_b = vars_b - vars_a
        if not unique_a or not unique_b:
            # 完全に同じ変数集合なら、共有から 2 個選ぶ
            output_candidates = [(v1, v2) for v1, v2 in combinations(sorted(all_vars), 2)]
        else:
            # 各式からユニーク変数を 1 個ずつ取る
            output_candidates = [(va, vb) for va in unique_a for vb in unique_b]

        for outs in output_candidates[:5]:  # 1 ペアで最大 5 個の I/O 構成
            if len(new_cases) >= 500:
                break
            outputs = set(outs)
            inputs = all_vars - outputs
            if len(inputs) == 0:
                continue
            io_key = (tuple(sorted(eq_keys)), frozenset(inputs), frozenset(outputs))
            if io_key in seen_io:
                continue
            seen_io.add(io_key)

            # context 生成
            shared_vars_display = sorted(cand["shared"])[:3]
            shared_str = ", ".join(f"${v}$" for v in shared_vars_display)
            ctx = cand["template"]["context_template"].format(
                reactor_type="CSTR" if "CSTR" in (ea.get("domain","") + eb.get("domain","")) else "reactor",
                shared_vars=shared_str,
            )

            case = {
                "case_id": f"ms3_{len(new_cases):04d}",
                "original_core_id": f"ms3_{len(new_cases):04d}",
                "variant_type": "multisource_v3",
                "context": ctx,
                "input_variables": sorted(inputs),
                "output_variables": sorted(outputs),
                "correct_model_ids": eq_keys,
            }
            new_cases.append(case)

    print(f"\n生成ケース数: {len(new_cases)}")
    # 内訳：ODE 含む比率
    n_with_ode = sum(1 for i, c in enumerate(new_cases) if i < len(candidates) and candidates[i]["priority"] == 1)
    print(f"  うち ODE 含む: {n_with_ode} ({100*n_with_ode/len(new_cases):.1f}%)" if new_cases else "")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_cases, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
