"""DAE（Differential-Algebraic Equation）ケース生成 v1.

【ユーザー指定アルゴリズム（厳密実装）】
0. X ∈ {1, 2, ..., 10}
1. 任意のソースから ODE/PDE を 1 件選び正解集合に加える
2. ODE の LHS から微分された変数を抽出 → output
3. 正解集合の (変数 - output) を含む式を、同ソース優先で探して追加
4. |correct_set| == X まで 3 を繰り返す（探索打ち切りで断念可）
5. inputs = (正解集合の全変数) - outputs
6. 正解集合 + I/O から、Claude API で物理シナリオ説明を生成
7. 各 X について 100 ケース生成

【出力】
  dae_cases.json: 各ケースに variant_type = "dae_X{N}" 付与

【セットアップ】
  pip install anthropic python-dotenv
  ANTHROPIC_API_KEY を .env に設定
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

import anthropic

ROOT = Path(__file__).parent
EQS_JSON = ROOT / "unified_equations.json"
OUT_JSON = ROOT / "dae_cases.json"
REPORT = ROOT / "dae_generation_report.json"

MODEL = "claude-sonnet-4-5-20250929"  # コスト効率重視（OPus でも可）
RATE_LIMIT_INTERVAL = 1.0   # API 呼出間隔（秒）
N_PER_X = 100               # 各 X につき何件生成するか
X_MAX = 10                  # X の最大値
MAX_ATTEMPTS_PER_CASE = 30  # 失敗時のリトライ上限


# ============================================================
# 微分方程式の判定と微分変数の抽出
# ============================================================

def is_ode(eq_text: str) -> bool:
    if not eq_text:
        return False
    return any(p in eq_text for p in [
        "frac{d", "\\partial", "\\dot{", "\\ddot{", "\\nabla",
    ])


def extract_differentiated_vars(eq_text: str, var_names: list[str]) -> list[str]:
    """LHS から微分された変数を抽出"""
    if not eq_text:
        return []
    # LHS のみを対象に（= の左側）
    lhs = eq_text.split("=", 1)[0] if "=" in eq_text else eq_text
    diffvars = []
    for v in var_names:
        v_esc = re.escape(v)
        patterns = [
            # \frac{dY}{dt} or \frac{d Y}{dt}
            rf"\\frac\s*\{{\s*d\s*{v_esc}\s*\}}\s*\{{\s*d[^\}}]+\s*\}}",
            # \frac{d^2 Y}{dt^2}
            rf"\\frac\s*\{{\s*d\^?\d+\s*{v_esc}\s*\}}\s*\{{\s*d[^\}}]+\s*\}}",
            # \dot{Y}, \ddot{Y}
            rf"\\dot\s*\{{\s*{v_esc}\s*\}}",
            rf"\\ddot\s*\{{\s*{v_esc}\s*\}}",
            # \frac{\partial Y}{\partial t}
            rf"\\frac\s*\{{\s*\\partial\s+{v_esc}\s*\}}\s*\{{\s*\\partial[^\}}]+\s*\}}",
            # シンプル: dY/dt（LaTeX なしの簡易形）
            rf"\bd{v_esc}\s*/\s*d[a-zA-Z]+",
        ]
        for p in patterns:
            if re.search(p, lhs):
                diffvars.append(v)
                break
    return diffvars


# ============================================================
# ケース生成コアアルゴリズム
# ============================================================

def find_ode_with_diffvar(eqs: list[dict]) -> list[tuple[dict, list[str]]]:
    """全 ODE 式と、それぞれの微分変数を抽出"""
    odes = []
    for e in eqs:
        eq_text = e.get("equation", "")
        if not is_ode(eq_text):
            continue
        var_names = list(e.get("variables", {}).keys())
        diff_vars = extract_differentiated_vars(eq_text, var_names)
        if diff_vars:
            odes.append((e, diff_vars))
    return odes


def build_correct_set(seed_ode: dict, output_vars: list[str], X: int,
                      eqs_by_src: dict, eqs_all: list[dict],
                      rng: random.Random) -> Optional[list[dict]]:
    """seed_ode から始めて、X 件の正解集合を構築"""
    correct_set = [seed_ode]
    outputs = set(output_vars)
    used_keys = {f"{seed_ode['source_id']}__{seed_ode['eq_id']}"}

    while len(correct_set) < X:
        # 現在の正解集合の (変数 - outputs) を計算
        current_vars = set()
        for e in correct_set:
            current_vars.update(e.get("variables", {}).keys())
        non_output_vars = current_vars - outputs

        if not non_output_vars:
            return None

        # 同ソース優先で式探索
        seed_src = seed_ode["source_id"]
        same_src_candidates = [
            e for e in eqs_by_src.get(seed_src, [])
            if f"{e['source_id']}__{e['eq_id']}" not in used_keys
            and any(v in e.get("variables", {}) for v in non_output_vars)
        ]
        other_src_candidates = [
            e for e in eqs_all
            if e["source_id"] != seed_src
            and f"{e['source_id']}__{e['eq_id']}" not in used_keys
            and any(v in e.get("variables", {}) for v in non_output_vars)
        ]

        # 同ソース優先で選ぶ
        pool = same_src_candidates if same_src_candidates else other_src_candidates
        if not pool:
            return None  # もう追加できる式がない

        next_eq = rng.choice(pool)
        correct_set.append(next_eq)
        used_keys.add(f"{next_eq['source_id']}__{next_eq['eq_id']}")

    return correct_set


# ============================================================
# Claude API で物理シナリオ説明を生成
# ============================================================

def generate_description(client, correct_set: list[dict], inputs: set[str],
                         outputs: set[str], model: str = MODEL) -> Optional[str]:
    """物理シナリオの説明を Claude で生成"""
    # 正解式を整形
    eqs_str = "\n".join([
        f"- [{e['source_id']}/{e['eq_id']}, domain: {e.get('domain','')}]: "
        f"{e.get('equation','')[:200]}"
        for e in correct_set
    ])
    sample_vars = sorted(list(outputs))[:3]
    var_defs = []
    for e in correct_set:
        for k, v in e.get("variables", {}).items():
            if k in outputs or k in list(inputs)[:5]:
                var_defs.append(f"{k}: {v}")
    var_defs_str = "\n".join(var_defs[:10])

    prompt = f"""You are a chemical engineering expert. Given a set of equations from
literature, and specified input/output variables, write a concise physical
scenario description (2-3 sentences) in English describing the modeling task.

EQUATIONS:
{eqs_str}

INPUT VARIABLES (known): {sorted(inputs)[:10]}
OUTPUT VARIABLES (to be solved): {sorted(outputs)}

VARIABLE DEFINITIONS (excerpt):
{var_defs_str}

Write a 2-3 sentence description of the physical modeling scenario. Focus on:
- What system is being modeled (reactor, heat exchanger, etc.)
- What is computed (the outputs) given the inputs
- The physical meaning of the coupled equations

Output only the description, no preamble or formatting."""

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            return text
        except anthropic.RateLimitError:
            time.sleep(30 * (attempt + 1))
        except Exception as e:
            print(f"  Claude error: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(5)
    return None


# ============================================================
# メイン
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-x", type=int, default=N_PER_X, help="各 X につき生成数")
    ap.add_argument("--x-max", type=int, default=X_MAX, help="X の最大値")
    ap.add_argument("--model", default=MODEL, help="Claude モデル名")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="API 呼出なし（説明文ダミー）")
    args = ap.parse_args()

    # API key 確認
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add to .env")
    client = anthropic.Anthropic(api_key=api_key) if not args.dry_run else None

    # DB 読込
    print(f"Loading equations from {EQS_JSON}...")
    eqs = json.load(open(EQS_JSON))
    print(f"  {len(eqs)} 式")

    # ソース別 index
    eqs_by_src = defaultdict(list)
    for e in eqs:
        eqs_by_src[e["source_id"]].append(e)

    # ODE 式と微分変数を事前計算
    print("ODE 式と微分変数を抽出...")
    odes_with_diff = find_ode_with_diffvar(eqs)
    print(f"  ODE: {len(odes_with_diff)} 件、微分変数判定可能")

    # 既存ケース読込（重複避け）
    cases_path = ROOT / "training_cases.json"
    existing = json.load(open(cases_path)) if cases_path.exists() else []
    existing_signatures = set()
    for c in existing:
        sig = tuple(sorted(c.get("correct_model_ids", [])))
        existing_signatures.add(sig)
    print(f"既存ケース正解集合シグネチャ: {len(existing_signatures)}")

    rng = random.Random(args.seed)
    all_dae_cases = []
    report = {"per_X": {}, "model": args.model}
    last_api_time = 0.0

    for X in range(1, args.x_max + 1):
        print(f"\n=== X = {X} (target: {args.n_per_x} ケース) ===")
        x_cases = []
        n_attempted = 0
        n_dof_fail = 0
        n_build_fail = 0
        n_dup = 0
        n_desc_fail = 0

        max_attempts = args.n_per_x * 20
        seen_sigs_this_X = set()

        while len(x_cases) < args.n_per_x and n_attempted < max_attempts:
            n_attempted += 1
            # ステップ 1: 任意の ODE を選ぶ
            seed_ode, diff_vars = rng.choice(odes_with_diff)
            # ステップ 2: 出力 = 微分変数
            output_var = rng.choice(diff_vars)
            outputs = {output_var}
            # ステップ 3-4: X 件まで構築
            correct_set = build_correct_set(
                seed_ode, list(outputs), X, eqs_by_src, eqs, rng
            )
            if not correct_set:
                n_build_fail += 1
                continue

            # 重複チェック
            sig = tuple(sorted(f"{e['source_id']}__{e['eq_id']}" for e in correct_set))
            if sig in existing_signatures or sig in seen_sigs_this_X:
                n_dup += 1
                continue
            seen_sigs_this_X.add(sig)

            # ステップ 5: inputs = (correct vars) - outputs
            all_vars = set()
            for e in correct_set:
                all_vars.update(e.get("variables", {}).keys())
            inputs = all_vars - outputs
            if not inputs:
                n_dof_fail += 1
                continue

            # DoF 確認：|unknowns| == n_equations
            # outputs は 1 個（微分変数）。inputs は残り。
            # 方程式数 = X。未知数 = outputs 個 = 1 だが、X 方程式と整合させるには
            # ★ 実際は X 個の方程式があるなら、未知数も X 個必要。
            # ここで「outputs に含めるべき変数を増やす」必要あり
            # → outputs はとりあえず 1 個（微分変数）に固定。残り (X-1) は
            #    各追加式の代表変数を自動で出力に追加する（DoF 整合のため）
            if X > 1:
                # 追加式の代表変数（その式の variables 内で他に outputs にない 1 個）を選ぶ
                for i, e in enumerate(correct_set[1:], 1):
                    e_vars = list(e.get("variables", {}).keys())
                    candidates = [v for v in e_vars if v not in outputs and v in all_vars]
                    if not candidates:
                        continue
                    new_out = rng.choice(candidates)
                    outputs.add(new_out)
                    if len(outputs) == X:
                        break

            inputs = all_vars - outputs
            if len(outputs) != X:
                n_dof_fail += 1
                continue
            if not inputs:
                n_dof_fail += 1
                continue

            # ステップ 6: 説明文生成
            if args.dry_run:
                desc = f"[DRY RUN] {X}-equation DAE case combining sources."
            else:
                # rate limit (1 RPS)
                elapsed = time.time() - last_api_time
                if elapsed < RATE_LIMIT_INTERVAL:
                    time.sleep(RATE_LIMIT_INTERVAL - elapsed)
                last_api_time = time.time()
                desc = generate_description(client, correct_set, inputs, outputs, args.model)
                if not desc:
                    n_desc_fail += 1
                    continue

            # ケース作成
            case = {
                "case_id": f"dae_X{X}_{len(x_cases):03d}",
                "original_core_id": f"dae_X{X}_{len(x_cases):03d}",
                "variant_type": f"dae_X{X}",
                "context": desc,
                "input_variables": sorted(inputs),
                "output_variables": sorted(outputs),
                "correct_model_ids": [f"{e['source_id']}__{e['eq_id']}" for e in correct_set],
            }
            x_cases.append(case)

            if len(x_cases) % 10 == 0:
                print(f"  進捗: {len(x_cases)}/{args.n_per_x} (試行 {n_attempted})")

        report["per_X"][X] = {
            "target": args.n_per_x,
            "generated": len(x_cases),
            "attempts": n_attempted,
            "build_fail": n_build_fail,
            "dof_fail": n_dof_fail,
            "dup": n_dup,
            "desc_fail": n_desc_fail,
        }
        all_dae_cases.extend(x_cases)
        print(f"  X={X} 完了: {len(x_cases)} / {args.n_per_x}")

    # 保存
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_dae_cases, f, ensure_ascii=False, indent=2)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DAE ケース生成完了: {len(all_dae_cases)} 件")
    print(f"  保存: {OUT_JSON}")
    print(f"  レポート: {REPORT}")
    print(f"{'='*60}\n")

    print("=== X 別生成数 ===")
    for X, info in report["per_X"].items():
        print(f"  X={X}: {info['generated']:>3}/{info['target']} "
              f"(試行 {info['attempts']}, 失敗内訳：build={info['build_fail']}, "
              f"dof={info['dof_fail']}, dup={info['dup']}, desc={info['desc_fail']})")


if __name__ == "__main__":
    main()
