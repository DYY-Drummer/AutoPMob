#!/usr/bin/env python3
"""出典横断で入出力変数を選び直したケース（multisource_random_io）の説明文を、入出力変数に合うように Anthropic API で書き直す.

本人の決定（2026-10-21）：選び直した 198 件は、元の 33 件（multisource_original）の説明文をそのまま使っていたので、
説明文が入出力変数と合っていなかった（例 ms_cstr_heat_001_rv2：説明文は滞留時間を求めると言うが、滞留時間は入力）。
評価に使う 111 件（v1 の除外 experiments/kato1020/excluded_cases.json に入っていないもの）の説明文を書き直す。
元の 33 件のうち、説明文が自分の出力変数と合っていない 2 件（ORIGINALS_TO_FIX）も、同じ指示で直す（本人の決定、2026-10-21）。
生成はすべて API（モデルは本人が選んだ Opus 4.8。LLM との比較で評価する 4 モデルとは別）。

  python3 rewrite_reselected_io_contexts.py pilot --n 3   # 3 件だけ書き直して表示（training_cases.json は変えない）
  python3 rewrite_reselected_io_contexts.py run           # まだ合格の記録がないケースを書き直す（不合格は最大 3 回）
  python3 rewrite_reselected_io_contexts.py apply         # 合格した書き直しを training_cases.json の context に入れ、記録を書く

生の応答は experiments/reselected_io_contexts_raw.jsonl に追記する。
記録 experiments/reselected_io_contexts.json には、ケースごとに元の説明文、書き直した説明文、入出力変数、モデル、プロンプトの版、トークン数を残す。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)
import anthropic  # noqa: E402

from two_stage_query_conditioned import TRAINING_JSON, load_equations, eq_key, norm  # noqa: E402

EXP = ROOT / "experiments"
RAW = EXP / "reselected_io_contexts_raw.jsonl"
OUT = EXP / "reselected_io_contexts.json"
EXCLUDED = EXP / "kato1020" / "excluded_cases.json"
# 元の説明文が自分の出力変数と合っていないケース（2026-10-21 に 33 件を読んで確かめた）。
# ms_pbr_chemeng_001：触媒重量と出口濃度を求めると言うが、出力は反応速度と滞留時間（出口濃度は入力）。
# ms_pfr_arrhenius_001：「PFR の大きさを決める」と言うが、体積は入力。
ORIGINALS_TO_FIX = ("ms_pbr_chemeng_001", "ms_pfr_arrhenius_001")
MODEL = "claude-opus-4-8"
MAX_TOKENS = 2000
PROMPT_VERSION = "v6"  # v1〜v3（2026-10-21）は、元の説明文が挙げていない入力まで列挙することがあった。v4 で、与えるものとして書く量（元の出力で、いまは入力）をプログラムで決めて渡すようにした。v5・v6 で、新しい出力でない量を求めさせないこと（「PFR の大きさを決める」のような暗黙の要求も含む）を加えた
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You edit the problem descriptions of an evaluation set for retrieving process-model equations.
A description states, in plain words, a physical system and which quantities must be determined.

You receive a description written for one choice of quantities to determine (original_outputs) and a new choice for the same system (new_outputs). "now_given" lists the quantities that the original asked to determine but that are now given. Each quantity comes with its symbol and its meaning taken from the source equations. Rewrite the description for the new choice.

Rules:
- Describe the same physical system and keep the original wording, changing only what the new choice requires.
- State every quantity in new_outputs as a quantity to determine.
- The original may ask, explicitly or implicitly, to determine quantities that are not in new_outputs, including quantities that are not variables of the system at all (for example, "size a reactor" asks for its volume). Remove or reword every such request, so that only the quantities in new_outputs are to be determined. This applies even when new_outputs equals original_outputs.
- Present every quantity in now_given as given. Do not name any other quantity as given unless the original already names it.
- Never present a quantity in new_outputs as given.
- Refer to quantities in plain words by their meaning, as the original does. Do not write symbols or equations.
- Do not add information beyond the original description and the meanings of the quantities.
- Write one or two sentences in English, as short as the changes allow.

Return JSON: {"context": "<the rewritten description>"}."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"context": {"type": "string"}},
    "required": ["context"],
    "additionalProperties": False,
}
SYMBOLISH = re.compile(r"[$\\^_{}=]")


def targets() -> tuple[list, dict]:
    """(書き直すケース, case_id → 元のケース)。入出力を選び直した 111 件と、ORIGINALS_TO_FIX の 2 件（元のケース＝自分）."""
    cases = json.load(open(TRAINING_JSON, encoding="utf-8"))
    excluded = set(json.load(open(EXCLUDED, encoding="utf-8"))["excluded_case_ids"])
    base = {c["case_id"]: c for c in cases if c.get("variant_type") == "multisource_original"}
    ts = [c for c in cases if c.get("variant_type") == "multisource_random_io" and c["case_id"] not in excluded]
    for c in ts:
        b = base[c["original_core_id"]]
        assert sorted(c["correct_model_ids"]) == sorted(b["correct_model_ids"]), c["case_id"]
    fix = [base[i] for i in ORIGINALS_TO_FIX]
    base = {**base, **{c["case_id"]: c for c in fix}}
    for c in fix:
        c.setdefault("original_core_id", c["case_id"])
    return ts + [{**c, "original_core_id": c["case_id"]} for c in fix], base


def meanings(case: dict, eqk: dict) -> dict:
    """記号 → 正解式の変数の定義（重複を除いて出てきた順）."""
    out: dict = {}
    for m in case["correct_model_ids"]:
        for s, d in ((eqk.get(norm(m)) or {}).get("variables") or {}).items():
            if isinstance(d, str) and norm(d) and norm(d) not in out.setdefault(norm(s), []):
                out[norm(s)].append(norm(d))
    return out


def payload(case: dict, base: dict, eqk: dict) -> dict:
    mean = meanings(case, eqk)
    def vs(syms):
        return [{"symbol": norm(s), "meaning": mean.get(norm(s), [])} for s in syms]
    new_in = {norm(v) for v in case["input_variables"]}
    return {
        "original_description": norm(base["context"]),
        "original_outputs": vs(base["output_variables"]),
        "new_outputs": vs(case["output_variables"]),
        "now_given": vs([v for v in base["output_variables"] if norm(v) in new_in]),
    }


def request_params(p: dict) -> dict:
    return dict(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(p, ensure_ascii=False, indent=1)}],
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )


def check(text: str | None, p: dict) -> str | None:
    """形の確認。合格なら None、不合格なら理由."""
    if not text:
        return "empty"
    if SYMBOLISH.search(text):
        return "symbol or equation"
    r = len(text) / max(1, len(p["original_description"]))
    if not 0.5 <= r <= 2.5:
        return f"length ratio {r:.2f}"
    return None


def call(client, params: dict):
    last = None
    for attempt in range(4):
        try:
            return client.messages.create(**params), None
        except anthropic.RateLimitError as e:
            last = e
            time.sleep(20 * (attempt + 1))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last = e
                time.sleep(5 * (attempt + 1))
                continue
            return None, f"{e.status_code}: {e.message}"
        except anthropic.APIConnectionError as e:
            last = e
            time.sleep(5 * (attempt + 1))
    return None, f"gave up: {last}"


def rewrite_one(client, case: dict, p: dict) -> dict:
    msg, err = call(client, request_params(p))
    text = None
    if msg is not None:
        if msg.stop_reason == "refusal":
            err = "refusal"
        else:
            try:
                text = norm(json.loads("".join(b.text for b in msg.content if b.type == "text"))["context"])
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                err = f"parse: {e}"
    rec = {
        "case_id": case["case_id"], "prompt_version": PROMPT_VERSION,
        "model_requested": MODEL, "model_served": getattr(msg, "model", None),
        "stop_reason": getattr(msg, "stop_reason", None),
        "usage": ({"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
                  if msg is not None else None),
        "context": text, "error": err, "check": None if err else check(text, p),
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(RAW, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_raw() -> list:
    if not RAW.exists():
        return []
    return [json.loads(line) for line in open(RAW, encoding="utf-8") if line.strip()]


def passed(recs: list) -> dict:
    """case_id → 今の版で合格した最初の記録."""
    out = {}
    for r in recs:
        if r["prompt_version"] == PROMPT_VERSION and not r["error"] and r["check"] is None and r["case_id"] not in out:
            out[r["case_id"]] = r
    return out


def cmd_pilot(args) -> None:
    ts, base = targets()
    eqk = {eq_key(e): e for e in load_equations() if eq_key(e)}
    client = anthropic.Anthropic()
    pick = [c for c in ts if c["case_id"] in set(args.ids.split(","))] if args.ids else ts[:args.n]
    for c in pick:
        p = payload(c, base[c["original_core_id"]], eqk)
        r = rewrite_one(client, c, p)
        print(f"\n[{c['case_id']}] new in={c['input_variables']} out={c['output_variables']}")
        print("  元:", p["original_description"])
        print("  新:", r["context"], "| check:", r["check"], "| error:", r["error"], "| usage:", r["usage"])


def cmd_run(args) -> None:
    ts, base = targets()
    eqk = {eq_key(e): e for e in load_equations() if eq_key(e)}
    client = anthropic.Anthropic()
    for attempt in range(MAX_ATTEMPTS):
        done = passed(load_raw())
        todo = [c for c in ts if c["case_id"] not in done]
        print(f"試行 {attempt + 1}: 残り {len(todo)} 件")
        if not todo:
            break
        for c in todo:
            r = rewrite_one(client, c, payload(c, base[c["original_core_id"]], eqk))
            if r["error"] or r["check"]:
                print(f"  {c['case_id']}: NG {r['error'] or r['check']}")
    done = passed(load_raw())
    print(f"合格 {len(done)} / {len(ts)} 件")


def cmd_apply(args) -> None:
    ts, base = targets()
    recs = load_raw()
    done = passed(recs)
    missing = [c["case_id"] for c in ts if c["case_id"] not in done]
    if missing:
        sys.exit(f"合格していないケースがある: {missing[:5]} ほか {len(missing)} 件")
    cases = json.load(open(TRAINING_JSON, encoding="utf-8"))
    by_id = {c["case_id"]: c for c in cases}
    rows = []
    for c in ts:
        r = done[c["case_id"]]
        b = base[c["original_core_id"]]
        rows.append({"case_id": c["case_id"], "original_case_id": b["case_id"],
                     "context_before": norm(b["context"]), "context_after": r["context"],
                     "input_variables": c["input_variables"], "output_variables": c["output_variables"],
                     "model_served": r["model_served"], "usage": r["usage"], "time": r["time"]})
        assert norm(by_id[c["case_id"]]["context"]) in (norm(b["context"]), r["context"]), c["case_id"]
        by_id[c["case_id"]]["context"] = r["context"]
    with open(TRAINING_JSON, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    tin = sum(r["usage"]["input_tokens"] for r in recs if r["usage"])
    tout = sum(r["usage"]["output_tokens"] for r in recs if r["usage"])
    meta = {"decision": "本人の決定（2026-10-21）：入出力を選び直した出典横断のケース 111 件と、説明文が出力変数と合っていなかった元のケース 2 件の説明文を、入出力変数に合うように書き直す",
            "model": MODEL, "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
            "n_cases": len(rows), "n_calls": len(recs), "input_tokens": tin, "output_tokens": tout}
    json.dump({"meta": meta, "cases": rows}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"training_cases.json の {len(rows)} 件の context を書き換えた → {OUT.name}（呼び出し {len(recs)} 回、入力 {tin:,}・出力 {tout:,} トークン）")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot"); p.add_argument("--n", type=int, default=3); p.add_argument("--ids", default="")
    p.set_defaults(f=cmd_pilot)
    sub.add_parser("run").set_defaults(f=cmd_run)
    sub.add_parser("apply").set_defaults(f=cmd_apply)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
