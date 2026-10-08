#!/usr/bin/env python3
"""ケース側の変数の言い換え・記号なしの説明文・別の記号を Anthropic API で生成する.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.2
このセッション（Claude Code）の出力では作らない。生成はすべて API（モデルは本人が確定した Opus 5.5）。

  python3 make_var_descriptions.py pilot --n 20   # 通常の API で 20 件。品質・トークン数・料金の見積もり
  python3 make_var_descriptions.py submit         # 合格の記録がないケースを Message Batches API で送る（未回収のバッチがあれば止まる。--force で続行）
  python3 make_var_descriptions.py collect        # バッチが終わるのを待って結果を取り込む
  python3 make_var_descriptions.py retry          # 不合格・失敗のケースを通常の API で作り直す（1 ケース最大 3 回。未回収のバッチがあれば止まる。--force で続行）
  python3 make_var_descriptions.py finalize       # 品質の確認 → var_descriptions.json・_qc.json・_sample50.csv

生の応答は experiments/var_descriptions_raw/*.jsonl に追記する（追跡しない）。
合否・作り直しの回数・finalize の選択に使うのは今のプロンプト版（PROMPT_VERSION）の記録だけ。料金は全版の記録を合算する。
バッチは送ったときの版を batches.json に残し、collect はその版を取り込む記録に付ける（版の記録がない項目は unknown ＝ 今の版とは数えない）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)
import anthropic  # noqa: E402

from two_stage_query_conditioned import load_equations, load_cases, norm, eq_key  # noqa: E402
from external_baselines import SETTINGS, variant_matches  # noqa: E402
from var_desc_qc import align_symbol, check_output, word_jaccard  # noqa: E402

EXP = ROOT / "experiments"
RAW_DIR = EXP / "var_descriptions_raw"
OUT_JSON = EXP / "var_descriptions.json"
QC_JSON = EXP / "var_descriptions_qc.json"
SAMPLE_CSV = EXP / "var_descriptions_sample50.csv"
SPEC = "docs/superpowers/specs/2026-10-01-variable-description-matching-design.md"
MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
PROMPT_VERSION = "v1"
MAX_ATTEMPTS = 3   # 最初の 1 回 ＋ 作り直し 2 回
PRICE = {"standard": (4.0, 20.0), "batch": (2.0, 10.0)}  # USD / 100 万トークン（入力, 出力）。2026-09-25 時点
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You help build an evaluation set for retrieving equations from the literature.
Each request describes a modelling task: a short description ("context") and its input and output variables. Each variable comes with its symbol, its role, and one or more descriptions of that symbol taken from the source equations.

Rewrite the task as an engineer would state it without knowing the notation of the source papers. Return JSON with:

1. "context_nosym": the context with every mention of a variable symbol (for example "$T$", "C_A" or "\\alpha") replaced by plain words for that quantity. Keep all other content. Do not add information.

2. "variables": one entry for each given variable, with
   - "symbol": the symbol exactly as given.
   - "paraphrase": a description of the quantity in your own words, about 3 to 15 words. Do not use the symbol, any other symbol, or mathematical notation. Do not copy the given descriptions; reword them. Omit units. Keep every qualifier that distinguishes this quantity from similar ones, such as inlet, outlet, jacket, coolant, wall, initial, steady-state, reference, or the species or phase it refers to.
   - "alt_symbol": another symbol that a different textbook or engineer might use for the same quantity, in the same LaTeX style as the given symbols. It must differ from the given symbol, from every other given symbol, and from every other alt_symbol in this request.
   - "meaning_conflict": true only if the given descriptions of this symbol name different quantities. In that case, describe the quantity of the first description.

No two variables may share the same paraphrase."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "context_nosym": {"type": "string"},
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "paraphrase": {"type": "string"},
                    "alt_symbol": {"type": "string"},
                    "meaning_conflict": {"type": "boolean"},
                },
                "required": ["symbol", "paraphrase", "alt_symbol", "meaning_conflict"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["context_nosym", "variables"],
    "additionalProperties": False,
}


def setting_a_cases() -> list:
    return [c for c in load_cases() if variant_matches(c.get("variant_type", ""), SETTINGS["A"])]


def eq_index() -> dict:
    return {eq_key(e): e for e in load_equations() if eq_key(e)}


def case_payload(case: dict, eqk: dict) -> dict:
    """LLM に渡す内容（式の LaTeX と ID は渡さない）."""
    ins = [norm(v) for v in case.get("input_variables") or [] if norm(v)]
    outs = [norm(v) for v in case.get("output_variables") or [] if norm(v)]
    variables = []
    for s in dict.fromkeys(ins + outs):
        descs = []
        for m in case.get("correct_model_ids") or []:
            e = eqk.get(norm(m))
            if not e:
                continue
            for k, v in (e.get("variables") or {}).items():
                if norm(k) == s and isinstance(v, str) and norm(v) and norm(v) not in descs:
                    descs.append(norm(v))
        role = "input and output" if (s in ins and s in outs) else ("input" if s in ins else "output")
        variables.append({"symbol": s, "role": role, "descriptions": descs})
    return {"context": norm(case.get("context") or ""), "variables": variables}


def request_params(payload: dict) -> dict:
    return dict(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=1)}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )


def parse_message(msg) -> tuple:
    """(出力 dict or None, エラー文字列 or None)."""
    if msg.stop_reason == "refusal":
        return None, "refusal"
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    if not text:
        return None, f"empty ({msg.stop_reason})"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"json: {e}"


def record(case_id: str, source: str, msg=None, error: str | None = None, *,
           prompt_version: str | None = None) -> dict:
    """raw 記録 1 件。prompt_version は、この出力を作った指示の版（None なら今の PROMPT_VERSION）.

    pilot・retry は今の指示でその場で作るので省略でよい。バッチは送った時点の版で作られているので、
    collect は batches.json に残した版を渡す（取り込む時点で PROMPT_VERSION が変わっていても、送った版で記録する）。
    """
    out, err = parse_message(msg) if msg is not None else (None, error)
    return {
        "case_id": case_id, "source": source,
        "prompt_version": PROMPT_VERSION if prompt_version is None else prompt_version,
        "model_requested": MODEL, "model_served": getattr(msg, "model", None),
        "stop_reason": getattr(msg, "stop_reason", None),
        "usage": ({"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
                  if msg is not None else None),
        "output": out, "error": err,
        "time": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }


def append_raw(name: str, rec: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DIR / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_raw() -> list:
    recs = []
    for p in sorted(RAW_DIR.glob("*.jsonl")):
        recs += [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    return sorted(recs, key=lambda r: r["time"])


def call_standard(client, params: dict, use_fallbacks: bool) -> tuple:
    """通常の API で 1 件（429・5xx・接続エラーは待って最大 4 回）。(メッセージ or None, エラー or None)."""
    last = None
    for attempt in range(4):
        try:
            if use_fallbacks:
                return client.beta.messages.create(**params, betas=[FALLBACK_BETA],
                                                   extra_body={"fallbacks": "default"}), None
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


def consumable(output) -> bool:
    """finalize が KeyError・AttributeError を出さずに読める形か（形だけを見る純粋な述語）.

    True になるのは、output が dict で、context_nosym が str、variables が list、
    その各要素が dict で symbol・paraphrase・alt_symbol が str、meaning_conflict が bool のとき。
    check_output の hard が空でも（meaning_conflict がない出力）、hard が中身の問題だけでも
    （変数が symbol だけ、context_nosym がない出力）、この形でなければ finalize で落ちる。
    """
    if not isinstance(output, dict):
        return False
    if not isinstance(output.get("context_nosym"), str):
        return False
    variables = output.get("variables")
    if not isinstance(variables, list):
        return False
    return all(isinstance(v, dict)
               and isinstance(v.get("symbol"), str)
               and isinstance(v.get("paraphrase"), str)
               and isinstance(v.get("alt_symbol"), str)
               and isinstance(v.get("meaning_conflict"), bool)
               for v in variables)


def judge(rec: dict, payload: dict) -> tuple:
    """(合格か, hard の問題, 目視の印)。合格は、finalize が読める形（consumable）で hard がないときだけ."""
    out = rec.get("output")
    hard, review = check_output(payload, out)
    return (consumable(out) and not hard), hard, review


def by_case(recs: list, payloads: dict) -> dict:
    """case_id ごとの記録（時刻順）。今のプロンプト版（PROMPT_VERSION）の記録だけを残す.

    版の違う記録は別の指示で作られたものなので、合格の判定・作り直しの回数・finalize の選択に使わない
    （版が書かれていない記録も今の版とは確認できないので除く）。料金の合算は別で、finalize は全記録を数える。
    """
    out = {}
    for r in recs:
        if r["case_id"] in payloads and r.get("prompt_version") == PROMPT_VERSION:
            out.setdefault(r["case_id"], []).append(r)
    return out


def pick(rs: list, payload: dict) -> tuple:
    """合格の最新記録。なければ、finalize が読める形（consumable）で記号の集合がそろった最新記録（不合格）。
    どちらもなければ (None, True).

    不合格の側にも consumable を求める。変数が symbol だけ・context_nosym がない・型が違う出力を返すと
    finalize が KeyError・AttributeError で落ちるので、そのような記録は「記録なし」と同じに扱う
    （consumable は output が truthy な dict であることを含む）。
    """
    ok = [r for r in rs if judge(r, payload)[0]]
    if ok:
        return ok[-1], False
    full = [r for r in rs if consumable(r.get("output"))
            and not any(h.startswith(("symbol-set|", "no-output|")) for h in judge(r, payload)[1])]
    return (full[-1], True) if full else (None, True)


def load_batches() -> list:
    """送ったバッチの記録（batches.json）。まだ送っていなければ空."""
    log = RAW_DIR / "batches.json"
    if not log.exists():
        return []
    with open(log, encoding="utf-8") as f:
        return json.load(f)


def uncollected(batches: list) -> list:
    """まだ回収していないバッチの id（collected が False の項目。collected の記録がない項目も回収済みとは言えないので含める）."""
    return [b["id"] for b in batches if not b.get("collected")]


def batch_error_text(result) -> str:
    """成功でなかったバッチの結果（r.result）を、raw 記録の error 用の文字列にする.

    errored: anthropic 0.103.1 では result.error（ErrorResponse）の中の .error（エラー本体）に
    type（invalid_request_error・api_error・overloaded_error など）と message があるので、
    「batch errored: <type>: <message>」にする。階層や項目が欠けていても例外は出さず、取れた分だけで返す。
    canceled・expired など: 「batch <type>」.
    """
    kind = getattr(result, "type", "unknown")
    if kind != "errored":
        return f"batch {kind}"
    err = getattr(getattr(result, "error", None), "error", None)
    detail = ": ".join(str(x) for x in (getattr(err, "type", None), getattr(err, "message", None)) if x)
    return f"batch errored: {detail}" if detail else "batch errored"


def client_() -> "anthropic.Anthropic":
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def payloads_all() -> dict:
    eqk = eq_index()
    return {c["case_id"]: case_payload(c, eqk) for c in setting_a_cases()}


def cmd_pilot(args) -> None:
    payloads = payloads_all()
    ids = random.Random(20261001).sample(sorted(payloads), args.n)
    client = client_()
    tin = tout = n_ok = 0
    for cid in ids:
        msg, err = call_standard(client, request_params(payloads[cid]), not args.no_fallbacks)
        rec = record(cid, "pilot", msg, err)
        append_raw("pilot", rec)
        ok, hard, review = judge(rec, payloads[cid])
        n_ok += ok
        if rec["usage"]:
            tin += rec["usage"]["input_tokens"]
            tout += rec["usage"]["output_tokens"]
        print(f"  {cid}: {'OK' if ok else 'NG'} served={rec['model_served']} usage={rec['usage']} "
              f"err={rec['error']} hard={hard[:3]} review={review[:3]}", flush=True)
    n, N = len(ids), len(payloads)
    pin, pout = PRICE["batch"]
    est_in, est_out = tin / n * N, tout / n * N
    print(f"pilot: {n_ok}/{n} 合格。1 件あたり平均トークン 入力 {tin / n:.0f}・出力 {tout / n:.0f}")
    print(f"全 {N} 件をバッチで送る見積もり: 入力 {est_in / 1e6:.2f}M・出力 {est_out / 1e6:.2f}M → "
          f"約 {est_in / 1e6 * pin + est_out / 1e6 * pout:.1f} USD（{MODEL}, バッチ単価 入力 {pin}・出力 {pout} USD/100万）")


def cmd_submit(args) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    pending = uncollected(load_batches())
    if pending and not args.force:
        print(f"未回収のバッチがある: {', '.join(pending)}。先に collect で回収すること"
              "（回収前に送ると、結果が未取り込みの同じケースをもう一度送って料金が二重にかかる）。"
              "それでも送るなら --force を付ける", file=sys.stderr)
        sys.exit(2)
    payloads = payloads_all()
    groups = by_case(load_raw(), payloads)
    todo = [cid for cid in sorted(payloads) if not any(judge(r, payloads[cid])[0] for r in groups.get(cid, []))]
    if not todo:
        print("送るケースはない")
        return
    batch = client_().messages.batches.create(requests=[
        Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**request_params(payloads[cid])))
        for cid in todo])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = RAW_DIR / "batches.json"
    batches = load_batches()                 # 送っているあいだの更新を失わないよう、記録は書く直前に読み直す
    batches.append({"id": batch.id, "n": len(todo), "collected": False,
                    "prompt_version": PROMPT_VERSION,    # 送った時点の版。collect が取り込む記録にこの版を付ける
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    json.dump(batches, open(log, "w"), indent=1)
    print(f"submitted batch {batch.id}: {len(todo)} requests")


def cmd_collect(args) -> None:
    log = RAW_DIR / "batches.json"
    batches = load_batches()                 # uncollected と同じ読み方（記録がなければ空、collected がなければ未回収）
    client = client_()
    for b in batches:
        if b.get("collected"):
            continue
        pv = b.get("prompt_version", "unknown")   # 送った時点の版。版の記録がない項目は unknown（今の版とは数えない）
        while True:
            cur = client.messages.batches.retrieve(b["id"])
            if cur.processing_status == "ended":
                break
            print(f"  {b['id']}: {cur.processing_status} (processing={cur.request_counts.processing})", flush=True)
            time.sleep(60)
        n = Counter()
        for r in client.messages.batches.results(b["id"]):
            if r.result.type == "succeeded":
                rec = record(r.custom_id, "batch", r.result.message, prompt_version=pv)
            else:
                rec = record(r.custom_id, "batch", None, batch_error_text(r.result), prompt_version=pv)
            append_raw("batch", rec)
            n[r.result.type] += 1
        b["collected"] = True
        json.dump(batches, open(log, "w"), indent=1)
        print(f"collected {b['id']}: {dict(n)}")


def cmd_retry(args) -> None:
    pending = uncollected(load_batches())
    if pending and not args.force:
        print(f"未回収のバッチがある: {', '.join(pending)}。先に collect で回収すること"
              "（回収前に作り直すと、結果が未取り込みの同じケースを通常の API でもう一度作って料金が二重にかかる）。"
              "それでも作り直すなら --force を付ける", file=sys.stderr)
        sys.exit(2)
    payloads = payloads_all()
    groups = by_case(load_raw(), payloads)
    todo = [cid for cid in sorted(payloads)
            if not any(judge(r, payloads[cid])[0] for r in groups.get(cid, []))
            and len(groups.get(cid, [])) < MAX_ATTEMPTS]
    print(f"retry: {len(todo)} cases")
    client = client_()
    for cid in todo:
        for _ in range(MAX_ATTEMPTS - len(groups.get(cid, []))):
            msg, err = call_standard(client, request_params(payloads[cid]), not args.no_fallbacks)
            rec = record(cid, "retry", msg, err)
            append_raw("retry", rec)
            ok, hard, _ = judge(rec, payloads[cid])
            print(f"  {cid}: {'OK' if ok else 'NG'} err={rec['error']} hard={hard[:3]}", flush=True)
            if ok:
                break


def quantiles(a: list) -> dict:
    import numpy as np
    x = np.asarray(a, dtype=float)
    return {"mean": round(float(x.mean()), 4),
            **{f"p{q}": round(float(np.percentile(x, q)), 4) for q in (10, 50, 90)}}


def cmd_finalize(args) -> None:
    payloads = payloads_all()
    raw = load_raw()
    groups = by_case(raw, payloads)
    out_cases, missing, failed = {}, [], []
    hard_counts, review_counts, served = Counter(), Counter(), Counter()
    overlaps, n_vars, n_conflict = [], 0, 0
    for cid in sorted(payloads):
        rec, qc_failed = pick(groups.get(cid, []), payloads[cid])
        if rec is None:
            missing.append(cid)
            continue
        hard, review = check_output(payloads[cid], rec["output"])
        if qc_failed:
            failed.append(cid)
        hard_counts.update(h.split("|")[0] for h in hard)
        review_counts.update(r.split("|")[0] for r in review)
        served[rec["model_served"]] += 1
        # 鍵は payload の綴りにする（NFC だけ違う記号もそろえる。var_conditions が、ケースの元の記号で引くため）
        psyms = [pv["symbol"] for pv in payloads[cid]["variables"]]
        vars_out = {align_symbol(v["symbol"], psyms):
                    {"paraphrase": v["paraphrase"], "alt_symbol": v["alt_symbol"],
                     "meaning_conflict": bool(v["meaning_conflict"])}
                    for v in rec["output"]["variables"]}
        for pv in payloads[cid]["variables"]:
            v = vars_out.get(pv["symbol"])
            if v is None:
                continue
            n_vars += 1
            n_conflict += v["meaning_conflict"]
            overlaps.append(max((word_jaccard(v["paraphrase"], d) for d in pv["descriptions"]), default=0.0))
        out_cases[cid] = {"context_nosym": rec["output"]["context_nosym"], "variables": vars_out,
                          "model_served": rec["model_served"], "source": rec["source"],
                          "prompt_version": rec["prompt_version"],
                          "qc_failed": qc_failed, "qc_hard": hard, "qc_review": review}
    usage = {}
    for r in raw:      # 料金は全版の記録を合算する（版が違っても払ったので）。groups は今の版だけ
        if not r.get("usage"):
            continue
        u = usage.setdefault(r["source"], {"input_tokens": 0, "output_tokens": 0, "n": 0})
        u["input_tokens"] += r["usage"]["input_tokens"]
        u["output_tokens"] += r["usage"]["output_tokens"]
        u["n"] += 1
    for src, u in usage.items():
        pin, pout = PRICE["batch" if src == "batch" else "standard"]
        u["usd"] = round(u["input_tokens"] / 1e6 * pin + u["output_tokens"] / 1e6 * pout, 2)
    meta = {"model": MODEL, "effort": EFFORT, "prompt_version": PROMPT_VERSION, "spec": SPEC,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_cases": len(out_cases), "n_missing": len(missing), "n_qc_failed": len(failed),
            "served_models": dict(served)}
    json.dump({"meta": meta, "cases": out_cases}, open(OUT_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    qc = {"n_cases_setting_A": len(payloads), "missing": missing, "qc_failed": failed,
          "hard_issue_counts": dict(hard_counts), "review_flag_counts": dict(review_counts),
          "n_variables": n_vars, "meaning_conflict": n_conflict,
          "meaning_conflict_ratio": n_conflict / n_vars if n_vars else 0.0,
          "word_overlap_with_source": quantiles(overlaps) if overlaps else None,
          "served_models": dict(served), "usage": usage,
          "usd_total": round(sum(u["usd"] for u in usage.values()), 2)}
    json.dump(qc, open(QC_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pool = [(cid, pv) for cid in sorted(out_cases) for pv in payloads[cid]["variables"]
            if pv["symbol"] in out_cases[cid]["variables"]]
    with open(SAMPLE_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "symbol", "role", "source_descriptions", "paraphrase", "alt_symbol",
                    "meaning_conflict", "meaning_preserved", "note"])
        for cid, pv in random.Random(50).sample(pool, min(50, len(pool))):
            v = out_cases[cid]["variables"][pv["symbol"]]
            w.writerow([cid, pv["symbol"], pv["role"], " | ".join(pv["descriptions"]), v["paraphrase"],
                        v["alt_symbol"], v["meaning_conflict"], "", ""])
    print(json.dumps({k: qc[k] for k in ("hard_issue_counts", "review_flag_counts", "meaning_conflict_ratio",
                                         "word_overlap_with_source", "served_models", "usd_total")},
                     ensure_ascii=False, indent=1))
    print(f"cases={len(out_cases)} missing={len(missing)} qc_failed={len(failed)} → {OUT_JSON}")
    if missing:
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--no-fallbacks", action="store_true")
    s = sub.add_parser("submit")
    s.add_argument("--force", action="store_true",
                   help="未回収のバッチがあっても送る（二重に送って料金が二重にかかる恐れがある）")
    sub.add_parser("collect")
    r = sub.add_parser("retry")
    r.add_argument("--no-fallbacks", action="store_true")
    r.add_argument("--force", action="store_true",
                   help="未回収のバッチがあっても作り直す（バッチと二重に作って料金が二重にかかる恐れがある）")
    sub.add_parser("finalize")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    {"pilot": cmd_pilot, "submit": cmd_submit, "collect": cmd_collect,
     "retry": cmd_retry, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
