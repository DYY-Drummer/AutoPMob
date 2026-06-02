"""Phase 5: sweep result analyzer.

run_phase5_sweep.py が出力した _summary.json を読み込み，
- 主指標で全 config を ランキング
- 各ハイパーパラメータの感度分析（margin-of-mean）
- Top-N 表示 + best config を JSON で出力（Phase 5.B refine の入力）
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_rows(summary: dict, mode: str = "reranker-10S",
                 metric: str = "Recall@K_correct") -> list[dict]:
    rows = []
    for r in summary.get("results", []):
        if r.get("status") not in ("ok", "cached"):
            continue
        s = r.get("summary", {}).get(mode, {})
        val = s.get(metric)
        if val is None:
            continue
        row = {"config_id": r["config_id"], **r["config"],
               "primary": float(val),
               "MAP": s.get("MAP"),
               "MRR_first": s.get("MRR_first"),
               "FullRecall@3": s.get("FullRecall@3"),
               "Recall@K_correct": s.get("Recall@K_correct"),
               "status": r["status"]}
        rows.append(row)
    return rows


def _summarize_config_json(data: dict, mode: str) -> dict:
    """個別 config JSON（set_aware_reranker.py 出力）から重要メトリクス mean を抽出."""
    res = data.get("results", {}).get(mode, {})
    out = {}
    for m in ("MRR_first", "MAP", "Recall@K_correct", "FullRecall@3",
              "multi_only__Recall@K_correct"):
        v = res.get(m)
        if isinstance(v, dict) and "mean" in v:
            out[m] = v["mean"]
    return out


def rows_from_config_globs(globs: list[str], mode: str = "reranker-10S",
                           metric: str = "Recall@K_correct") -> list[dict]:
    """複数の個別 config JSON を直接読み込んで rows を作る（sweep を跨いで結合可能）."""
    import glob as _glob
    seen = set()
    rows = []
    for g in globs:
        for path in sorted(_glob.glob(g)):
            if "_summary" in Path(path).name:
                continue
            try:
                data = json.loads(Path(path).read_text())
            except Exception:
                continue
            cfg = data.get("config", {})
            # config_id を sweep 由来パラメータから合成
            cid_parts = []
            for k in ("lr", "hidden_dim", "epochs", "margin", "top_k", "n_neg_samples"):
                if k in cfg:
                    cid_parts.append(f"{k}={cfg[k]}")
            cid = "_".join(cid_parts) or Path(path).stem
            if cid in seen:
                continue
            seen.add(cid)
            s = _summarize_config_json(data, mode)
            val = s.get(metric)
            if val is None:
                continue
            rows.append({"config_id": cid,
                         "lr": cfg.get("lr"), "hidden_dim": cfg.get("hidden_dim"),
                         "epochs": cfg.get("epochs"), "margin": cfg.get("margin"),
                         "top_k": cfg.get("top_k"), "n_neg_samples": cfg.get("n_neg_samples"),
                         "primary": float(val),
                         "MAP": s.get("MAP"), "MRR_first": s.get("MRR_first"),
                         "FullRecall@3": s.get("FullRecall@3"),
                         "Recall@K_correct": s.get("Recall@K_correct"),
                         "source": Path(path).parent.name, "status": "ok"})
    return rows


def rank(rows: list[dict], metric_key: str = "primary",
         desc: bool = True) -> list[dict]:
    return sorted(rows, key=lambda r: r.get(metric_key) or 0.0, reverse=desc)


def sensitivity(rows: list[dict], param: str,
                metric_key: str = "primary") -> list[dict]:
    """同じパラメータ値で他パラメータを変えた config 群を平均化."""
    buckets = defaultdict(list)
    for r in rows:
        v = r.get(param)
        if v is None: continue
        m = r.get(metric_key)
        if m is None: continue
        buckets[v].append(m)
    out = []
    for v in sorted(buckets, key=lambda x: (isinstance(x, str), x)):
        vals = buckets[v]
        out.append({"value": v, "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals))})
    return out


def print_table(rows: list[dict], top_n: int = 10,
                metric_label: str = "Recall@K_correct"):
    print(f"\n=== TOP {top_n} configs by {metric_label} ===")
    cols = ["lr", "hidden_dim", "epochs", "margin", "top_k", "n_neg_samples"]
    header = "rank  " + "  ".join(f"{c:>11s}" for c in cols) + \
             f"  {'primary':>9s}  {'MAP':>7s}  {'MRR_f':>7s}  {'FullR@3':>9s}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows[:top_n], 1):
        row_str = f"{i:>4d}  "
        for c in cols:
            v = r.get(c)
            if v is None: row_str += f"{'-':>11s}  "
            elif isinstance(v, float): row_str += f"{v:>11.4g}  "
            else: row_str += f"{v:>11}  "
        row_str += (f"{r['primary']:>9.4f}  "
                    f"{(r.get('MAP') or 0):>7.4f}  "
                    f"{(r.get('MRR_first') or 0):>7.4f}  "
                    f"{(r.get('FullRecall@3') or 0):>9.4f}")
        print(row_str)


def print_sensitivity(rows: list[dict], params: list[str],
                       metric_label: str = "Recall@K_correct"):
    print(f"\n=== Parameter sensitivity ({metric_label}, mean ± std over other params) ===")
    for p in params:
        s = sensitivity(rows, p)
        if not s: continue
        print(f"\n  -- {p} --")
        print(f"  {'value':>12s}  {'n':>4s}  {'mean':>8s}  {'std':>7s}  {'min':>8s}  {'max':>8s}")
        for s_row in s:
            v = s_row["value"]
            vstr = f"{v:.4g}" if isinstance(v, float) else str(v)
            print(f"  {vstr:>12s}  {s_row['n']:>4d}  "
                  f"{s_row['mean']:>8.4f}  {s_row['std']:>7.4f}  "
                  f"{s_row['min']:>8.4f}  {s_row['max']:>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary",
                    help="run_phase5_sweep.py の _summary.json（--config-glob と排他）")
    ap.add_argument("--config-glob", nargs="+",
                    help="個別 config JSON の glob を複数指定（sweep を跨いで結合）。"
                         "例: 'experiments/phase5_sweep/phase_A/*.json' 'experiments/phase5_sweep/phase_local_lr/*.json'")
    ap.add_argument("--metric", default="Recall@K_correct",
                    choices=["Recall@K_correct", "MAP", "MRR_first", "FullRecall@3"],
                    help="ランキング指標")
    ap.add_argument("--mode", default="reranker-10S")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out-best", help="best config を JSON で出力（Phase 5.B 入力）")
    args = ap.parse_args()

    if args.config_glob:
        rows = rows_from_config_globs(args.config_glob, mode=args.mode, metric=args.metric)
        print(f"Loaded {len(rows)} configs from globs: {args.config_glob}")
        # swept params = 値が複数種類あるパラメータを自動検出
        swept_params = []
        for p in ("lr", "hidden_dim", "epochs", "margin", "top_k", "n_neg_samples"):
            vals = {r.get(p) for r in rows if r.get(p) is not None}
            if len(vals) > 1:
                swept_params.append(p)
    elif args.summary:
        summary = load_summary(Path(args.summary))
        rows = extract_rows(summary, mode=args.mode, metric=args.metric)
        print(f"Loaded {len(rows)} successful configs from {args.summary}")
        swept_params = list(summary.get("sweep_definition", {}).keys())
    else:
        ap.error("--summary か --config-glob のいずれかが必要")

    if not rows:
        print("No successful configs to analyze."); return

    ranked = rank(rows)
    print_table(ranked, top_n=args.top_n, metric_label=args.metric)

    print_sensitivity(rows, swept_params, metric_label=args.metric)

    best = ranked[0]
    print(f"\n=== Best config ===")
    print(json.dumps({k: best[k] for k in swept_params if k in best}, indent=2))
    print(f"  {args.metric} = {best['primary']:.4f}")

    if args.out_best:
        out = {"metric": args.metric, "best_config": {k: best[k] for k in swept_params if k in best},
               "best_primary": best["primary"],
               "best_full_row": best, "swept_params": swept_params}
        Path(args.out_best).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nbest config saved: {args.out_best}")


if __name__ == "__main__":
    main()
