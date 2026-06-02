"""Phase 5: reranker hyperparameter sweep.

各 config を set_aware_reranker.py の子プロセスとして起動し，結果 JSON を集約．
A100×4 サーバでは CUDA_VISIBLE_DEVICES を 0..3 にラウンドロビンしつつ
最大 --max-parallel 個の subprocess を並列起動する．

CPU でもそのまま動く（CUDA_VISIBLE_DEVICES が無視されるだけ）．
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
EXP_DIR = ROOT / "experiments"
SWEEP_DIR = EXP_DIR / "phase5_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- デフォルト sweep 設計 ----------------
DEFAULT_SWEEP_A = {
    "lr": [3e-4, 1e-3, 3e-3],
    "hidden_dim": [64, 128, 256],
    "epochs": [20, 40],
    "margin": [0.1, 0.3],
}

DEFAULT_SWEEP_B_TEMPLATE = {
    # refine 用：ベスト config の周辺を探索（実行時に best を読んで生成）
    "lr_offsets": [0.5, 1.0, 2.0],          # best_lr * これら
    "hidden_dim_offsets": [-64, 0, 64],     # best_hidden_dim + これら
    "epochs": [30, 50, 80],
    "margin_offsets": [-0.05, 0, 0.05],     # best_margin + これら
    "top_k": [30, 50, 100],
    "n_neg_samples": [4, 8, 16],
}

# 固定パラメータ
FIXED_PARAMS = {
    "seeds": 10,
    "modes": "baseline,reranker-10S",
    "variants": "original,multisource_v3,dae_",
    "batch_size": 16,
    "weight_decay": 1e-4,
    "top_k": 50,
    "n_neg_samples": 8,
}


def make_configs(sweep_dict: dict) -> list[dict]:
    keys = list(sweep_dict.keys())
    combos = list(itertools.product(*[sweep_dict[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


def config_id(cfg: dict) -> str:
    """安定 ID（ファイル名で使う）."""
    parts = []
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, float):
            sv = f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
        else:
            sv = str(v)
        parts.append(f"{k}={sv}")
    return "_".join(parts)


def run_one(cfg: dict, gpu_id: int | None, output_dir: Path,
            python_exe: str, fixed: dict, dry_run: bool) -> dict:
    """1 config を子プロセスで実行し，結果を返す."""
    cid = config_id(cfg)
    out_path = output_dir / f"{cid}.json"

    if out_path.exists() and not dry_run:
        # 既に走り済み → skip
        try:
            data = json.loads(out_path.read_text())
            return {"config_id": cid, "config": cfg, "result_path": str(out_path),
                    "status": "cached", "summary": _summarize(data)}
        except Exception:
            pass  # 壊れていたら再実行

    cmd = [python_exe, "set_aware_reranker.py",
           "--seeds", str(fixed["seeds"]),
           "--modes", fixed["modes"],
           "--variants", fixed["variants"],
           "--output", str(out_path),
           "--top-k", str(cfg.get("top_k", fixed["top_k"])),
           "--epochs", str(cfg.get("epochs", 15)),
           "--lr", str(cfg.get("lr", 1e-3)),
           "--hidden-dim", str(cfg.get("hidden_dim", 64)),
           "--margin", str(cfg.get("margin", 0.1)),
           "--batch-size", str(cfg.get("batch_size", fixed["batch_size"])),
           "--n-neg-samples", str(cfg.get("n_neg_samples", fixed["n_neg_samples"])),
           "--weight-decay", str(cfg.get("weight_decay", fixed["weight_decay"]))]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # 子プロセスの print を即時 flush（ライブ進捗用）
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if dry_run:
        return {"config_id": cid, "cmd": cmd, "gpu": gpu_id, "status": "dry_run"}

    log_path = output_dir / f"{cid}.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    dt = time.time() - t0

    if proc.returncode != 0:
        return {"config_id": cid, "config": cfg, "status": "failed",
                "returncode": proc.returncode, "log": str(log_path), "elapsed_s": dt}

    if not out_path.exists():
        return {"config_id": cid, "config": cfg, "status": "no_output",
                "log": str(log_path), "elapsed_s": dt}

    data = json.loads(out_path.read_text())
    return {"config_id": cid, "config": cfg, "result_path": str(out_path),
            "status": "ok", "elapsed_s": dt, "summary": _summarize(data)}


def _summarize(data: dict) -> dict:
    """重要メトリクスだけ抽出（reranker-10S の MRR_first / Recall@K_correct / MAP の mean）."""
    out = {}
    for mode, res in data.get("results", {}).items():
        s = {}
        for m in ("MRR_first", "MAP", "Recall@K_correct", "FullRecall@3",
                  "multi_only__Recall@K_correct"):
            v = res.get(m)
            if isinstance(v, dict) and "mean" in v:
                s[m] = v["mean"]
        out[mode] = s
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B", "custom"], default="A",
                        help="A: broad sweep, B: refine（要 best_config_path）, custom: --sweep-json で指定")
    parser.add_argument("--best-config-path",
                        help="Phase B 用，A の analyze 結果 JSON")
    parser.add_argument("--sweep-json",
                        help="Phase custom 用，{パラメータ: [候補値]} の JSON")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="同時実行 config 数（≈ GPU 枚数）")
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="使用 GPU 枚数（0 なら CPU で逐次）")
    parser.add_argument("--output-subdir", default=None,
                        help="experiments/phase5_sweep/<subdir> 配下に出力（既定: phase_<A|B|custom>）")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true",
                        help="コマンドだけ表示，実行しない")
    parser.add_argument("--summary-out",
                        help="集約結果 JSON のパス（既定: <subdir>/_summary.json）")
    args = parser.parse_args()

    # --- sweep 定義 ---
    if args.phase == "A":
        sweep = DEFAULT_SWEEP_A
        subdir = args.output_subdir or "phase_A"
    elif args.phase == "B":
        if not args.best_config_path:
            sys.exit("Phase B には --best-config-path が必要")
        best = json.loads(Path(args.best_config_path).read_text())["best_config"]
        t = DEFAULT_SWEEP_B_TEMPLATE
        sweep = {
            "lr": [best["lr"] * o for o in t["lr_offsets"]],
            "hidden_dim": [max(16, best["hidden_dim"] + o) for o in t["hidden_dim_offsets"]],
            "epochs": t["epochs"],
            "margin": [max(0.01, best["margin"] + o) for o in t["margin_offsets"]],
            "top_k": t["top_k"],
            "n_neg_samples": t["n_neg_samples"],
        }
        subdir = args.output_subdir or "phase_B"
    else:
        if not args.sweep_json:
            sys.exit("Phase custom には --sweep-json が必要")
        sweep = json.loads(Path(args.sweep_json).read_text())
        subdir = args.output_subdir or "phase_custom"

    out_dir = SWEEP_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out) if args.summary_out else (out_dir / "_summary.json")

    configs = make_configs(sweep)
    print(f"Sweep: {sweep}")
    print(f"Total configs: {len(configs)}")
    print(f"Output dir: {out_dir}")
    print(f"Parallel: {args.max_parallel}, GPUs: {args.n_gpus}, dry_run: {args.dry_run}")

    # GPU ラウンドロビン割当
    def gpu_for(idx: int) -> int | None:
        if args.n_gpus <= 0:
            return None
        return idx % args.n_gpus

    results = []
    t0 = time.time()

    if args.max_parallel <= 1 or args.dry_run:
        for i, cfg in enumerate(configs):
            r = run_one(cfg, gpu_for(i), out_dir, args.python_exe, FIXED_PARAMS, args.dry_run)
            results.append(r)
            print(f"  [{i+1}/{len(configs)}] {r['config_id']}: {r['status']}", flush=True)
            if r.get("summary", {}).get("reranker-10S"):
                s = r["summary"]["reranker-10S"]
                print(f"      rer R@K_c={s.get('Recall@K_correct', 0):.4f}  "
                      f"MAP={s.get('MAP', 0):.4f}  MRR_f={s.get('MRR_first', 0):.4f}")
    else:
        with ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
            future_to_cfg = {ex.submit(run_one, cfg, gpu_for(i), out_dir,
                                       args.python_exe, FIXED_PARAMS, args.dry_run): (i, cfg)
                             for i, cfg in enumerate(configs)}
            for fut in as_completed(future_to_cfg):
                r = fut.result()
                results.append(r)
                i, cfg = future_to_cfg[fut]
                print(f"  [{len(results)}/{len(configs)}] {r['config_id']}: {r['status']}", flush=True)
                if r.get("summary", {}).get("reranker-10S"):
                    s = r["summary"]["reranker-10S"]
                    print(f"      rer R@K_c={s.get('Recall@K_correct', 0):.4f}  "
                          f"MAP={s.get('MAP', 0):.4f}  MRR_f={s.get('MRR_first', 0):.4f}")

    elapsed = time.time() - t0
    summary = {
        "phase": args.phase,
        "sweep_definition": sweep,
        "fixed_params": FIXED_PARAMS,
        "n_configs": len(configs),
        "elapsed_s": elapsed,
        "results": results,
    }
    if args.dry_run:
        print(f"\nDry-run: not writing summary to {summary_path}")
    else:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nTotal elapsed: {elapsed/60:.1f} min")
        print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
