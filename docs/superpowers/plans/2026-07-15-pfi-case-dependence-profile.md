# PFIケース別依存プロファイル Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ある特徴量を置換したとき正解→不正解に転じるケースを per-case で同定し、その依存ケース群を属性で特徴づける記述プロファイル分析を実装する。

**Architecture:** 既存 `analyze_pfi.py`（集約PFI）を拡張し、ケース内置換の per-case Recall 低下・反転・自ケース値 `sep` を `experiments/pfi_per_case.json` に出力する（集約 `pfi_results.json` は不変）。新規 `analyze_pfi_profile.py` がそのダンプを読み、依存/頑健の属性対比・検定・図を生成する。重い訓練は1回、プロファイル/作図は軽量に再実行可能。

**Tech Stack:** Python, numpy, scipy.stats（mannwhitneyu / spearmanr）, matplotlib, pytest。

## Global Constraints

- 対象は設定A（1,823件）のみ。メトリクスは `Recall@K_correct`（`evaluate_multi_eq.case_metrics` の同名キー）。
- 集約出力 `experiments/pfi_results.json` は現行と同一に保つ（訓練はseed固定で決定的）。
- per-case 収集は `scope="case"`（ケース内置換）のみ。対象は単独10特徴 ＋ `GROUP_var`。
- `sep`（自ケース値）は単独特徴のみ定義。群（`GROUP_var`）では `null`。
- 二値 `flip` は `base_R==1.0` のケースに限定。連続 `drop` は全ケール対象。
- 依存 = `base_R==1.0` かつ `flip==1`／頑健 = `base_R==1.0` かつ `flip==0`。
- 図は作文ルール準拠：エラーバー（SEM）明示・軸ラベルは定義済み語・主観語なし・自己完結キャプション。
- テストは pytest、`tests/` 配下、既知値と `tmp_path` で純関数を検証（`tests/test_significance.py` に倣う）。
- 各コミットメッセージ末尾に `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

---

## File Structure

- `analyze_pfi.py`（**変更**）: per-case 純関数2つを追加、`score_to_RK` に per-case 返却を追加、`run()` に per-case 収集とダンプ、`run()` へ smoke 用パラメータを追加。
- `analyze_pfi_profile.py`（**新規**）: per-case ダンプを読むプロファイル純関数群＋図/統計出力の `main()`。
- `tests/test_pfi_percase.py`（**新規**）: Task 1・2 の純関数テスト。
- `tests/test_pfi_profile.py`（**新規**）: Task 4 のプロファイル純関数テスト。
- 出力: `experiments/pfi_per_case.json`, `experiments/pfi_profile_stats.json`, `docs/figures/fig_pfi_dependence_profile.png` / `.pdf`。

---

## Task 1: per-case 純関数（`per_case_signal`, `sep_for_feature`）

**Files:**
- Modify: `analyze_pfi.py`（`FEATURES` 定義の直後、`train_and_cache` の前に追加）
- Test: `tests/test_pfi_percase.py`

**Interfaces:**
- Produces:
  - `per_case_signal(base_R: float, perm_Rs: list[float]) -> dict` — キー `perm_R_mean, drop, flip, flip_rate`。
  - `sep_for_feature(feats: np.ndarray, cands: list[int], corr: set[int], col: int) -> float` — 正解行平均−不正解行平均（片側が空なら `nan`）。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_pfi_percase.py`:
```python
import math
import numpy as np
from analyze_pfi import per_case_signal, sep_for_feature


def test_per_case_signal_full_flip():
    # base=1.0、全置換で 1.0 未満 → flip=1, flip_rate=1.0
    s = per_case_signal(1.0, [0.5, 0.0, 0.5, 0.5])
    assert abs(s["perm_R_mean"] - 0.375) < 1e-9
    assert abs(s["drop"] - 0.625) < 1e-9
    assert s["flip"] == 1
    assert abs(s["flip_rate"] - 1.0) < 1e-9


def test_per_case_signal_minority_no_flip():
    # base=1.0、過半未満(2/5)しか崩れない → flip=0
    s = per_case_signal(1.0, [1.0, 1.0, 1.0, 0.5, 0.5])
    assert s["flip"] == 0
    assert abs(s["flip_rate"] - 0.4) < 1e-9


def test_per_case_signal_not_solved_never_flips():
    # base<1.0 は flip 対象外（定義上 flip=0）だが drop は測る
    s = per_case_signal(0.5, [0.0, 0.0])
    assert s["flip"] == 0
    assert abs(s["drop"] - 0.5) < 1e-9


def test_sep_for_feature_basic():
    # col0: 正解行(cands 2,4)=[0.8,0.6] 平均0.7、不正解行=[0.1,0.3] 平均0.2 → sep=0.5
    feats = np.array([[0.1], [0.8], [0.3], [0.6]], dtype=np.float32)
    cands = [1, 2, 3, 4]
    corr = {2, 4}
    assert abs(sep_for_feature(feats, cands, corr, 0) - 0.5) < 1e-6


def test_sep_for_feature_nan_when_one_side_empty():
    feats = np.array([[0.1], [0.8]], dtype=np.float32)
    assert math.isnan(sep_for_feature(feats, [1, 2], {1, 2}, 0))  # 不正解行なし
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python3 -m pytest tests/test_pfi_percase.py -v`
Expected: FAIL（`ImportError: cannot import name 'per_case_signal'`）

- [ ] **Step 3: 純関数を実装**

`analyze_pfi.py` に追加（`PERM_TARGETS` 定義の直後）:
```python
def per_case_signal(base_R: float, perm_Rs: list) -> dict:
    """置換前 Recall と N_PERM 回の置換後 Recall から per-case 信号を計算.

    flip: base_R==1.0（baseline 解済み）かつ過半の置換で Recall<1.0 になったら 1。
    drop: 期待低下 base_R - mean(perm_Rs)（全ケース対象、連続の主信号）。
    """
    perm = np.asarray(perm_Rs, dtype=float)
    perm_mean = float(perm.mean()) if perm.size else float("nan")
    n_below = int(np.sum(perm < 1.0))
    flip_rate = n_below / perm.size if perm.size else 0.0
    flip = 1 if (base_R == 1.0 and flip_rate >= 0.5) else 0
    return {"perm_R_mean": perm_mean, "drop": float(base_R - perm_mean),
            "flip": flip, "flip_rate": float(flip_rate)}


def sep_for_feature(feats, cands, corr, col: int) -> float:
    """そのケースで特徴 col が正解を不正解からどれだけ分離しているか.

    sep = mean(正解候補行の col) - mean(不正解候補行の col)。
    正解/不正解いずれかの行が無ければ nan。feats 行は cands と同順。
    """
    corr_mask = np.array([c in corr for c in cands], dtype=bool)
    pos = feats[corr_mask, col]
    neg = feats[~corr_mask, col]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(pos.mean() - neg.mean())
```

- [ ] **Step 4: テスト成功を確認**

Run: `python3 -m pytest tests/test_pfi_percase.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: コミット**

```bash
git add analyze_pfi.py tests/test_pfi_percase.py
git commit -m "feat(pfi): per-case信号(drop/flip)とsep分離度の純関数を追加

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `score_to_RK` に per-case 返却を追加

**Files:**
- Modify: `analyze_pfi.py:169-183`（`score_to_RK`）
- Test: `tests/test_pfi_percase.py`（追記）

**Interfaces:**
- Consumes: `Reranker`（`two_stage_query_conditioned`）, `compute_all_ranks`, `case_metrics`（既存 import）。
- Produces: `score_to_RK(model, cache, feat_override=None, return_per_case=False)` — `return_per_case=True` で `(aggregate: float, per_case: list[float])`。`per_case[i]` は cache 順のケース単位 `Recall@K_correct`。既定 `False` で現行どおり float を返す。

- [ ] **Step 1: 失敗するテストを書く（不変量：per-caseの平均＝集約値）**

`tests/test_pfi_percase.py` に追記:
```python
import torch
from two_stage_query_conditioned import Reranker
from analyze_pfi import score_to_RK


def _tiny_cache():
    # 2ケース、候補5件ずつ、1特徴列。corr は cands の一部。
    rng = np.random.RandomState(0)
    cache = []
    for cid, corr in [("c1", {10}), ("c2", {20, 21})]:
        cands = [10, 11, 12] if cid == "c1" else [20, 21, 22, 23]
        feats = rng.rand(len(cands), 3).astype(np.float32)
        cache.append({"feats": feats, "corr": set(corr), "cands": cands,
                      "variant": "original", "case_id": cid,
                      "n_input": 3, "n_output": 1, "n_sources": 1})
    return cache


def test_score_to_RK_per_case_mean_equals_aggregate():
    torch.manual_seed(0)
    cache = _tiny_cache()
    model = Reranker(3, 8)
    agg, pc = score_to_RK(model, cache, return_per_case=True)
    assert len(pc) == len(cache)
    assert all(0.0 <= v <= 1.0 for v in pc)
    assert abs(float(np.mean(pc)) - agg) < 1e-9


def test_score_to_RK_default_returns_float():
    torch.manual_seed(0)
    out = score_to_RK(Reranker(3, 8), _tiny_cache())
    assert isinstance(out, float)
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python3 -m pytest tests/test_pfi_percase.py -k score_to_RK -v`
Expected: FAIL（`return_per_case` 未対応で `TypeError` もしくは戻り値が float 単体）

- [ ] **Step 3: `score_to_RK` を修正**

`analyze_pfi.py` の `score_to_RK` を置換:
```python
def score_to_RK(model, cache, feat_override=None, return_per_case=False):
    """キャッシュ各ケースを（任意で置換後 feats で）採点し集約 Recall@K_correct を返す.

    return_per_case=True のとき (aggregate, per_case_list) を返す。
    per_case_list[i] は cache 順のケース単位 Recall@K_correct。
    """
    case_results = []
    with torch.no_grad():
        for idx, rec in enumerate(cache):
            feats = rec["feats"] if feat_override is None else feat_override[idx]
            scores = model(torch.tensor(feats, dtype=torch.float32)).numpy().ravel()
            order = sorted(range(len(rec["cands"])), key=lambda k: -scores[k])
            ranked = [rec["cands"][k] for k in order]
            ranks = compute_all_ranks(ranked, rec["corr"], miss_rank=10_000)
            cm = case_metrics(ranks)
            for kf in ("variant", "case_id", "n_input", "n_output", "n_sources"):
                cm[kf] = rec[kf]
            case_results.append(cm)
    agg = aggregate_metrics(case_results).get(METRIC, 0.0)
    if return_per_case:
        per_case = [cm.get(METRIC, 0.0) for cm in case_results]
        return agg, per_case
    return agg
```

- [ ] **Step 4: テスト成功を確認**

Run: `python3 -m pytest tests/test_pfi_percase.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: コミット**

```bash
git add analyze_pfi.py tests/test_pfi_percase.py
git commit -m "feat(pfi): score_to_RKにper-case Recall返却を追加（集約は不変）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `run()` に per-case 収集・ダンプ・smoke パラメータを追加

**Files:**
- Modify: `analyze_pfi.py`（`run()` 全体、および `if __name__` ブロック）
- 出力: `experiments/pfi_per_case.json`（新規）、`experiments/pfi_results.json`（不変）

**Interfaces:**
- Consumes: `train_and_cache`, `permute_feats`, `score_to_RK(...,return_per_case=True)`, `per_case_signal`, `sep_for_feature`, `PERM_TARGETS`, `FEATURES`。
- Produces: `run(n_seeds=None, n_perm=None)` — `n_seeds` 指定時は `DATA_SEEDS[:n_seeds]`、`n_perm` 指定時はその回数（smoke 用）。既定 None で現行（10 seed × 20置換）。副作用として `experiments/pfi_per_case.json` を出力。

- [ ] **Step 1: `run()` シグネチャと集約ループを smoke 対応に変更**

`analyze_pfi.py` の `def run():` を次の冒頭に変更し、`DATA_SEEDS`/`N_PERM` 直参照を局所変数へ差し替える。

置換1 — 関数定義とデータseed:
```python
def run(n_seeds=None, n_perm=None):
    from scipy import stats
    seeds = DATA_SEEDS[:n_seeds] if n_seeds else DATA_SEEDS
    n_perm = n_perm or N_PERM
```
置換2 — `for si, seed in enumerate(DATA_SEEDS):` → `for si, seed in enumerate(seeds):`
置換3 — 置換回数 `for pi in range(N_PERM):` → `for pi in range(n_perm):`（集約ループ内の1箇所）
置換4 — 出力メタ `"n_seeds": len(DATA_SEEDS)` → `"n_seeds": len(seeds)`、`"n_perm": N_PERM` → `"n_perm": n_perm`

- [ ] **Step 2: per-case 収集を既存ループに統合（置換compute重複なし）**

`run()` の seed ループを次の形に変更する。ポイント：base を per-case 付きで取得、seed毎に `sep` を1回計算、既存の集約ループ（`scope="case"` の枝）で得た per-case を蓄積する。

seed ループ先頭（`base = score_to_RK(model, cache)` 付近）を置換:
```python
    scopes = ["case", "global"]
    drops = {sc: {lab: [] for lab in PERM_TARGETS} for sc in scopes}
    baselines = []

    # per-case（scope="case" のみ、単独10特徴＋GROUP_var）
    PC_LABELS = [name for _, name, _ in FEATURES] + ["GROUP_var"]
    COL_OF = {name: c for c, name, _ in FEATURES}
    percase_records = []

    for si, seed in enumerate(seeds):
        model, cache = train_and_cache(seed, cases, ek, et, ev, ed, es, cl, cs)
        base, base_pc = score_to_RK(model, cache, return_per_case=True)
        baselines.append(base)
        print(f"[seed={seed}] baseline {METRIC} = {base:.4f}  (test {len(cache)} ケース)")

        # sep を seed 内で1回（単独特徴のみ）
        seps = {name: [sep_for_feature(rec["feats"], rec["cands"], rec["corr"], COL_OF[name])
                       for rec in cache]
                for name in COL_OF}
        # per-case 置換後 Recall の蓄積器
        pc_perm = {lab: [[] for _ in cache] for lab in PC_LABELS}

        for sc in scopes:
            for lab, cols in PERM_TARGETS.items():
                ds = []
                collect_pc = (sc == "case" and lab in PC_LABELS)
                for pi in range(n_perm):
                    pf = permute_feats(cache, cols, perm_seed=seed * 1000 + pi, scope=sc)
                    if collect_pc:
                        rk, rk_pc = score_to_RK(model, cache, feat_override=pf, return_per_case=True)
                        for i, v in enumerate(rk_pc):
                            pc_perm[lab][i].append(v)
                    else:
                        rk = score_to_RK(model, cache, feat_override=pf)
                    ds.append(base - rk)
                drops[sc][lab].append(float(np.mean(ds)))

        # per-case レコード生成
        for lab in PC_LABELS:
            for i, rec in enumerate(cache):
                sig = per_case_signal(base_pc[i], pc_perm[lab][i])
                percase_records.append({
                    "seed": seed, "case_id": rec["case_id"], "feature": lab,
                    "base_R": float(base_pc[i]), "perm_R_mean": sig["perm_R_mean"],
                    "drop": sig["drop"], "flip": sig["flip"], "flip_rate": sig["flip_rate"],
                    "sep": (seps[lab][i] if lab in seps else None),
                    "n_correct": len(rec["corr"]), "n_input": rec["n_input"],
                    "n_output": rec["n_output"], "n_sources": rec["n_sources"],
                    "variant": rec["variant"],
                })
```

- [ ] **Step 3: per-case ダンプを出力（集約出力の直後）**

`run()` 内、既存の `json.dump(out, open(outp, ...))`（`pfi_results.json` 出力）の直後に追加:
```python
    pc_out = {
        "config": {"setting": "A", "set_mask": list(SET_MASK), "top_k": TOP_K, "epochs": EPOCHS},
        "metric": METRIC, "n_seeds": len(seeds), "n_perm": n_perm, "scope": "case",
        "features": PC_LABELS, "records": percase_records,
    }
    pc_path = ROOT / "experiments" / "pfi_per_case.json"
    json.dump(pc_out, open(pc_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved per-case: {pc_path}  ({len(percase_records)} records)")
```

- [ ] **Step 4: `__main__` に smoke 引数を追加**

`analyze_pfi.py` 末尾を置換:
```python
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None, help="先頭N seedのみ使用（smoke）")
    ap.add_argument("--n-perm", type=int, default=None, help="置換回数（smoke）")
    a = ap.parse_args()
    run(n_seeds=a.seeds, n_perm=a.n_perm)
```

- [ ] **Step 5: smoke 実行でダンプの健全性を確認**

Run:
```bash
python3 analyze_pfi.py --seeds 1 --n-perm 2 && python3 -c "
import json
d=json.load(open('experiments/pfi_per_case.json'))
recs=d['records']; feats=set(r['feature'] for r in recs)
print('records',len(recs),'| features',len(feats))
assert d['scope']=='case'
assert 'GROUP_var' in feats and 'gDom' in feats
r=recs[0]
assert set(r)>= {'seed','case_id','feature','base_R','drop','flip','flip_rate','sep','n_correct','n_input','n_output','n_sources','variant'}
# 単独特徴は sep 非null、GROUP_var は null
assert any(r['sep'] is not None for r in recs if r['feature']=='gComp')
assert all(r['sep'] is None for r in recs if r['feature']=='GROUP_var')
print('per-case dump OK')
"
```
Expected: `per-case dump OK`（`experiments/pfi_results.json` も生成され、構造は従来どおり）

- [ ] **Step 6: コミット**

```bash
git add analyze_pfi.py
git commit -m "feat(pfi): run()にper-case収集とpfi_per_case.json出力・smoke引数を追加

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `analyze_pfi_profile.py` のプロファイル純関数

**Files:**
- Create: `analyze_pfi_profile.py`（関数のみ。`main()` は Task 5）
- Test: `tests/test_pfi_profile.py`

**Interfaces:**
- Produces:
  - `load_per_case(path) -> tuple[list[dict], dict]` — `(records, config)`。
  - `split_dependent_robust(records, feature) -> tuple[list, list]` — `base_R==1.0` を依存(`flip==1`)/頑健(`flip==0`)へ。
  - `attr_array(recs, attr) -> np.ndarray` — 非null数値配列。
  - `mannwhitney_effect(dep, rob) -> dict` — `n_dep,n_rob,mean_dep,sem_dep,mean_rob,sem_rob,mannwhitney_p,rank_biserial_r`。r は `2U/(n_dep·n_rob)-1`（正＝依存が大）。
  - `standardized_mean_diff(dep, rob) -> float` — `(mean_dep-mean_rob)/pooled_sd`。
  - `spearman_drop_sep(records, feature) -> dict` — `sep` 非null記録上の `{rho,p}`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_pfi_profile.py`:
```python
import json
import numpy as np
from analyze_pfi_profile import (
    load_per_case, split_dependent_robust, attr_array,
    mannwhitney_effect, standardized_mean_diff, spearman_drop_sep,
)


def _recs():
    # gComp: solved 8件。dep(flip=1) は n_input 大・sep 大、rob(flip=0) は小。
    # dep/rob 各4件（完全分離）→ mannwhitney 両側 p≈0.029 で安定して <0.1。
    out = []
    for flip, ni, sep, drop in [
        (1, 12, 0.50, 0.40), (1, 11, 0.45, 0.35), (1, 13, 0.55, 0.42), (1, 10, 0.48, 0.38),
        (0, 4, 0.05, 0.00), (0, 5, 0.03, 0.00), (0, 3, 0.06, 0.01), (0, 6, 0.04, 0.00),
    ]:
        out.append({"feature": "gComp", "base_R": 1.0, "flip": flip,
                    "n_input": ni, "sep": sep, "drop": drop, "n_correct": 1})
    # base_R<1.0 は依存/頑健から除外される
    out.append({"feature": "gComp", "base_R": 0.5, "flip": 0,
                "n_input": 9, "sep": 0.20, "drop": 0.20, "n_correct": 2})
    return out


def test_split_dependent_robust_conditions_on_solved():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    assert len(dep) == 4 and len(rob) == 4  # base_R<1.0 は除外


def test_mannwhitney_effect_sign_positive_when_dep_larger():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    e = mannwhitney_effect(attr_array(dep, "n_input"), attr_array(rob, "n_input"))
    assert e["mean_dep"] > e["mean_rob"]
    assert e["rank_biserial_r"] > 0.9   # 完全分離に近い
    assert e["mannwhitney_p"] < 0.1


def test_standardized_mean_diff_positive():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    smd = standardized_mean_diff(attr_array(dep, "sep"), attr_array(rob, "sep"))
    assert smd > 1.0


def test_spearman_drop_sep_positive():
    s = spearman_drop_sep(_recs(), "gComp")
    assert s["rho"] > 0.5


def test_load_per_case(tmp_path):
    doc = {"config": {"setting": "A"}, "records": [{"feature": "gComp", "base_R": 1.0}]}
    p = tmp_path / "pc.json"
    p.write_text(json.dumps(doc))
    recs, cfg = load_per_case(p)
    assert cfg["setting"] == "A" and recs[0]["feature"] == "gComp"
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python3 -m pytest tests/test_pfi_profile.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'analyze_pfi_profile'`）

- [ ] **Step 3: 純関数を実装**

`analyze_pfi_profile.py`（新規）:
```python
"""PFIケース別依存プロファイル：置換で正解→不正解に転じるケースを属性で特徴づける.

入力 : experiments/pfi_per_case.json（analyze_pfi.py が出力）
出力 : experiments/pfi_profile_stats.json, docs/figures/fig_pfi_dependence_profile.png/.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent
SETAWARE = [("gComp", "＋補完性", "#1f77b4"),
            ("gCoh", "＋一貫性", "#2ca02c"),
            ("gDom", "＋ドメイン一致", "#d62728")]
ATTRS = ["n_correct", "n_input", "n_output", "n_sources", "sep"]


def load_per_case(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["records"], d.get("config", {})


def split_dependent_robust(records, feature):
    solved = [r for r in records if r["feature"] == feature and r["base_R"] == 1.0]
    dep = [r for r in solved if r["flip"] == 1]
    rob = [r for r in solved if r["flip"] == 0]
    return dep, rob


def attr_array(recs, attr):
    return np.array([r[attr] for r in recs if r.get(attr) is not None], dtype=float)


def mannwhitney_effect(dep, rob):
    res = {"n_dep": int(dep.size), "n_rob": int(rob.size),
           "mean_dep": float(dep.mean()) if dep.size else float("nan"),
           "sem_dep": float(dep.std(ddof=1) / np.sqrt(dep.size)) if dep.size > 1 else 0.0,
           "mean_rob": float(rob.mean()) if rob.size else float("nan"),
           "sem_rob": float(rob.std(ddof=1) / np.sqrt(rob.size)) if rob.size > 1 else 0.0,
           "mannwhitney_p": float("nan"), "rank_biserial_r": float("nan")}
    if dep.size and rob.size:
        U, p = stats.mannwhitneyu(dep, rob, alternative="two-sided")
        res["mannwhitney_p"] = float(p)
        res["rank_biserial_r"] = float(2.0 * U / (dep.size * rob.size) - 1.0)
    return res


def standardized_mean_diff(dep, rob):
    n1, n2 = dep.size, rob.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = np.sqrt(((n1 - 1) * dep.var(ddof=1) + (n2 - 1) * rob.var(ddof=1)) / (n1 + n2 - 2))
    return float((dep.mean() - rob.mean()) / sp) if sp > 0 else 0.0


def spearman_drop_sep(records, feature):
    recs = [r for r in records if r["feature"] == feature and r.get("sep") is not None]
    if len(recs) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(recs)}
    rho, p = stats.spearmanr([r["sep"] for r in recs], [r["drop"] for r in recs])
    return {"rho": float(rho), "p": float(p), "n": len(recs)}
```

- [ ] **Step 4: テスト成功を確認**

Run: `python3 -m pytest tests/test_pfi_profile.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: コミット**

```bash
git add analyze_pfi_profile.py tests/test_pfi_profile.py
git commit -m "feat(pfi): 依存/頑健プロファイルの統計純関数を追加

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `analyze_pfi_profile.py` の `main()`（統計JSON＋図）

**Files:**
- Modify: `analyze_pfi_profile.py`（`main()` と図関数を追加、`if __name__` ブロック）
- 出力: `experiments/pfi_profile_stats.json`, `docs/figures/fig_pfi_dependence_profile.png` / `.pdf`

**Interfaces:**
- Consumes: Task 4 の全純関数、`SETAWARE`, `ATTRS`。
- Produces: `sep_profile(records, feature, n_bins=5) -> tuple[list,list,list]`（`xs, ys, es`）、`build_stats(records) -> dict`、`main()`。

- [ ] **Step 1: 失敗するテストを書く（`sep_profile` と `build_stats`）**

`tests/test_pfi_profile.py` に追記:
```python
from analyze_pfi_profile import sep_profile, build_stats


def test_sep_profile_bins_increase():
    recs = [{"feature": "gComp", "base_R": 1.0, "flip": 1,
             "sep": s, "drop": s, "n_input": 5, "n_correct": 1,
             "n_output": 1, "n_sources": 1}
            for s in np.linspace(0, 1, 40)]
    xs, ys, es = sep_profile(recs, "gComp", n_bins=4)
    assert len(xs) == 4
    assert ys[-1] > ys[0]            # sep が大きいビンほど drop 大
    assert all(e >= 0 for e in es)


def test_build_stats_has_features_and_attrs():
    recs = _recs()
    st = build_stats(recs)
    assert "gComp" in st["features"]
    g = st["features"]["gComp"]
    assert "n_input" in g["attrs"] and "sep" in g["attrs"]
    assert "spearman_drop_sep" in g
    assert g["n_dependent"] == 4 and g["n_robust"] == 4
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python3 -m pytest tests/test_pfi_profile.py -k "sep_profile or build_stats" -v`
Expected: FAIL（`ImportError: cannot import name 'sep_profile'`）

- [ ] **Step 3: `sep_profile`・`build_stats`・図・`main` を実装**

`analyze_pfi_profile.py` に追記:
```python
def sep_profile(records, feature, n_bins=5):
    """sep の分位ビンごとの平均 drop ±SEM（全ケース対象）."""
    recs = [r for r in records if r["feature"] == feature and r.get("sep") is not None]
    if len(recs) < n_bins:
        return [], [], []
    seps = np.array([r["sep"] for r in recs])
    drops = np.array([r["drop"] for r in recs])
    edges = np.quantile(seps, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    xs, ys, es = [], [], []
    for b in range(n_bins):
        m = (seps >= edges[b]) & (seps < edges[b + 1])
        if m.sum() == 0:
            continue
        xs.append(float(0.5 * (edges[b] + edges[b + 1])))
        ys.append(float(drops[m].mean()))
        es.append(float(drops[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else 0.0)
    return xs, ys, es


def build_stats(records):
    feats = {}
    all_feats = sorted(set(r["feature"] for r in records))
    for name in all_feats:
        dep, rob = split_dependent_robust(records, name)
        solved = len(dep) + len(rob)
        entry = {"n_dependent": len(dep), "n_robust": len(rob),
                 "flip_fraction_of_solved": (len(dep) / solved) if solved else 0.0,
                 "spearman_drop_sep": spearman_drop_sep(records, name), "attrs": {}}
        for attr in ATTRS:
            entry["attrs"][attr] = mannwhitney_effect(attr_array(dep, attr), attr_array(rob, attr))
        feats[name] = entry
    return {"metric": "Recall@K_correct", "n_features": len(all_feats), "features": feats}


def _make_figure(records, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import japanize_matplotlib  # noqa
    except Exception:
        for fam in ["Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "AppleGothic"]:
            try:
                plt.rcParams["font.family"] = fam; break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # (a) 分離度ビン × 平均drop
    for name, label, col in SETAWARE:
        xs, ys, es = sep_profile(records, name, n_bins=5)
        if xs:
            axes[0].errorbar(xs, ys, yerr=es, marker="o", lw=2.2, capsize=3, color=col, label=label)
    axes[0].axhline(0, color="#bbbbbb", lw=1, zorder=0)
    axes[0].set_title("(a) 分離度別の期待Recall低下")
    axes[0].set_xlabel("分離度 sep（正解式−不正解式の特徴値差）")
    axes[0].set_ylabel("置換によるRecall@K_correctの期待低下")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=9, loc="upper left")

    # (b) gComp/gCoh の 依存−頑健 標準化平均差（属性別・横棒）
    bar_feats = [("gComp", "#1f77b4"), ("gCoh", "#2ca02c")]
    y = np.arange(len(ATTRS)); h = 0.38
    for bi, (name, col) in enumerate(bar_feats):
        dep, rob = split_dependent_robust(records, name)
        smds = [standardized_mean_diff(attr_array(dep, a), attr_array(rob, a)) for a in ATTRS]
        smds = [0.0 if (s != s) else s for s in smds]  # nan→0
        axes[1].barh(y + (bi - 0.5) * h, smds, height=h, color=col, label=name)
    axes[1].axvline(0, color="#888888", lw=1)
    axes[1].set_yticks(y); axes[1].set_yticklabels(ATTRS)
    axes[1].set_title("(b) 依存ケースを特徴づける属性")
    axes[1].set_xlabel("標準化平均差（依存−頑健）")
    axes[1].grid(alpha=0.25, axis="x"); axes[1].legend(fontsize=9)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/pfi_per_case.json")
    ap.add_argument("--out-json", default="experiments/pfi_profile_stats.json")
    ap.add_argument("--out-fig", default="docs/figures/fig_pfi_dependence_profile.png")
    a = ap.parse_args()
    records, config = load_per_case(ROOT / a.input)
    stats_out = build_stats(records)
    stats_out["config"] = config
    json.dump(stats_out, open(ROOT / a.out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Saved stats: {a.out_json}")
    _make_figure(records, ROOT / a.out_fig)
    print(f"Saved figure: {a.out_fig}")
    # コンソール要約
    for name, label, _ in SETAWARE:
        g = stats_out["features"].get(name, {})
        s = g.get("spearman_drop_sep", {})
        print(f"{name:6s} 依存={g.get('n_dependent')}/{g.get('n_dependent',0)+g.get('n_robust',0)} "
              f"Spearman(drop,sep) rho={s.get('rho')} p={s.get('p')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テスト成功を確認**

Run: `python3 -m pytest tests/test_pfi_profile.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: smoke ダンプで統計JSON＋図が生成されることを確認**

（Task 3 Step 5 の smoke で `experiments/pfi_per_case.json` が既にある前提。無ければ先に実行）

Run:
```bash
python3 analyze_pfi_profile.py && python3 -c "
import json, os
st=json.load(open('experiments/pfi_profile_stats.json'))
assert 'gComp' in st['features'] and 'gDom' in st['features']
assert 'sep' in st['features']['gComp']['attrs']
assert os.path.exists('docs/figures/fig_pfi_dependence_profile.png')
assert os.path.exists('docs/figures/fig_pfi_dependence_profile.pdf')
print('profile stats + figure OK')
"
```
Expected: `profile stats + figure OK`

- [ ] **Step 6: コミット**

```bash
git add analyze_pfi_profile.py tests/test_pfi_profile.py
git commit -m "feat(pfi): 依存プロファイルの統計JSONと2パネル図を出力するmainを追加

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 本番実行（10 seed）と成功基準の確認

**Files:**
- 出力更新: `experiments/pfi_per_case.json`, `experiments/pfi_results.json`, `experiments/pfi_profile_stats.json`, `docs/figures/fig_pfi_dependence_profile.png/.pdf`
- Modify（任意）: `development_log.tex`（devlog 規約に沿って1段落追記）

**Interfaces:**
- Consumes: 完成した `analyze_pfi.py` / `analyze_pfi_profile.py`。

- [ ] **Step 1: 本番PFIを実行（10 seed × 20置換。長時間）**

Run: `python3 analyze_pfi.py`
Expected: `Saved: experiments/pfi_results.json` と `Saved per-case: experiments/pfi_per_case.json (... records)`。集約表のコンソール出力が従来と同様に出る。

- [ ] **Step 2: プロファイルを生成**

Run: `python3 analyze_pfi_profile.py`
Expected: `Saved stats:` / `Saved figure:` と、gComp/gCoh/gDom の依存件数・Spearman の要約行。

- [ ] **Step 3: 成功基準を検証**

Run:
```bash
python3 -c "
import json
st=json.load(open('experiments/pfi_profile_stats.json'))['features']
gd=st['gDom']; gc=st['gComp']; gh=st['gCoh']
print('gDom  依存件数=',gd['n_dependent'],' flip率=',round(gd['flip_fraction_of_solved'],4))
print('gComp 依存件数=',gc['n_dependent'],' Spearman(drop,sep) rho=',gc['spearman_drop_sep']['rho'])
print('gCoh  依存件数=',gh['n_dependent'],' Spearman(drop,sep) rho=',gh['spearman_drop_sep']['rho'])
# 成功基準（符号の健全性チェック。値は実測を正直に採用）
assert gd['n_dependent'] <= gc['n_dependent'], 'gDomの依存がgCompを超える→要考察'
print('sanity: gDom依存 <= gComp依存 OK')
"
```
Expected: gDom の依存件数がほぼ0で gComp/gCoh より小さい。gComp/gCoh の Spearman ρ の符号を確認（正なら「分離度が高いほど低下大」＝機構と整合）。**値が仮説と食い違う場合は改変せず、そのまま結果として記録し考察する。**

- [ ] **Step 4: 図の作文ルール準拠を目視確認**

`docs/figures/fig_pfi_dependence_profile.png` を開き、(1) 両パネルにエラーバー、(2) 軸ラベルが定義済み語（sep/期待低下/標準化平均差）、(3) 凡例あり、(4) 主観語なし、を確認。不足あれば `_make_figure` を修正して再実行・再コミット。

- [ ] **Step 5: 全テスト通過を確認**

Run: `python3 -m pytest tests/test_pfi_percase.py tests/test_pfi_profile.py -v`
Expected: 全 PASS（14 passed）

- [ ] **Step 6: 成果物をコミット**

```bash
git add experiments/pfi_per_case.json experiments/pfi_results.json \
        experiments/pfi_profile_stats.json \
        docs/figures/fig_pfi_dependence_profile.png docs/figures/fig_pfi_dependence_profile.pdf
git commit -m "feat(pfi): ケース別依存プロファイルの本番結果と図を追加（設定A・10seed）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review（記入済）

**Spec coverage:**
- §3 per-case 定義（drop/flip/依存・頑健）→ Task 1（純関数）＋ Task 3（収集）。
- §3 scope="case" 限定・GROUP_var 併走 → Task 3 の `PC_LABELS`・`collect_pc`。
- §4 属性軸＋sep・Mann–Whitney・効果量・Spearman → Task 4。
- §5.1 score_to_RK 拡張・集約不変・sep はキャッシュから → Task 2・Task 3。
- §5.2 profile スクリプト・図2パネル・和文フォント → Task 5。
- §6 スキーマ（pfi_per_case / pfi_profile_stats）→ Task 3 Step 3・Task 5 build_stats。
- §8 成功基準（gDom≈0・Spearman符号・集約不変）→ Task 6 Step 3、Task 3 Step 5。
- §9 リスク（冗長性=GROUP_var、分数Recall=flip限定+連続drop、seed重複=プール）→ 各タスクに反映。

**Placeholder scan:** コード段は実コードのみ。TODO/TBDなし。

**Type consistency:** `per_case_signal`/`sep_for_feature`/`score_to_RK(...,return_per_case=)`/`split_dependent_robust`/`mannwhitney_effect`/`standardized_mean_diff`/`spearman_drop_sep`/`sep_profile`/`build_stats` の名称・引数はタスク間で一致。レコードのキー（`feature,base_R,flip,drop,sep,n_input,...`）は Task 3 生成と Task 4/5 消費で一致。

**非スコープの確認:** 設定B/DAE・発見モデル・用量反応の完全連結は spec 通り除外。
