"""分割の単位をファイル（出典 ID）から文献に直し、提案手法を学習し直して、ファイル単位の分割の値と LLM の値と比べる.

背景（2026-11-09）：出典 ID は PDF のファイル名で、同じ本（Seborg の本全体と第 2 章、Corriou の 2 版と第 1 章の抜き出し）や、
同じ論文（OpenAlex の収集で 2〜3 分野の検索に重なった 35 報）が、別々のファイルとして入っている。分割は出典 ID 単位なので、
同じ文献の別のファイルから作ったケースが訓練とテストに分かれていた（設定 A で Seborg は 10 通りの分割のうち 7 通り）。
ここでは同じ文献のファイルを 1 つにまとめ、文献単位で層化分割し直し、提案手法（reranker-10S、第 1 段 (0.3, 0.7)・N1 = 2,000、
不正解の選び方 hard＝10 通りの分割すべてで検証用のケースが選んだ設定）を同じ設定・同じ乱数で学習し直す。
LLM の答えはケースごとに保存してあるので取り直さず、新しいテストケースのうち答えがあるものだけで比べる。

同じ文献とみなす規則（docs サブコマンド）:
  1. ファイル名から収集の番号（01_、FB05_、FB2_10_ など）と .pdf を除いた名前が同じ（年が頭に付く初期のファイルは年を後ろに回す）。
  2. 文献集めの記録（docs/paper_collection/*.json）で DOI が同じ。
  3. 手で挙げた教科書（BOOK_GROUPS）：Seborg（2 ファイル）、Corriou（2004 年の初版の第 1 章も含めて 4 ファイル）、
     Marlin（本全体と第 3・4 章の 3 ファイル。式の重なりはほぼないが、同じ本なので 1 つにする）。
  生成した基礎式集（handbook_X と handbook_X_adv）は文献でなく、式も重ならないので、今までどおり別の単位のまま。

手順:
  python3 doc_split_rerun.py docs                       # → experiments/doc_split/source_documents.json
  python3 doc_split_rerun.py splits                     # → experiments/doc_split/splits.json
  python3 doc_split_rerun.py train --setting A          # → experiments/doc_split/raw/A__{種}.json（1 分割あたり約 70 秒）
  python3 doc_split_rerun.py compare                    # → experiments/doc_split/compare.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_llm_db_baseline as rl  # noqa: E402
from two_stage_query_conditioned import load_equations, get_src, eq_key, norm, case_src, in_vars, out_vars  # noqa: E402

OUT = ROOT / "experiments" / "doc_split"
DOCS_JSON = OUT / "source_documents.json"
SPLITS_JSON = OUT / "splits.json"
RAW = OUT / "raw"
COMPARE_JSON = OUT / "compare.json"
PAPER_META = [ROOT / "docs" / "paper_collection" / f for f in (
    "fetched_papers_metadata.json", "fetched_papers_fallback_metadata.json",
    "fetched_papers_fallback2_metadata.json", "fetched_papers_fallback3_metadata.json")]
MODE = "reranker-10S"
LLM_RESULTS = {"Opus 5.5": "results_claude-opus-5-5__eqd__ex-none__high__v4.json",
               "Haiku 5.5": "results_claude-haiku-5-5__eqd__ex-none__high__v4.json"}

# 手で挙げた教科書：同じ本が別々のファイルとして入っているもの（docs/appendix_materials/papers.csv の推定と、式の重なりで確認）
BOOK_GROUPS = {
    "Seborg_Process_Dynamics_and_Control": [
        "SEBORG_3rd_Edition_Process_Dynamics_and.pdf", "SEBORG_Process_Dynamic_and_Control_Chap_2.pdf"],
    "Corriou_Process_Control": [
        "2018_Book_ProcessControl.pdf", "2018_Book_ProcessControl_Chapter1.pdf",
        "2018_Book_ProcessControl_Chapter1_6.pdf", "9781852337766-c1.pdf"],
    "Marlin_Process_Control": [
        "Process_Control_Designing_Processes_and_Control_Systems_for_Dynamic_Performance.pdf",
        "Process_Control_Chapter3.pdf", "Process_Control_Chapter4.pdf"],
}

_PREFIX = re.compile(r"^(?:FB\d*_\d+_|FB\d+_|\d{1,2}_)")   # 収集の番号：01_、FB05_、FB2_10_、FB3_01_
_YEAR_FIRST = re.compile(r"^(\d{4})_(.+)$")


# ---- 同じ文献のファイルをまとめる ------------------------------------------------------------

def source_stem(name: str) -> str:
    """ファイル名から収集の番号と .pdf を除いた、文献の名前（小文字）。年が頭に付く名前は年を後ろに回す（教科書は除く）."""
    t = re.sub(r"\.pdf$", "", _PREFIX.sub("", name), flags=re.IGNORECASE)
    m = _YEAR_FIRST.match(t)
    if m and not t.startswith("2018_Book"):
        t = f"{m.group(2)}_{m.group(1)}"
    return t.lower()


def group_sources(files, doi_groups, book_groups) -> dict:
    """ファイルを文献にまとめる：{文献の名前: [ファイル（昇順）]}。名前が同じ・DOI が同じ・同じ本、のどれかでつながれば同じ文献.

    1 ファイルの文献の名前はファイル名そのまま。複数ファイルの文献は、本なら BOOK_GROUPS の名前、そうでなければ最初のファイルの stem。
    """
    files = list(files)
    present = set(files)
    parent = {f: f for f in files}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    by_stem = defaultdict(list)
    for f in files:
        by_stem[source_stem(f)].append(f)
    for members in by_stem.values():
        for g in members[1:]:
            union(members[0], g)
    for grp in doi_groups:
        grp = [f for f in grp if f in present]
        for g in grp[1:]:
            union(grp[0], g)
    book_name = {}
    for name, grp in book_groups.items():
        grp = [f for f in grp if f in present]
        for g in grp[1:]:
            union(grp[0], g)
        for g in grp:
            book_name[g] = name
    members_of = defaultdict(list)
    for f in files:
        members_of[find(f)].append(f)
    docs = {}
    for ms in members_of.values():
        ms = sorted(ms)
        if len(ms) == 1:
            name = ms[0]
        else:
            names = {book_name[f] for f in ms if f in book_name}
            name = sorted(names)[0] if names else source_stem(ms[0])
        while name in docs:          # 名前の衝突は避ける（起きないはずだが、起きたら気づけるように）
            name += "_"
        docs[name] = ms
    return dict(sorted(docs.items()))


def doc_labels(case_srcs, file_to_doc: dict) -> list:
    """ケースの出典（ファイル名）を文献の名前に置き換える。対応がないものはそのまま."""
    return [file_to_doc.get(s, s) for s in case_srcs]


def doi_groups_from_metadata() -> list:
    """文献集めの記録で DOI が同じファイル名の組（DOI がない記録は使わない）."""
    by_doi = defaultdict(set)
    for p in PAPER_META:
        if not p.exists():
            continue
        for x in json.loads(p.read_text(encoding="utf-8")).get("papers", []):
            if x.get("filename") and x.get("doi"):
                by_doi[x["doi"].strip().lower()].add(Path(x["filename"]).name)
    return [sorted(v) for v in by_doi.values() if len(v) > 1]


# ---- 集計 ----------------------------------------------------------------------------------

def llm_case_means(per_case: list) -> dict:
    """LLM の結果（ケース × 回）を、ケースごとの回の平均にする。*_equiv の列があればそれも."""
    keys = [k for k in ("recall@20", "AP@20", "recall@20_equiv", "AP@20_equiv") if per_case and k in per_case[0]]
    by = defaultdict(list)
    for r in per_case:
        by[r["case_id"]].append(r)
    return {c: {"n_runs": len(rs), **{k: float(np.mean([r[k] for r in rs])) for k in keys}} for c, rs in by.items()}


def mean_sd_over_seeds(per_seed: dict) -> dict:
    """分割（種）ごとの値の、分割間の平均と標準偏差（ddof=1）."""
    a = np.asarray(list(per_seed.values()), dtype=float)
    if a.size == 0:      # その種類のケースが 1 つの分割にもない（設定 B の単一出典・出典横断）
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "per_seed": {}}
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "per_seed": {str(k): float(v) for k, v in per_seed.items()}}


def case_kind(c: dict) -> str:
    v = c.get("variant_type", "")
    return "single" if v == "original" else "cross" if v.startswith("multisource_") else "synth" if v.startswith("dae_") else v


# ---- docs ------------------------------------------------------------------------------------

def cmd_docs(args) -> None:
    from set_aware_reranker import latex_key
    eqs = load_equations()
    n_eq = Counter(get_src(e) for e in eqs)
    files = sorted(n_eq)
    doi_groups = doi_groups_from_metadata()
    docs = group_sources(files, doi_groups, BOOK_GROUPS)
    stem_multi = {s for s, v in Counter(source_stem(f) for f in files).items() if v > 1}
    doi_files = {f for g in doi_groups for f in g}
    book_files = {f for g in BOOK_GROUPS.values() for f in g}
    lat = defaultdict(lambda: defaultdict(set))   # doc -> latex -> files
    file_to_doc = {f: d for d, ms in docs.items() for f in ms}
    for e in eqs:
        k = latex_key(e.get("equation"))
        if k:
            lat[file_to_doc[get_src(e)]][k].add(get_src(e))
    out_docs = {}
    for d, ms in docs.items():
        ev = []
        if len(ms) > 1:
            if any(source_stem(f) in stem_multi for f in ms):
                ev.append("same name after removing the collection prefix")
            if len(doi_files & set(ms)) > 1:
                ev.append("same DOI in docs/paper_collection")
            if set(ms) & book_files:
                ev.append("hand-listed textbook (BOOK_GROUPS)")
        out_docs[d] = {"files": ms, "n_equations": {f: n_eq[f] for f in ms}, "evidence": ev,
                       "n_latex_in_2_or_more_files": sum(1 for v in lat[d].values() if len(v) > 1)}
    multi = {d: v for d, v in out_docs.items() if len(v["files"]) > 1}
    out = {
        "note": "同じ文献が別々のファイル（出典 ID）として入っているものをまとめた対応。分割の単位に使う。DB の式は変えない。",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": __doc__.split("同じ文献とみなす規則（docs サブコマンド）:")[1].split("手順:")[0].strip(),
        "n_files": len(files), "n_documents": len(docs),
        "n_documents_with_several_files": len(multi),
        "n_files_in_them": sum(len(v["files"]) for v in multi.values()),
        "n_equations_in_them": sum(sum(v["n_equations"].values()) for v in multi.values()),
        "documents": out_docs, "file_to_doc": file_to_doc,
        "data_sha256": {rl.EQ_JSON.name: rl.sha256(rl.EQ_JSON)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"files {len(files)} -> documents {len(docs)}; documents with several files {len(multi)} "
          f"(files {out['n_files_in_them']}, equations {out['n_equations_in_them']} of {len(eqs)})")
    for d, v in sorted(multi.items(), key=lambda kv: -sum(kv[1]["n_equations"].values())):
        print(f"  {d[:60]:60s} files {len(v['files'])}  eqs {sum(v['n_equations'].values()):4d}  "
              f"same LaTeX in 2+ files {v['n_latex_in_2_or_more_files']:3d}  [{'; '.join(v['evidence'])}]")
    print(f"wrote {DOCS_JSON}")


def load_file_to_doc() -> dict:
    return json.loads(DOCS_JSON.read_text(encoding="utf-8"))["file_to_doc"]


# ---- splits ----------------------------------------------------------------------------------

def cmd_splits(args) -> None:
    """run_llm_db_baseline.cmd_splits と同じ手順（同じ外す一覧、同じ特徴量、同じ乱数）で、出典の印だけを文献にして層化分割する."""
    from set_aware_reranker import stratified_src_split
    from two_stage_query_conditioned import load_cases
    if SPLITS_JSON.exists() and not args.force:
        sys.exit(f"{SPLITS_JSON} はもうある。作り直すなら --force")
    file_to_doc = load_file_to_doc()
    fixed = json.loads(rl.SPLITS_JSON.read_text(encoding="utf-8"))
    exclude_file = fixed["provenance"]["exclude_file"]
    excluded = set(json.loads((ROOT / exclude_file).read_text(encoding="utf-8"))["excluded_case_ids"])
    eqs = load_equations()
    ek = [eq_key(e) for e in eqs if eq_key(e)]
    ki = {k: i for i, k in enumerate(ek)}
    all_cases = load_cases()
    out = {"splits": {}, "n_cases": {}, "n_documents": {}, "check": {}}
    for setting, wanted in rl.SETTINGS.items():
        cases = [c for c in all_cases if rl.variant_matches(c.get("variant_type", ""), wanted) and c["case_id"] not in excluded]
        cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
        cs = doc_labels([case_src(c) for c in cases], file_to_doc)
        # set_aware_reranker.run_mode（split_mode="stratified"）と同じ特徴量（4 つ目はファイル数のまま）
        feats = [(len(cl[i]), len(in_vars(cases[i])), len(out_vars(cases[i])),
                  len({ek[j].split("__")[0] for j in cl[i]})) for i in range(len(cases))]
        out["n_cases"][setting] = len(cases)
        out["n_documents"][setting] = len(set(cs))
        out["splits"][setting], out["check"][setting] = {}, {}
        old = fixed["splits"][setting]
        for seed in rl.ALL_SEEDS:
            sp = stratified_src_split(cs, feats, seed)
            parts = {part: [cases[i]["case_id"] for i, s in enumerate(cs) if s in sp[part] and cl[i]]
                     for part in ("train", "val", "test")}
            assert not (sp["train"] & sp["test"]) and not (sp["val"] & sp["test"]) and not (sp["train"] & sp["val"])
            out["splits"][setting][str(seed)] = parts
            same_as_old = parts["test"] == old[str(seed)]["test"]
            out["check"][setting][str(seed)] = {"n_documents": {p: len(sp[p]) for p in ("train", "val", "test")},
                                                "test_identical_to_file_split": same_as_old}
            print(f"{setting} seed={seed}: train {len(parts['train'])} val {len(parts['val'])} test {len(parts['test'])}"
                  f"  documents {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}  same test as file split: {same_as_old}")
    out["provenance"] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": rl._git("rev-parse", "HEAD"),
        "unit": "document（experiments/doc_split/source_documents.json の file_to_doc で、ケースの出典ファイルを文献に置き換えた）",
        "source_documents_sha256": rl.sha256(DOCS_JSON),
        "exclude_file": exclude_file, "excluded_case_ids": sorted(excluded),
        "data_sha256": {p.name: rl.sha256(p) for p in (rl.CASES_JSON, rl.EQ_JSON)},
        "rule": "run_llm_db_baseline.cmd_splits と同じ：stratified_src_split を同じ特徴量タプル（正解式数・入力変数数・出力変数数・"
                "正解式の出典ファイル数）と同じ種で呼ぶ。違いは出典の印がファイルでなく文献であることだけ。"
                "train/val/test は正解式が DB にあるケースだけ。",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {SPLITS_JSON}")


def load_doc_splits() -> dict:
    return json.loads(SPLITS_JSON.read_text(encoding="utf-8"))["splits"]


# ---- train -----------------------------------------------------------------------------------

def cmd_train(args) -> None:
    import set_aware_reranker as sar
    from dump_proposed_top20 import inputs
    file_to_doc = load_file_to_doc()
    splits = load_doc_splits()
    seeds = rl.ALL_SEEDS if args.seeds == "all" else [int(s) for s in args.seeds.split(",")]
    RAW.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        f_out = RAW / f"{args.setting}__{seed}.json"
        if f_out.exists() and not args.force:
            print(f"skip {args.setting} {seed}（もうある）", flush=True)
            continue
        sel = rl._selected_ref(args.setting, seed)
        ref_f = rl.REF_RUN / "raw" / "select" / f"{args.setting}_{sel['w']}_{sel['neg']}__{seed}.json"
        cfg = json.loads(ref_f.read_text(encoding="utf-8"))["config"]
        cases, ek, et, ev, ed, es, ex, cl, cs = inputs(cfg)
        cs_doc = doc_labels(cs, file_to_doc)
        t0 = time.perf_counter()
        res = sar.run_mode(
            MODE, [seed], cases, ek, et, ev, ed, es, cl, cs_doc, cfg["top_k"], cfg["epochs"], cfg["lr"],
            save_per_case=True, hidden_dim=cfg["hidden_dim"], margin=cfg["margin"], batch_size=cfg["batch_size"],
            n_neg_samples=cfg["n_neg_samples"], weight_decay=cfg["weight_decay"], loss_type=cfg["loss"],
            hard_neg=cfg["hard_neg"], greedy=cfg["greedy"], split_mode=cfg["split"], train_greedy=cfg["train_greedy"],
            greedy_train_cap=cfg["greedy_train_cap"], stop_dof=cfg["stop_dof"], stage1_w_text=cfg["stage1_w_text"],
            stage1_w_var=cfg["stage1_w_var"], fast=cfg["fast"], neg_pool=cfg["neg_pool"], eval_val=cfg["eval_val"],
            equiv_ids=ex if cfg.get("equiv") else None, equiv_src=es if cfg.get("equiv") else None)
        test_ids = [r["case_id"] for r in res["per_case"] if r.get("split", "test") == "test"]
        if test_ids != splits[args.setting][str(seed)]["test"]:
            sys.exit(f"{args.setting} {seed}: 学習し直しのテストケースが splits.json と食い違う "
                     f"({len(test_ids)} vs {len(splits[args.setting][str(seed)]['test'])})")
        out = {"config": {**cfg, "seed_list": str(seed), "output": str(f_out), "split_unit": "document",
                          "source_documents": str(DOCS_JSON.relative_to(ROOT)), "config_from": str(ref_f.relative_to(ROOT))},
               "results": {MODE: res}}
        f_out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {f_out}: test {len(test_ids)} cases, {time.perf_counter() - t0:.0f} s", flush=True)


# ---- compare ---------------------------------------------------------------------------------

def _test_records(path: Path, seed: int) -> list:
    d = json.loads(path.read_text(encoding="utf-8"))
    return rl.split_test_records(d["results"][MODE]["per_case"], seed)


def _case_metrics(rec: dict) -> dict:
    m = rl.metrics_from_ranks(rec["ranks"])
    out = {"recall@20": m["recall@20"], "AP@20": m["AP@20"]}
    if "ranks_equiv" in rec:
        me = rl.metrics_from_ranks(rec["ranks_equiv"])
        out.update({"recall@20_equiv": me["recall@20"], "AP@20_equiv": me["AP@20"]})
    return out


def _means(per_case: dict, ids, key: str) -> float:
    return float(np.mean([per_case[c][key] for c in ids]))


def cmd_compare(args) -> None:
    cases = rl.load_case_index()
    new_splits, old_splits = load_doc_splits(), rl.load_splits()
    llm = {name: llm_case_means(json.loads((rl.OUT_DIR / f).read_text(encoding="utf-8"))["per_case"])
           for name, f in LLM_RESULTS.items() if (rl.OUT_DIR / f).exists()}
    metrics = ("recall@20", "AP@20", "recall@20_equiv", "AP@20_equiv")
    label = {"recall@20": "Recall@20", "AP@20": "MAP@20", "recall@20_equiv": "Recall@20 (same eq. elsewhere = correct)",
             "AP@20_equiv": "MAP@20 (same eq. elsewhere = correct)"}
    out = {"note": "proposed: reranker-10S, (0.3, 0.7), N1 = 2,000, hard negatives, 15 epochs（検証で選ばれた設定を固定）。"
                   "file split: experiments/kato1021/raw/select（既存の値）。document split: experiments/doc_split/raw。"
                   "LLM: ケースごとの 3 回の平均。LLM との比較は、文献単位の分割のテストケースのうち LLM の答えがあるものだけで、"
                   "同じケースで提案手法と比べる。検定はしない。",
           "settings": {}}
    for setting in ("A", "B"):
        seeds = [s for s in rl.ALL_SEEDS if (RAW / f"{setting}__{s}.json").exists()]
        if not seeds:
            continue
        sel = rl._selected_ref(setting, seeds[0])
        new_pc, old_pc = {}, {}
        for s in seeds:
            new_pc[s] = {r["case_id"]: _case_metrics(r) for r in _test_records(RAW / f"{setting}__{s}.json", s)}
            old_pc[s] = {r["case_id"]: _case_metrics(r) for r in _test_records(
                rl.REF_RUN / "raw" / "select" / f"{setting}_{sel['w']}_{sel['neg']}__{s}.json", s)}
        res = {"seeds": seeds, "n_test": {"document_split": {str(s): len(new_pc[s]) for s in seeds},
                                           "file_split": {str(s): len(old_pc[s]) for s in seeds}},
               "proposed": {}, "by_kind": {}, "vs_llm": {}}
        for key in metrics:
            res["proposed"][key] = {
                "document_split": mean_sd_over_seeds({s: _means(new_pc[s], new_pc[s], key) for s in seeds}),
                "file_split": mean_sd_over_seeds({s: _means(old_pc[s], old_pc[s], key) for s in seeds})}
        for kind in ("single", "cross", "synth"):
            res["by_kind"][kind] = {}
            for key in ("recall@20", "AP@20"):
                d = {}
                for name, pc in (("document_split", new_pc), ("file_split", old_pc)):
                    vals = {s: _means(pc[s], [c for c in pc[s] if case_kind(cases[c]) == kind], key)
                            for s in seeds if any(case_kind(cases[c]) == kind for c in pc[s])}
                    if vals:
                        d[name] = mean_sd_over_seeds(vals)
                res["by_kind"][kind][key] = d
            res["by_kind"][kind]["n_test_document_split"] = {
                str(s): sum(1 for c in new_pc[s] if case_kind(cases[c]) == kind) for s in seeds}
        for name, pc_llm in llm.items():
            cov = {s: [c for c in new_pc[s] if c in pc_llm] for s in seeds}
            entry = {"n_test_with_llm_answer": {str(s): len(cov[s]) for s in seeds},
                     "n_test": {str(s): len(new_pc[s]) for s in seeds}}
            for key in metrics:
                a = {s: _means(pc_llm, cov[s], key) for s in seeds}
                b = {s: _means(new_pc[s], cov[s], key) for s in seeds}
                entry[key] = {"llm": mean_sd_over_seeds(a), "proposed_same_cases": mean_sd_over_seeds(b),
                              "llm_minus_proposed": rl.compare_splits(a, b)}
            for kind in ("single", "cross", "synth"):
                ids = {s: [c for c in cov[s] if case_kind(cases[c]) == kind] for s in seeds}
                ss = [s for s in seeds if ids[s]]
                if not ss:
                    continue
                entry[f"by_kind/{kind}"] = {key: {"llm": mean_sd_over_seeds({s: _means(pc_llm, ids[s], key) for s in ss}),
                                                 "proposed_same_cases": mean_sd_over_seeds({s: _means(new_pc[s], ids[s], key) for s in ss})}
                                           for key in ("recall@20", "AP@20")}
            res["vs_llm"][name] = entry
        out["settings"][setting] = res
        # ---- print
        print(f"\n### setting {setting}: {len(seeds)} splits; test cases per split: document split "
              f"{min(res['n_test']['document_split'].values())}–{max(res['n_test']['document_split'].values())}, "
              f"file split {min(res['n_test']['file_split'].values())}–{max(res['n_test']['file_split'].values())}")
        print(f"  {'proposed':44s} {'file split':>18s} {'document split':>18s}")
        for key in metrics:
            o, n = res["proposed"][key]["file_split"], res["proposed"][key]["document_split"]
            print(f"  {label[key]:44s} {o['mean']:.3f} ± {o['sd']:.3f}     {n['mean']:.3f} ± {n['sd']:.3f}")
        for kind in ("single", "cross", "synth"):
            for key in ("recall@20", "AP@20"):
                d = res["by_kind"][kind][key]
                if "document_split" in d and "file_split" in d:
                    print(f"    {kind:6s} {label[key]:36s} {d['file_split']['mean']:.3f} ± {d['file_split']['sd']:.3f}"
                          f"     {d['document_split']['mean']:.3f} ± {d['document_split']['sd']:.3f}")
        for name, entry in res["vs_llm"].items():
            cov = entry["n_test_with_llm_answer"]; nt = entry["n_test"]
            print(f"  vs {name}: test cases with an LLM answer {sum(cov.values())} of {sum(nt.values())} "
                  f"(per split {min(cov.values())}–{max(cov.values())})")
            for key in metrics:
                e = entry[key]; c = e["llm_minus_proposed"]
                print(f"    {label[key]:42s} LLM {e['llm']['mean']:.3f} ± {e['llm']['sd']:.3f}   proposed {e['proposed_same_cases']['mean']:.3f}"
                      f" ± {e['proposed_same_cases']['sd']:.3f}   LLM higher in {c['wins']}/{c['n']} splits (mean diff {c['mean_diff']:+.3f})")
            for kind in ("single", "cross", "synth"):
                k = entry.get(f"by_kind/{kind}")
                if k is None:
                    continue
                print(f"      {kind:6s} R@20 LLM {k['recall@20']['llm']['mean']:.3f} proposed {k['recall@20']['proposed_same_cases']['mean']:.3f}"
                      f"   MAP@20 LLM {k['AP@20']['llm']['mean']:.3f} proposed {k['AP@20']['proposed_same_cases']['mean']:.3f}")
    out["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    COMPARE_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {COMPARE_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("docs").set_defaults(fn=cmd_docs)
    p = sub.add_parser("splits"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_splits)
    p = sub.add_parser("train")
    p.add_argument("--setting", default="A", choices=["A", "B"])
    p.add_argument("--seeds", default="all")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_train)
    sub.add_parser("compare").set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
