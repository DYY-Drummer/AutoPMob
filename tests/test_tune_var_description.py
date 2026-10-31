# tests/test_tune_var_description.py
"""tune_var_description.py のテスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §7
速いテスト（データは読まない）: 選択の規則・引数の解析・merge と、run の組み立て（重い関数を偽物に差し替える）。
遅いテスト（RUN_SLOW=1 のときだけ）: 格子の既定の点（Δτ=0・重み (0.3, 0.7)）が本実験の S1・D1 を再現する.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_var_description as rvd
import tune_var_description as tvd

ROOT = Path(__file__).resolve().parents[1]
MAIN_A = ROOT / "experiments/var_desc_A.json"
slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="RUN_SLOW=1 のときだけ（数分かかる）")

DTAUS = [0.0, 0.02, 0.04, 0.06, 0.08]
WTEXTS = [0.3, 0.5, 0.7]  # 同点のときの優先順位（既定の (0.3, 0.7) に近い順）


# ---- 選択の規則（val の R@K が最大。同点は本実験の既定に近い方） ----

def entry(dtau, w_text, val_rk, test_rk=0.5):
    """格子の 1 点。w_var = 1 − w_text."""
    w_var = round(1.0 - w_text, 10)
    return {"tau_delta": dtau, "tau": 0.85 + dtau, "w_text": w_text, "w_var": w_var,
            "val": {"Recall@K_correct": val_rk, "MAP": val_rk - 0.2, "Recall@20": val_rk + 0.1, "n_cases": 20},
            "test": {"Recall@K_correct": test_rk, "MAP": test_rk - 0.2, "Recall@20": test_rk + 0.1, "n_cases": 10}}


def grid(val_of=lambda d, w: 0.6):
    """Δτ × 重み の 15 点（優先順位の順）。val_of(Δτ, w_text) で val の R@K を決める."""
    return [entry(d, w, val_of(d, w)) for d in DTAUS for w in WTEXTS]


def test_select_entry_takes_the_highest_val_recall():
    es = grid(lambda d, w: 0.5)
    es[7]["val"]["Recall@K_correct"] = 0.9  # 7 番目 = (Δτ 0.04, 文章の重み 0.5)
    assert tvd.select_entry(es) is es[7]
    assert (es[7]["tau_delta"], es[7]["w_text"]) == (0.04, 0.5)
    # 既定から一番遠い点でも、val が最大なら選ぶ（同点の規則は、最大が複数あるときだけ働く）
    far = grid(lambda d, w: 0.5)
    far[-1]["val"]["Recall@K_correct"] = 0.5001
    assert tvd.select_entry(far) is far[-1]


def test_select_entry_never_reads_the_test_split():
    es = grid(lambda d, w: 0.5)
    es[4]["val"]["Recall@K_correct"] = 0.7
    for e in es:
        e["test"]["Recall@K_correct"] = 0.0 if e is es[4] else 1.0  # test では es[4] が最悪
    assert tvd.select_entry(es) is es[4]
    for e in es:
        del e["test"]  # test の欄がなくても選べる（見ていない）
    assert tvd.select_entry(es) is es[4]


@pytest.mark.parametrize("k", range(15))
def test_select_entry_full_tie_break_order(k):
    """全点が同点のとき、優先順位は Δτ の小さい方 → 文章の重み 0.3 → 0.5 → 0.7 の順（k 番目より前を除いた中で k 番目が勝つ）."""
    es = grid()
    pool = es[k:]
    random.Random(k).shuffle(pool)  # 並びには依存しない
    assert tvd.select_entry(pool) is es[k]
    expected = [(d, w) for d in DTAUS for w in WTEXTS][k]
    assert (es[k]["tau_delta"], es[k]["w_text"]) == expected


def test_select_entry_prefers_a_smaller_tau_delta_before_looking_at_the_weights():
    a = entry(0.0, 0.7, 0.6)    # Δτ = 0 だが、重みは既定から遠い
    b = entry(0.02, 0.3, 0.6)   # 重みは既定だが、Δτ が大きい
    assert tvd.select_entry([b, a]) is a


def test_select_entry_with_fixed_tau_orders_by_the_weights_only():
    """S1 は τ = 1 のままなので、Δτ はすべて 0。重みだけで決まる."""
    es = [entry(0.0, 0.7, 0.8), entry(0.0, 0.5, 0.8), entry(0.0, 0.3, 0.8)]
    assert tvd.select_entry(es) is es[2]
    assert tvd.select_entry(es[:2]) is es[1]


def test_select_entry_float_noise_is_a_tie_but_a_real_difference_is_not():
    near = [entry(0.04, 0.5, 0.6), entry(0.0, 0.3, 0.6 - 1e-15)]
    assert tvd.select_entry(near) is near[1]  # 丸めの誤差だけ低い = 同点 → 既定に近い方
    real = [entry(0.04, 0.5, 0.6), entry(0.0, 0.3, 0.6 - 1e-6)]
    assert tvd.select_entry(real) is real[0]  # 本物の差は、既定に近くても負ける


def test_select_entry_ranks_other_weights_by_distance_from_the_default():
    """--weights に既定の 3 組以外を渡したときも、文章の重みが既定の 0.3 に近い順（等距離は小さい方）."""
    es = [entry(0.0, 0.5, 0.6), entry(0.0, 0.4, 0.6), entry(0.0, 0.2, 0.6)]
    assert tvd.select_entry(es) is es[2]            # 0.2 と 0.4 は 0.3 から等距離 → 小さい方
    assert tvd.select_entry(es[:2]) is es[1]        # 0.4 は 0.5 より 0.3 に近い


def test_select_entry_rejects_an_empty_list_and_non_finite_val():
    with pytest.raises(ValueError, match="格子点がない"):
        tvd.select_entry([])
    with pytest.raises(ValueError, match="有限でない"):
        tvd.select_entry([entry(0.0, 0.3, float("nan")), entry(0.02, 0.3, 0.5)])


# ---- 引数の解析 ----

def test_parse_tau_deltas():
    assert tvd.parse_tau_deltas("0,0.02,0.04,0.06,0.08") == [0.0, 0.02, 0.04, 0.06, 0.08]
    assert tvd.parse_tau_deltas(" 0 , +0.02 ,-0.02 ") == [0.0, 0.02, -0.02]
    assert tvd.parse_tau_deltas("0.05") == [0.05]
    for bad in ["", ",", "a", "0,,0.02", "0,nan", "0,inf", "0.02,0.02", "0,0.0"]:
        with pytest.raises(argparse.ArgumentTypeError):
            tvd.parse_tau_deltas(bad)


def test_parse_weights():
    assert tvd.parse_weights("0.7:0.3,0.5:0.5,0.3:0.7") == [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
    assert tvd.parse_weights(" 0.3 : 0.7 ") == [(0.3, 0.7)]
    assert tvd.parse_weights("1:0") == [(1.0, 0.0)]
    for bad in ["", "0.7", "0.7:0.3:0.1", "a:b", "0.7:", ":0.3", "0:0", "-0.1:1.1", "nan:1",
                "0.5:0.5,0.5:0.5", "0.7:0.3,,0.5:0.5"]:
        with pytest.raises(argparse.ArgumentTypeError):
            tvd.parse_weights(bad)


def test_parse_seed_list_and_renames():
    assert tvd.parse_seed_list("42, 123,456") == [42, 123, 456]
    for bad in ["", "a", "42,42", "4.5", "42,,123"]:
        with pytest.raises(argparse.ArgumentTypeError):
            tvd.parse_seed_list(bad)
    assert tvd.parse_renames("0.5,1.0") == [0.5, 1.0]
    assert tvd.parse_renames("") == []  # 付け替えの評価を省く（本実験の --renames と同じ）
    for bad in ["0", "1.5", "-0.5", "a", "0.5,0.5", "0.5,0.501"]:  # 最後は r50 が重なる
        with pytest.raises(argparse.ArgumentTypeError):
            tvd.parse_renames(bad)


def test_run_arguments_default_to_the_designed_grid():
    a = tvd.build_parser().parse_args(["run", "--setting", "A", "--output", "x.json"])
    assert a.tau_deltas == [0.0, 0.02, 0.04, 0.06, 0.08]
    assert a.weights == [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
    assert a.renames == [0.5, 1.0] and a.seed_list is None and a.save_per_case is False
    assert tvd.SEEDS == rvd.SEEDS and tvd.HP is rvd.HP
    b = tvd.build_parser().parse_args(["run", "--setting", "B", "--output", "y.json", "--seed-list", "42,123",
                                       "--tau-deltas", "0,0.04", "--weights", "0.5:0.5", "--renames", "1.0",
                                       "--save-per-case"])
    assert (b.setting, b.seed_list, b.tau_deltas, b.weights, b.renames, b.save_per_case) == \
        ("B", [42, 123], [0.0, 0.04], [(0.5, 0.5)], [1.0], True)


@pytest.mark.parametrize("bad", [["--weights", "0.7"], ["--tau-deltas", "0,0"], ["--seed-list", "x"],
                                 ["--renames", "2"], ["--setting", "C"]], ids=lambda b: b[0])
def test_bad_run_arguments_are_a_usage_error(bad, capsys):
    argv = ["run", "--setting", "A", "--output", "x.json", *bad]  # 同じ引数が 2 回なら後のものが効く
    with pytest.raises(SystemExit) as exc:
        tvd.build_parser().parse_args(argv)
    assert exc.value.code == 2
    assert bad[0] in capsys.readouterr().err


# ---- run の組み立て（重い関数を偽物に差し替える。データ・E5・学習は使わない） ----

TR, VA, TE = [0, 1, 2, 3, 4, 5], [6, 7], [8, 9]
TAU0 = 0.85
D1_POINTS = [(d, w) for d in DTAUS for w in (0.7, 0.5, 0.3)]  # 既定の --weights の順
S1_POINTS = [(0.0, w) for w in (0.7, 0.5, 0.3)]


def best_val(cond, seed):
    """fake の val で最良の格子点（Δτ, 文章の重み）。None = 全点が同点（→ 既定の点が選ばれる）.

    seed 42・123・7 は決め打ち。ほかの seed は seed ごとに違う点（merge のテストで、seed ごとの値がばらつくように）."""
    if seed == 123:
        return None
    pts = D1_POINTS if cond == "D1" else S1_POINTS
    fixed = {42: pts[7] if cond == "D1" else pts[1], 7: pts[12] if cond == "D1" else pts[0]}
    return fixed.get(seed, pts[(seed * 7) % len(pts)])


def fake_rk(split, cond, seed, dtau, w_text):
    """fake の R@K。val は seed ごとの最良の点だけが高い。test は格子点ごとに違う値（選んだ点の値が見分けられる）."""
    if split == "val":
        return 0.40 + (0.20 if best_val(cond, seed) == (dtau, w_text) else 0.0)
    pts = D1_POINTS if cond == "D1" else S1_POINTS
    return 0.10 + 0.003 * pts.index((dtau, w_text)) + (0.001 if cond == "D1" else 0.0)


def fake_case(ci, rk):
    return {"n_correct": 1, "MRR_first": rk, "MRR_worst": rk, "MRR_avg": rk, "Precision@C": rk,
            "Recall@K_correct": rk, "AP": rk, "ranks": [1],
            "FullRecall": {3: rk, 5: rk, 10: rk, 20: rk}, "Recall": {3: rk, 5: rk, 10: rk, 20: rk},
            "variant": "original", "case_id": f"case_{ci}", "n_input": 2, "n_output": 1, "n_sources": 1}


@pytest.fixture
def fake_env(monkeypatch):
    """tune_var_description の重い関数（データの読み込み・E5・TF-IDF/SVD・学習・評価）を偽物にする。呼び出しを記録して返す."""
    rec = SimpleNamespace(prepare=[], tau=[], saves=[], trained=[], evaluated=[], svd_seeds=[], split_seeds=[])
    data = {"cases": list(range(10)), "eq_texts": ["x"] * 3, "eq_ptr": np.array([0, 1, 2, 3]),
            "correct_lists": [[0]] * 10}

    def view(cond, p=0.0):
        return SimpleNamespace(ctx="c", io_text="io", cond=cond, p=p)

    def tau_for(matcher, views, d, idx, tau_delta=0.0):
        rec.tau.append((matcher.kind, list(idx)))
        return tvd.SYMBOL_TAU if matcher.kind == "symbol" else TAU0 + tau_delta

    def prepare(idx, views, matcher, tau, X_ctx, X_eq, svd_sim, d, inc, w_text, w_var, top_k=50):
        rec.prepare.append(SimpleNamespace(idx=list(idx), cond=matcher.cond, p=matcher.p, tau=tau,
                                           w=(w_text, w_var), svd=svd_sim, X_ctx=X_ctx))
        return {"tag": (matcher.cond, matcher.p, tau, w_text, w_var)}, 0.0

    def train(seed, tr, prep, cl):
        assert list(tr) == TR and cl is data["correct_lists"]  # 学習は train だけ
        rec.trained.append(prep["tag"])
        return SimpleNamespace(uid=len(rec.trained) - 1, seed=seed, tag=prep["tag"])  # uid = 何番目に学習した模型か

    def evaluate(model, idx, prep, d):
        split = "val" if list(idx) == VA else "test"
        rec.evaluated.append((model.uid, list(idx), prep["tag"][1]))
        cond, _, tau, w_text, w_var = model.tag
        assert prep["tag"][3:] == (w_text, w_var), "モデルと違う重みで準備した特徴量を評価している"
        dtau = round(tau - (TAU0 if cond == "D1" else tvd.SYMBOL_TAU), 9)
        rk = fake_rk(split, cond, model.seed, dtau, w_text) - 0.2 * prep["tag"][1]  # 付け替えの割合 p だけ下がる
        return [fake_case(ci, rk) for ci in idx]

    real_save = tvd.save_json

    def save_json(obj, path):
        rec.saves.append([int(s) for s in obj["selected"]["D1"]])  # 保存のたびに、済んだ seed を処理した順に記録
        real_save(obj, path)

    monkeypatch.setattr(tvd, "load_data", lambda setting: dict(data, setting=setting))
    monkeypatch.setattr(tvd, "load_var_descriptions", lambda: {})
    monkeypatch.setattr(tvd, "make_tfidf", lambda texts: (SimpleNamespace(transform=lambda t: ("X_ctx", len(t))), "X_eq"))
    monkeypatch.setattr(tvd, "build_views", lambda cond, d, vd, p=0.0: [view(cond, p)] * len(d["cases"]))
    monkeypatch.setattr(tvd, "make_matcher", lambda cond, d, views: SimpleNamespace(
        kind="symbol" if cond in ("S0", "S1") else "description", cond=cond, p=views[0].p))

    def split_cases(d, seed):
        rec.split_seeds.append(seed)
        return {"train": list(TR), "val": list(VA), "test": list(TE)}

    def seed_svd_sim(seed, tfidf, X_eq):
        rec.svd_seeds.append(seed)
        return lambda vs: ("svd", vs[0].cond, vs[0].p)

    monkeypatch.setattr(tvd, "split_cases", split_cases)
    monkeypatch.setattr(tvd, "seed_svd_sim", seed_svd_sim)
    monkeypatch.setattr(tvd, "tau_for", tau_for)
    monkeypatch.setattr(tvd, "prepare", prepare)
    monkeypatch.setattr(tvd, "train", train)
    monkeypatch.setattr(tvd, "evaluate", evaluate)
    monkeypatch.setattr(tvd, "save_json", save_json)
    return rec


def run_fake(tmp_path, name, *extra, setting="A"):
    out = tmp_path / name
    tvd.main(["run", "--setting", setting, "--output", str(out), *extra])
    return json.loads(out.read_text(encoding="utf-8"))


def test_run_evaluates_the_whole_grid_on_val_and_test_and_selects_by_val(fake_env, tmp_path, capsys):
    doc = run_fake(tmp_path, "r.json", "--seed-list", "42,123,7", "--save-per-case")
    assert set(doc) == {"config", "tau", "grid", "selected", "results"}
    assert list(doc["results"]) == ["S1-tuned", "S1-tuned-r50", "S1-tuned-r100", "D1-tuned"]
    assert list(doc["tau"]) == ["D1"] and list(doc["tau"]["D1"]) == ["42", "123", "7"]
    assert set(doc["tau"]["D1"].values()) == {TAU0}  # Youden の τ（train で 1 回）
    for cond, n in (("S1", 3), ("D1", 15)):
        assert list(doc["grid"][cond]) == ["42", "123", "7"]
        for s in ("42", "123", "7"):
            es = doc["grid"][cond][s]
            assert len(es) == n
            for e in es:
                assert set(e) == {"tau_delta", "tau", "w_text", "w_var", "val", "test"}
                for part in ("val", "test"):
                    assert set(e[part]) == {"Recall@K_correct", "MAP", "Recall@20", "n_cases"}
                assert e["val"]["n_cases"] == len(VA) and e["test"]["n_cases"] == len(TE)
    # 格子の並び: Δτ（外）× 重み（内）。S1 は Δτ = 0 だけ、τ = 1
    d1 = doc["grid"]["D1"]["42"]
    assert [(e["tau_delta"], e["w_text"], e["w_var"]) for e in d1] == [(d, w, round(1 - w, 10)) for d, w in D1_POINTS]
    assert [e["tau"] for e in d1] == pytest.approx([TAU0 + d for d, _ in D1_POINTS])
    s1 = doc["grid"]["S1"]["42"]
    assert [(e["tau_delta"], e["tau"], e["w_text"]) for e in s1] == [(0.0, 1.0, 0.7), (0.0, 1.0, 0.5), (0.0, 1.0, 0.3)]
    # val で最良の点を選ぶ。全点が同点の seed 123 は既定（Δτ=0・重み (0.3, 0.7)）
    sel = doc["selected"]
    assert [(sel["D1"][s]["tau_delta"], sel["D1"][s]["w_text"]) for s in ("42", "123", "7")] == \
        [(0.04, 0.5), (0.0, 0.3), (0.08, 0.7)]
    assert [(sel["S1"][s]["tau_delta"], sel["S1"][s]["w_text"]) for s in ("42", "123", "7")] == \
        [(0.0, 0.5), (0.0, 0.3), (0.0, 0.7)]
    for cond in ("S1", "D1"):
        for s in ("42", "123", "7"):
            assert sel[cond][s] in doc["grid"][cond][s]  # 選んだ点は格子の点そのもの
    # 報告する値 = 選んだ点の test（test が最良の点ではない）
    for cond, label in (("S1", "S1-tuned"), ("D1", "D1-tuned")):
        got = {p["seed"]: p["Recall@K_correct"] for p in doc["results"][label]["per_seed"]}
        for seed in (42, 123, 7):
            e = sel[cond][str(seed)]
            assert got[seed] == pytest.approx(e["test"]["Recall@K_correct"])
            assert got[seed] == pytest.approx(fake_rk("test", cond, seed, e["tau_delta"], e["w_text"]))
        best_test = max(e["test"]["Recall@K_correct"] for e in doc["grid"][cond]["42"])
        assert got[42] < best_test  # seed 42 は val で選ぶので、test が最良の点にはならない
    # 付け替え: 選んだ S1 の模型で、選んだ重みのまま、p = 0.5・1.0 のテスト用ケースを評価する
    for seed in (42, 123, 7):
        base = sel["S1"][str(seed)]["test"]["Recall@K_correct"]
        for label, p in (("S1-tuned-r50", 0.5), ("S1-tuned-r100", 1.0)):
            got = {r["seed"]: r["Recall@K_correct"] for r in doc["results"][label]["per_seed"]}
            assert got[seed] == pytest.approx(base - 0.2 * p)
    # per_case: テストケースだけ（seed × 2 件）。ラベルは結果のラベル
    for label, res in doc["results"].items():
        assert [r["seed"] for r in res["per_seed"]] == [42, 123, 7]
        assert len(res["per_case"]) == 3 * len(TE) and {r["mode"] for r in res["per_case"]} == {label}
        assert {r["case_id"] for r in res["per_case"]} == {f"case_{ci}" for ci in TE}
        assert res["Recall@K_correct"]["mean"] == pytest.approx(
            round(float(np.mean([p["Recall@K_correct"] for p in res["per_seed"]])), 4))
    # 進捗: 格子の 1 点ごとに 1 行（S1 3 + D1 15）と、選んだ点ごとに 1 行を seed ごとに出す
    out = capsys.readouterr().out
    assert out.count("val R@K=") == 3 * (3 + 15) + 3 * 2
    assert out.count(" selected: ") == 3 * 2
    assert "[seed=42] D1 dtau=+0.04 w=(0.5,0.5)" in out
    assert "[seed=42] D1 selected: dtau=+0.04 w=(0.5,0.5)" in out


def test_run_prepares_every_case_with_the_selected_weights_and_fixed_tau(fake_env, tmp_path):
    run_fake(tmp_path, "r.json", "--seed-list", "42")
    calls = fake_env.prepare
    assert len(calls) == 3 + 2 + 15  # S1 の格子 3 + 付け替え 2 + D1 の格子 15
    s1, rename, d1 = calls[:3], calls[3:5], calls[5:]
    assert [c.cond for c in s1] == ["S1"] * 3 and [c.cond for c in d1] == ["D1"] * 15
    # 格子の点は train + val + test を準備する（val は選択用、test は報告用）。τ は S1 が 1、D1 が τ0 + Δτ
    assert all(c.idx == TR + VA + TE for c in s1 + d1)
    assert [c.tau for c in s1] == [1.0] * 3
    assert [c.tau for c in d1] == pytest.approx([TAU0 + d for d, _ in D1_POINTS])
    assert [c.w for c in d1] == [(w, round(1 - w, 10)) for _, w in D1_POINTS]
    # 付け替え: test だけを、選んだ重み（seed 42 は (0.5, 0.5)）・τ = 1 で
    assert [(c.cond, c.p, c.idx, c.tau, c.w) for c in rename] == \
        [("S1", 0.5, TE, 1.0, (0.5, 0.5)), ("S1", 1.0, TE, 1.0, (0.5, 0.5))]
    assert [c.svd for c in rename] == [("svd", "S1", 0.5), ("svd", "S1", 1.0)]  # 付け替えた view から作った SVD の類似度
    assert all(c.X_ctx == ("X_ctx", 10) for c in calls)  # 説明文の行列は条件ごとに 1 度だけ作ったもの
    # τ0 は D1 だけ train のケースで決める（記号版は tau_for が SYMBOL_TAU を返す）
    assert fake_env.tau == [("symbol", TR), ("description", TR)]
    assert fake_env.split_seeds == fake_env.svd_seeds == [42]  # 分割・SVD は seed ごとに 1 回（格子の点ごとに作り直さない）


def test_run_evaluates_the_renamed_cases_with_the_selected_s1_model(fake_env, tmp_path):
    run_fake(tmp_path, "r.json", "--seed-list", "42")
    s1_models = fake_env.trained[:3]  # 学習した順 = 格子の順（S1 は文章の重み 0.7・0.5・0.3）
    assert [t[3] for t in s1_models] == [0.7, 0.5, 0.3]
    renamed = [(uid, p) for uid, idx, p in fake_env.evaluated if idx == TE and p > 0]
    assert renamed == [(1, 0.5), (1, 1.0)]  # seed 42 の S1 は 2 番目の点（重み 0.5:0.5）が val で最良。その模型だけで評価する
    assert len(fake_env.trained) == 3 + 15  # 付け替えのために学び直さない


def test_run_passes_each_seed_to_the_split_and_the_svd(fake_env, tmp_path):
    run_fake(tmp_path, "r.json", "--seed-list", "42,123,7")
    assert fake_env.split_seeds == fake_env.svd_seeds == [42, 123, 7]


def test_run_saves_after_every_seed(fake_env, tmp_path):
    run_fake(tmp_path, "r.json", "--seed-list", "42,123,7")
    assert fake_env.saves == [[42], [42, 123], [42, 123, 7]]  # 途中で止まっても済んだ seed が残る


def test_run_config_records_the_arguments(fake_env, tmp_path):
    doc = run_fake(tmp_path, "r.json", "--seed-list", "42,123", "--save-per-case")
    cfg = doc["config"]
    assert "cmd" not in cfg
    assert (cfg["setting"], cfg["seed_list"], cfg["tau_deltas"], cfg["weights"], cfg["renames"]) == \
        ("A", [42, 123], DTAUS, [[0.7, 0.3], [0.5, 0.5], [0.3, 0.7]], [0.5, 1.0])
    assert cfg["save_per_case"] is True and cfg["output"] == str(tmp_path / "r.json")
    assert {k: cfg[k] for k in rvd.HP} == rvd.HP and cfg["spec"] == rvd.SPEC and cfg["n_cases"] == 10
    assert run_fake(tmp_path, "r2.json")["config"]["seed_list"] == rvd.SEEDS  # 省略 = 正典の 10 seed


def test_run_without_save_per_case_has_no_per_case(fake_env, tmp_path):
    doc = run_fake(tmp_path, "r.json", "--seed-list", "42")
    assert doc["config"]["save_per_case"] is False
    assert all("per_case" not in res for res in doc["results"].values())


def test_run_uses_only_the_requested_grid(fake_env, tmp_path):
    doc = run_fake(tmp_path, "r.json", "--seed-list", "42", "--tau-deltas", "0,0.04", "--weights", "0.5:0.5,0.3:0.7",
                   "--renames", "0.5")
    assert [(e["tau_delta"], e["w_text"]) for e in doc["grid"]["D1"]["42"]] == \
        [(0.0, 0.5), (0.0, 0.3), (0.04, 0.5), (0.04, 0.3)]
    assert [(e["tau_delta"], e["w_text"]) for e in doc["grid"]["S1"]["42"]] == [(0.0, 0.5), (0.0, 0.3)]
    assert list(doc["results"]) == ["S1-tuned", "S1-tuned-r50", "D1-tuned"]


def test_run_without_renames_evaluates_no_renamed_cases(fake_env, tmp_path):
    doc = run_fake(tmp_path, "r.json", "--seed-list", "42", "--renames", "")
    assert list(doc["results"]) == ["S1-tuned", "D1-tuned"] and doc["config"]["renames"] == []
    assert len(fake_env.prepare) == 3 + 15 and all(c.p == 0.0 for c in fake_env.prepare)


def test_run_rejects_a_seed_without_validation_cases(fake_env, tmp_path, monkeypatch):
    monkeypatch.setattr(tvd, "split_cases", lambda d, seed: {"train": list(TR), "val": [], "test": list(TE)})
    with pytest.raises(ValueError, match="val"):
        run_fake(tmp_path, "r.json", "--seed-list", "42")


# ---- merge ----

PART1, PART2 = "42,123,456,789,1024", "2024,3141,5926,7777,9999"


def make_parts(fake_env, tmp_path, part1=PART1, part2=PART2, extra2=()):
    """run の出力（偽物の評価）から、本番と同じ形の部分ファイル 2 つを作る."""
    return (run_fake(tmp_path, "p1.json", "--seed-list", part1, "--save-per-case"),
            run_fake(tmp_path, "p2.json", "--seed-list", part2, "--save-per-case", *extra2))


def parts_of(*docs):
    return {f"part{i + 1}.json": d for i, d in enumerate(docs)}


def test_merge_of_two_halves_equals_one_run_over_all_seeds(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path)
    whole = run_fake(tmp_path, "whole.json", "--save-per-case")
    merged = tvd.merge_parts(parts_of(p2, p1), "A", "merged.json")  # 入力の順は問わない
    assert merged["config"] == {**whole["config"], "output": "merged.json"}
    for key in ("tau", "grid", "selected"):
        assert merged[key] == whole[key], key
    seeds = [str(s) for s in rvd.SEEDS]  # 辞書の等しさは並びを見ないので、seed の昇順は別に確かめる
    assert list(merged["tau"]["D1"]) == seeds
    for key in ("grid", "selected"):
        for cond in ("S1", "D1"):
            assert list(merged[key][cond]) == seeds
    assert merged["results"] == whole["results"]  # per_seed の並び・要約の再計算・per_case まで一致


def test_merge_recomputes_the_summary_from_all_seeds(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path)
    merged = tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    for label, res in merged["results"].items():
        assert [r["seed"] for r in res["per_seed"]] == rvd.SEEDS
        vals = [r["Recall@K_correct"] for r in res["per_seed"]]
        assert res["Recall@K_correct"] == {"mean": round(float(np.mean(vals)), 4),
                                           "std": round(float(np.std(vals, ddof=1)), 4)}
        for half in (p1, p2):  # どちらの部分ファイルの要約とも違う（5 seed の要約を写していない）
            assert res["Recall@K_correct"] != half["results"][label]["Recall@K_correct"]
        assert res["mode"] == label and res["n_features"] == 10
        assert [r["seed"] for r in res["per_case"]] == sorted(r["seed"] for r in res["per_case"])
        assert len(res["per_case"]) == len(rvd.SEEDS) * len(TE)


def test_merge_rejects_a_config_mismatch(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path, extra2=("--weights", "0.5:0.5,0.3:0.7"))
    with pytest.raises(tvd.MergeError, match=r"config.*weights"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    # seed_list と output だけが違うのは許す（上のテストが通っている）。HP や n_cases の違いは許さない
    p1, p2 = make_parts(fake_env, tmp_path)
    p2["config"]["n_cases"] = 999
    with pytest.raises(tvd.MergeError, match=r"config.*n_cases"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")


def test_merge_rejects_a_setting_that_is_not_the_requested_one(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path)
    with pytest.raises(tvd.MergeError, match=r"setting.*'A'.*B"):
        tvd.merge_parts(parts_of(p1, p2), "B", "m.json")


def test_merge_rejects_overlapping_seeds(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path, part2="1024,2024,3141,5926,7777,9999")
    with pytest.raises(tvd.MergeError, match=r"重な.*1024"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")


def test_merge_rejects_missing_and_unexpected_seeds(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path, part2="2024,3141,5926,7777")
    with pytest.raises(tvd.MergeError, match=r"足りない.*9999"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    p1, p2 = make_parts(fake_env, tmp_path, part2="2024,3141,5926,7777,9999,43")
    with pytest.raises(tvd.MergeError, match=r"正典.*43"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")


def test_merge_of_a_single_part_with_all_seeds_is_allowed(fake_env, tmp_path):
    whole = run_fake(tmp_path, "whole.json", "--save-per-case")
    assert tvd.merge_parts(parts_of(whole), "A", "m.json")["results"] == whole["results"]


def test_merge_rejects_a_part_that_stopped_early(fake_env, tmp_path):
    """run が途中で止まった部分ファイル: config の seed_list にあって、結果にない seed を名指しする."""
    p1, p2 = make_parts(fake_env, tmp_path)
    p2["tau"]["D1"].pop("9999")
    for key in ("grid", "selected"):
        for cond in ("S1", "D1"):
            p2[key][cond].pop("9999")
    for res in p2["results"].values():
        res["per_seed"] = [r for r in res["per_seed"] if r["seed"] != 9999]
    with pytest.raises(tvd.MergeError, match=r"途中.*9999"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")


def test_merge_rejects_a_part_whose_structures_disagree_on_seeds(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path)
    del p2["selected"]["D1"]["9999"]
    with pytest.raises(tvd.MergeError, match=r"part2.json.*9999"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")


def test_merge_rejects_a_file_that_is_not_a_run_output_or_has_duplicate_seeds(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path)
    del p2["selected"]
    with pytest.raises(tvd.MergeError, match=r"part2.json.*selected"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    p1, p2 = make_parts(fake_env, tmp_path)
    p2["tau"] = {}
    with pytest.raises(tvd.MergeError, match=r"part2.json.*形式"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    p1, p2 = make_parts(fake_env, tmp_path)
    p2["config"] = ["not", "a", "dict"]
    with pytest.raises(tvd.MergeError, match=r"part2.json.*config"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    p1, p2 = make_parts(fake_env, tmp_path)
    rows = p2["results"]["D1-tuned"]["per_seed"]
    rows.append(dict(rows[0]))
    with pytest.raises(tvd.MergeError, match=r"part2.json.*D1-tuned.*重複"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    with pytest.raises(tvd.MergeError, match="入力"):
        tvd.merge_parts({}, "A", "m.json")


def test_merge_rejects_different_labels_and_missing_per_case(fake_env, tmp_path):
    p1, p2 = make_parts(fake_env, tmp_path)
    del p2["results"]["S1-tuned-r100"]
    with pytest.raises(tvd.MergeError, match="S1-tuned-r100"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    p1, p2 = make_parts(fake_env, tmp_path)
    del p2["results"]["D1-tuned"]["per_case"]
    with pytest.raises(tvd.MergeError, match="per_case"):
        tvd.merge_parts(parts_of(p1, p2), "A", "m.json")


def test_merge_without_per_case_keeps_the_results_free_of_it(fake_env, tmp_path):
    p1 = run_fake(tmp_path, "p1.json", "--seed-list", PART1)
    p2 = run_fake(tmp_path, "p2.json", "--seed-list", PART2)
    merged = tvd.merge_parts(parts_of(p1, p2), "A", "m.json")
    assert all("per_case" not in res for res in merged["results"].values())


def write_parts(fake_env, tmp_path, **kw):
    p1, p2 = make_parts(fake_env, tmp_path, **kw)
    return tmp_path / "p1.json", tmp_path / "p2.json"


def test_merge_command_writes_the_merged_file(fake_env, tmp_path):
    f1, f2 = write_parts(fake_env, tmp_path)
    out = tmp_path / "sub" / "merged.json"  # 出力先の親ディレクトリがなくても作る
    tvd.main(["merge", "--setting", "A", "--inputs", str(f1), str(f2), "--output", str(out)])
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["config"]["seed_list"] == rvd.SEEDS and merged["config"]["output"] == str(out)
    assert [r["seed"] for r in merged["results"]["D1-tuned"]["per_seed"]] == rvd.SEEDS


def test_merge_command_exits_with_a_message_and_writes_nothing_on_error(fake_env, tmp_path):
    f1, f2 = write_parts(fake_env, tmp_path, part2="1024,2024,3141,5926,7777,9999")
    out = tmp_path / "merged.json"
    with pytest.raises(SystemExit) as exc:
        tvd.main(["merge", "--setting", "A", "--inputs", str(f1), str(f2), "--output", str(out)])
    assert isinstance(exc.value.code, str) and "1024" in exc.value.code  # 文字列 = メッセージを標準エラーに出して終了コード 1
    assert not out.exists() and not (tmp_path / "merged.json.tmp").exists()


def test_merge_command_names_an_unreadable_input(tmp_path):
    out = tmp_path / "merged.json"
    with pytest.raises(SystemExit) as exc:
        tvd.main(["merge", "--setting", "A", "--inputs", str(tmp_path / "nope.json"), "--output", str(out)])
    assert "nope.json" in exc.value.code and not out.exists()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        tvd.main(["merge", "--setting", "A", "--inputs", str(bad), "--output", str(out)])
    assert "bad.json" in exc.value.code and not out.exists()


def test_merge_command_refuses_to_overwrite_an_input(fake_env, tmp_path):
    f1, f2 = write_parts(fake_env, tmp_path)
    before = f1.read_bytes()
    with pytest.raises(SystemExit) as exc:
        tvd.main(["merge", "--setting", "A", "--inputs", str(f1), str(f2), "--output", str(f1)])
    assert "入力" in exc.value.code and f1.read_bytes() == before


def test_merge_command_refuses_the_same_input_twice(fake_env, tmp_path):
    f1, f2 = write_parts(fake_env, tmp_path)
    with pytest.raises(SystemExit) as exc:
        tvd.main(["merge", "--setting", "A", "--inputs", str(f1), str(f1), "--output", str(tmp_path / "m.json")])
    assert "同じ入力" in exc.value.code and not (tmp_path / "m.json").exists()


def test_merge_command_process_exits_non_zero_and_prints_the_problem(fake_env, tmp_path):
    f1, f2 = write_parts(fake_env, tmp_path, part2="2024,3141,5926,7777")  # 9999 が足りない
    out = tmp_path / "merged.json"
    r = subprocess.run([sys.executable, "tune_var_description.py", "merge", "--setting", "A", "--inputs",
                        str(f1), str(f2), "--output", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "9999" in r.stderr and not out.exists()


# ---- run_var_description_tuning.sh（偽の tune_var_description.py で制御フローだけ確かめる。本物の run は走らせない） ----

STUB = r"""
import json, os, sys, time
args = sys.argv[1:]
opt = lambda name: args[args.index(name) + 1]
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
if args[0] == "run":
    if os.environ.get("STUB_FAIL_RUN") == opt("--setting") + ":" + opt("--seed-list").split(",")[0]:
        sys.exit(3)
    time.sleep(1.0)  # 失敗した run より遅く終わる（残りも待つことの確認）
    open(opt("--output"), "w").write("part")
else:
    if os.environ.get("STUB_FAIL_MERGE") == opt("--setting"):
        print("merge boom", file=sys.stderr)
        sys.exit(1)
    open(opt("--output"), "w").write("merged " + opt("--setting"))
"""
HALF1, HALF2 = "42,123,456,789,1024", "2024,3141,5926,7777,9999"


def run_script(tmp_path, **env):
    """スクリプトと偽の tune_var_description.py だけを置いた作業ディレクトリで実行する（スクリプトは自分の場所へ cd する）."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "run_var_description_tuning.sh").write_text((ROOT / "run_var_description_tuning.sh").read_text(encoding="utf-8"))
    (work / "tune_var_description.py").write_text(STUB, encoding="utf-8")
    log = tmp_path / "calls.log"
    r = subprocess.run(["bash", "run_var_description_tuning.sh"], cwd=work, capture_output=True, text=True,
                       env={**os.environ, "STUB_LOG": str(log), **env})
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return r, calls, work


def test_tuning_script_runs_four_parts_in_parallel_then_merges_each_setting(tmp_path):
    r, calls, work = run_script(tmp_path)
    assert r.returncode == 0 and r.stdout.strip().endswith("ALL DONE") and r.stderr == ""
    L = "experiments/var_desc_logs"
    runs, merges = calls[:4], calls[4:]
    assert sorted(runs) == sorted(
        ["run", "--setting", S, "--seed-list", seeds, "--save-per-case", "--output", f"{L}/tune_{S}_{i}.json"]
        for S in "AB" for i, seeds in ((1, HALF1), (2, HALF2)))
    assert sorted(int(x) for h in (HALF1, HALF2) for x in h.split(",")) == sorted(rvd.SEEDS)  # 前半と後半で正典の 10 seed
    assert merges == [["merge", "--setting", S, "--inputs", f"{L}/tune_{S}_1.json", f"{L}/tune_{S}_2.json",
                       "--output", f"experiments/var_desc_tuned_{S}.json"] for S in "AB"]  # 4 つの run が済んでから
    for S in "AB":
        assert (work / f"experiments/var_desc_tuned_{S}.json").read_text() == f"merged {S}"
        assert (work / f"{L}/tune_{S}_1.log").exists() and (work / f"{L}/tune_merge_{S}.log").exists()


def test_tuning_script_fails_loudly_and_waits_for_the_other_parts_when_a_run_fails(tmp_path):
    r, calls, work = run_script(tmp_path, STUB_FAIL_RUN="A:2024")
    assert r.returncode == 1
    assert "TUNING RUN FAILED (see experiments/var_desc_logs/tune_*.log)" in r.stderr
    assert "ALL DONE" not in r.stdout
    assert [c[0] for c in calls] == ["run"] * 4  # merge は走らせない
    L = work / "experiments/var_desc_logs"
    assert not (L / "tune_A_2.json").exists()  # 失敗した run
    assert all((L / f"tune_{S}_{i}.json").exists() for S, i in (("A", 1), ("B", 1), ("B", 2)))  # 残りは終わるまで待った
    assert not (work / "experiments/var_desc_tuned_A.json").exists()


def test_tuning_script_fails_loudly_when_a_merge_fails(tmp_path):
    r, calls, work = run_script(tmp_path, STUB_FAIL_MERGE="B")
    assert r.returncode == 1 and "TUNING MERGE FAILED (setting B; see experiments/var_desc_logs/tune_merge_B.log)" in r.stderr
    assert "ALL DONE" not in r.stdout
    assert (work / "experiments/var_desc_tuned_A.json").exists() and not (work / "experiments/var_desc_tuned_B.json").exists()
    assert "merge boom" in (work / "experiments/var_desc_logs/tune_merge_B.log").read_text()


# ---- 遅いテスト: 格子の既定の点が本実験を再現する（設定 A・seed 42） ----

@slow
def test_default_grid_point_reproduces_the_main_run(tmp_path):
    """最小の格子（Δτ=0・重み (0.3, 0.7) だけ）の test の値が、本実験の S1・D1（seed 42）と厳密に一致する.

    D1 は仕様 §7 に書かれた 0.35727607039663783。S1 と付け替え（S1-r50・r100）は本実験の結果ファイルから読む.
    同じ設定・同じ乱数の順なので、浮動小数点の丸めも含めて同じ値になる."""
    out = tmp_path / "tune.json"
    subprocess.run([sys.executable, "tune_var_description.py", "run", "--setting", "A", "--seed-list", "42",
                    "--tau-deltas", "0", "--weights", "0.3:0.7", "--save-per-case", "--output", str(out)],
                   cwd=ROOT, check=True)
    got = json.load(open(out))
    main = json.load(open(MAIN_A))["results"]

    def main_row(label):
        (row,) = [p for p in main[label]["per_seed"] if p["seed"] == 42]
        return row

    (d1,) = got["grid"]["D1"]["42"]
    (s1,) = got["grid"]["S1"]["42"]
    assert d1["test"]["Recall@K_correct"] == 0.35727607039663783
    assert d1["test"]["Recall@K_correct"] == main_row("D1")["Recall@K_correct"]
    assert s1["test"]["Recall@K_correct"] == main_row("S1")["Recall@K_correct"]
    assert (d1["tau_delta"], d1["w_text"], d1["w_var"]) == (0.0, 0.3, 0.7)
    assert got["selected"]["D1"]["42"] == d1 and got["selected"]["S1"]["42"] == s1
    # 報告する値も、本実験の行と同じ（付け替えは選んだ S1 の模型で評価した S1-r50・S1-r100）
    for new, old in (("S1-tuned", "S1"), ("S1-tuned-r50", "S1-r50"), ("S1-tuned-r100", "S1-r100"), ("D1-tuned", "D1")):
        (row,) = got["results"][new]["per_seed"]
        for k in ("Recall@K_correct", "MAP", "Recall@20", "n_cases"):
            assert row[k] == main_row(old)[k], (new, k)
        old_cases = [r for r in main[old]["per_case"] if r["seed"] == 42]
        new_cases = got["results"][new]["per_case"]
        assert len(new_cases) == len(old_cases) == 423
        assert [{**r, "mode": old} for r in new_cases] == old_cases, new  # ケースごとの順位まで同じ
    assert got["tau"]["D1"]["42"] == json.load(open(MAIN_A))["tau"]["D1"]["42"]  # Youden の τ0 も同じ
