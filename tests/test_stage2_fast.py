"""stage2_fast.py のテスト（第 2 段の特徴量をまとめて計算する高速版）.

高速版は、set_aware_reranker の既定の計算（compute_features_with_set・
compute_set_aware_features・stage1）と同じ値を返さなければならない（N1 = 50 の正典値を
同じ乱数で再現するため）。小さな合成 DB で、値がビット単位で一致することを確かめる。
"""
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from set_aware_reranker import compute_features_with_set, compute_set_aware_features
from stage2_fast import FastBank
from two_stage_query_conditioned import case_text, in_vars, io_vars, out_vars, stage1

DOMAINS = ["CSTR", "cstr", "Heat Exchanger", "", "Distillation", "Bioreactor"]


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(3)
    vocab = [f"x{i}" for i in range(14)] + ["T", "t", "C_A"]
    words = ["reactor", "heat", "flow", "tank", "mass", "balance", "cstr", "temperature",
             "exchanger", "column", "growth", "rate"]
    n_eq = 60
    ev, et, ed = [], [], []
    for i in range(n_eq):
        k = int(rng.integers(0, 5))  # 0 変数の式も含める
        ev.append(set(rng.choice(vocab, size=k, replace=False)) if k else set())
        et.append(" ".join(rng.choice(words, size=6)))
        ed.append(DOMAINS[i % len(DOMAINS)])
    cases = []
    for i in range(8):
        ins = list(rng.choice(vocab, size=int(rng.integers(0, 6)), replace=False))
        outs = list(rng.choice(vocab, size=int(rng.integers(1, 3)), replace=False))
        ctx = " ".join(rng.choice(words + ["cstr", "heat exchanger"], size=10))
        cases.append({"context": ctx, "input_variables": ins, "output_variables": outs})
    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    X_ctx = tfidf.transform([case_text(c) for c in cases])
    svd_sim = rng.random((len(cases), n_eq))
    ios = [io_vars(c) for c in cases]
    bank = FastBank(X_ctx, X_eq, cases, ev, ed, svd_sim)
    return dict(cases=cases, ev=ev, ed=ed, X_eq=X_eq, X_ctx=X_ctx, svd_sim=svd_sim, ios=ios,
                bank=bank, n_eq=n_eq)


def slow_features(t, ci, cands, mask):
    from sklearn.metrics.pairwise import cosine_similarity
    c = t["cases"][ci]
    ts = cosine_similarity(t["X_ctx"][ci], t["X_eq"]).ravel()
    return compute_features_with_set(
        cands, ts, t["ios"][ci], in_vars(c), out_vars(c), t["svd_sim"][ci],
        t["ev"], t["ed"], case_text(c), None, t["ios"][ci], False, mask)


@pytest.mark.parametrize("w", [(0.7, 0.3), (0.3, 0.7), (0.0, 1.0)])
def test_stage1_same_candidates_as_slow(toy, w):
    for ci in range(len(toy["cases"])):
        want = stage1(ci, toy["X_ctx"], toy["X_eq"], toy["ios"][ci], toy["ev"], 20,
                      w_text=w[0], w_var=w[1])
        assert toy["bank"].stage1(ci, 20, w[0], w[1]) == want


@pytest.mark.parametrize("mask", [(), ("Comp", "Coh", "Dom"), ("Coh",), ("Comp", "Coh", "Dom", "OutNov")])
def test_features_bitwise_equal_to_slow(toy, mask):
    for ci in range(len(toy["cases"])):
        cands = toy["bank"].stage1(ci, 25, 0.3, 0.7)
        got = toy["bank"].features(ci, cands, mask)
        want = slow_features(toy, ci, cands, mask)
        assert got.dtype == np.float32 and got.shape == want.shape
        assert np.array_equal(got, want), f"case {ci} mask {mask}"


def test_set_features_with_reference_set_bitwise_equal(toy):
    """逐次選択の参照集合（候補の一部を含む）でも、自分を除く扱いまで一致する."""
    for ci in range(len(toy["cases"])):
        cands = toy["bank"].stage1(ci, 30, 0.3, 0.7)
        for ref in (cands[2:5], [cands[0], cands[9], cands[17]], cands[:1]):
            c = toy["cases"][ci]
            want = compute_set_aware_features(cands, toy["ios"][ci], toy["ev"], toy["ed"],
                                              output_vars=out_vars(c), ref_indices=ref)
            got = toy["bank"].set_features(ci, cands, ref)
            assert np.array_equal(got, want), f"case {ci} ref {ref}"


def test_full_db_candidates(toy):
    """第 1 段を外す（N1 = DB の全式）場合も同じ値になる."""
    ci = 0
    cands = toy["bank"].stage1(ci, toy["n_eq"], 0.3, 0.7)
    assert sorted(cands) == list(range(toy["n_eq"]))
    mask = ("Comp", "Coh", "Dom")
    assert np.array_equal(toy["bank"].features(ci, cands, mask), slow_features(toy, ci, cands, mask))


# ---------------------------------------------------------------------------
# run_mode 全体：--fast は既定の計算と同じ結果（同じ乱数の消費）になる
# ---------------------------------------------------------------------------

def _toy_run_inputs(n_eq=320, n_case=40, seed=5):
    # SVD は 256 次元なので、TF-IDF の語彙が 256 を超える大きさにする
    rng = np.random.default_rng(seed)
    vocab = [f"x{i}" for i in range(18)] + ["T", "t"]
    words = [f"w{i}" for i in range(60)] + ["cstr", "heat", "exchanger"]
    srcs = [f"src{i}" for i in range(10)]
    ek, et, ev, ed, es = [], [], [], [], []
    for i in range(n_eq):
        s = srcs[i % len(srcs)]
        ek.append(f"{s}__eq{i}")
        et.append(" ".join(rng.choice(words, size=6)))
        k = int(rng.integers(1, 5))
        ev.append(set(rng.choice(vocab, size=k, replace=False)))
        ed.append(DOMAINS[i % len(DOMAINS)])
        es.append(s)
    cases, cl, cs = [], [], []
    for i in range(n_case):
        corr = sorted(rng.choice(n_eq, size=int(rng.integers(1, 4)), replace=False).tolist())
        vs = sorted(set().union(*(ev[j] for j in corr)))
        outs = vs[:1]
        ins = vs[1:]
        cases.append({"case_id": f"c{i}", "variant_type": "dae_X1",
                      "context": " ".join(rng.choice(words, size=9)),
                      "input_variables": ins, "output_variables": outs,
                      "correct_model_ids": [ek[j] for j in corr]})
        cl.append(corr)
        cs.append(ek[corr[0]].split("__")[0])
    return cases, ek, et, ev, ed, es, cl, cs


@pytest.mark.parametrize("kw", [
    dict(),
    dict(hard_neg=True),
    dict(greedy=True),
    dict(train_greedy=True),
    dict(neg_pool=6),
])
def test_run_mode_fast_equals_default(kw):
    from set_aware_reranker import run_mode
    cases, ek, et, ev, ed, es, cl, cs = _toy_run_inputs()
    common = dict(seeds=[42], cases=cases, eq_keys_l=ek, eq_texts_l=et, eq_vars_l=ev,
                  eq_domains_l=ed, eq_sources_l=es, correct_lists=cl, case_srcs=cs,
                  top_k=15, epochs=2, lr=1e-3, save_per_case=True, split_mode="stratified",
                  stage1_w_text=0.3, stage1_w_var=0.7)
    slow = run_mode("reranker-10S", **common, **kw)
    fast = run_mode("reranker-10S", **common, fast=True, **kw)
    assert fast["per_case"] == slow["per_case"]


def test_run_mode_fast_ablation_subset_equals_default():
    from set_aware_reranker import run_mode
    cases, ek, et, ev, ed, es, cl, cs = _toy_run_inputs()
    common = dict(seeds=[42], cases=cases, eq_keys_l=ek, eq_texts_l=et, eq_vars_l=ev,
                  eq_domains_l=ed, eq_sources_l=es, correct_lists=cl, case_srcs=cs,
                  top_k=15, epochs=2, lr=1e-3, save_per_case=True, split_mode="stratified")
    for mode in ("reranker-2+Spec", "reranker-7+Dom"):
        assert run_mode(mode, **common, fast=True)["per_case"] == run_mode(mode, **common)["per_case"]


def test_run_mode_on_trained_hands_over_the_trained_model():
    """on_trained には評価に使うのと同じモデルと候補・特徴量の関数が渡る（PFI 用）."""
    import torch
    from set_aware_reranker import run_mode
    from evaluate_multi_eq import compute_all_ranks
    cases, ek, et, ev, ed, es, cl, cs = _toy_run_inputs()
    got = {}

    def hook(ctx):
        got.update(ctx)
    r = run_mode("reranker-10S", [42], cases, ek, et, ev, ed, es, cl, cs, 15, 2, 1e-3,
                 save_per_case=True, split_mode="stratified", fast=True, on_trained=hook)
    assert got["seed"] == 42 and got["te"]
    want = {rec["case_id"]: rec["ranks"] for rec in r["per_case"]}
    with torch.no_grad():
        for ci in got["te"]:
            cands = got["get_cands"](ci)
            s = got["model"](torch.tensor(got["get_feats"](ci, cands))).numpy().ravel()
            ranked = [cands[k] for k in sorted(range(len(cands)), key=lambda k: -s[k])]
            ranks = sorted(compute_all_ranks(ranked, set(cl[ci]), miss_rank=10_000))
            assert ranks == want[cases[ci]["case_id"]]


def test_train_greedy_uses_hard_negatives_when_asked():
    """--train-greedy に --hard-neg を付けると、逐次の各段階でも点数の高い不正解を選ぶ（付けないときと結果が変わる）."""
    from set_aware_reranker import run_mode
    cases, ek, et, ev, ed, es, cl, cs = _toy_run_inputs()
    common = dict(seeds=[42], cases=cases, eq_keys_l=ek, eq_texts_l=et, eq_vars_l=ev,
                  eq_domains_l=ed, eq_sources_l=es, correct_lists=cl, case_srcs=cs,
                  top_k=15, epochs=2, lr=1e-3, save_per_case=True, split_mode="stratified",
                  train_greedy=True)
    params = {}

    def keep(name):
        def hook(ctx):
            params[name] = [p.detach().clone() for p in ctx["model"].parameters()]
        return hook
    run_mode("reranker-10S", **common, fast=True, on_trained=keep("rand"))
    hard_fast = run_mode("reranker-10S", **common, fast=True, hard_neg=True, on_trained=keep("hard"))
    hard_slow = run_mode("reranker-10S", **common, hard_neg=True)
    assert hard_fast["per_case"] == hard_slow["per_case"]
    # 訓練が変わる（訓練後の重みが無作為の不正解のときと異なる）
    assert any(not (a == b).all() for a, b in zip(params["hard"], params["rand"]))
