"""external_baselines.py（A2: BM25・E5 の外部ベースライン）の単体テストと回帰再現テスト.

仕様: docs/superpowers/specs/2026-09-07-a2-external-baselines-design.md
- BM25 は純 Python の参照実装（同じ式）と一致する
- min-max 正規化・Jaccard 行列・被覆率・ラベル生成
- tfidf:none, (0.7, 0.3) は strat_B.json の baseline（seed 42）を完全再現する
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from external_baselines import (
    BM25, minmax_rows, jaccard_matrix, coverage, parse_weights, parse_scorers, label,
    load_setting, evaluate_setting, ranks_of, coverage_from_ranks,
)
from evaluate_multi_eq import compute_all_ranks
from two_stage_query_conditioned import jaccard

ROOT = Path(__file__).resolve().parents[1]


def _ref_bm25(docs, query, k1=1.5, b=0.75):
    """Okapi BM25（Lucene 形式の非負 IDF）の純 Python 参照実装。空白区切りトークン。"""
    toks = [d.split() for d in docs]
    n = len(toks)
    avgdl = sum(len(t) for t in toks) / n
    df = Counter(t for d in toks for t in set(d))
    out = []
    for d in toks:
        tf = Counter(d)
        s = 0.0
        for t in query.split():
            if t not in df:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def test_bm25_matches_reference():
    docs = [
        "mass balance reactor",
        "energy balance heat exchanger reactor reactor",
        "reactor volume flow",
    ]
    bm = BM25(docs)
    query = "reactor balance balance"
    got = bm.score(query)
    exp = _ref_bm25(docs, query)
    assert np.allclose(got, exp, atol=1e-9)
    # 語彙にない語だけのクエリは全文書 0
    assert bm.score("unknownterm").tolist() == [0.0, 0.0, 0.0]
    assert bm.score_many(["reactor", "flow"]).shape == (2, 3)


def test_minmax_rows():
    s = np.array([[1.0, 3.0, 2.0], [5.0, 5.0, 5.0]])
    m = minmax_rows(s)
    assert np.allclose(m[0], [0.0, 1.0, 0.5])
    assert np.allclose(m[1], [0.0, 0.0, 0.0])  # 定数行は全 0


def test_jaccard_matrix_matches_python():
    qs = [{"a", "b"}, set(), {"z"}]
    es = [{"a"}, {"b", "c"}, set(), {"a", "b"}]
    J = jaccard_matrix(qs, es)
    assert J.dtype == np.float32
    assert J.shape == (3, 4)
    for i, q in enumerate(qs):
        for j, e in enumerate(es):
            assert J[i, j] == np.float32(jaccard(q, e))


def test_coverage():
    assert coverage([5, 1, 9, 2], {1, 2, 3}, 2) == 1 / 3
    assert coverage([5, 1, 9, 2], {1, 2, 3}, 4) == 2 / 3
    assert math.isnan(coverage([1], set(), 1))


def test_ranks_of_matches_compute_all_ranks_and_coverage():
    order = np.array([5, 1, 9, 2, 0, 7, 3, 4, 6, 8])
    corr = {1, 2, 3}
    ranks = ranks_of(order, corr)
    assert sorted(ranks) == sorted(compute_all_ranks(order.tolist(), corr))
    for k in (1, 2, 4, 7, 10):
        assert coverage_from_ranks(ranks, k) == coverage(order.tolist(), corr, k)
    assert math.isnan(coverage_from_ranks([], 5))


def test_parsers_and_label():
    assert parse_weights("1:0,0.7:0.3") == [(1.0, 0.0), (0.7, 0.3)]
    assert parse_scorers("tfidf:none,bm25:minmax") == [("tfidf", "none"), ("bm25", "minmax")]
    assert label("bm25", "minmax", 0.7, 0.3) == "bm25__minmax__w70-30"
    assert label("e5", "minmax", 1.0, 0.0) == "e5__minmax__w100-00"


def test_tfidf_none_reproduces_strat_B_seed42():
    """評価枠（分割・指標・混合式）が現行ベースラインと同一であることの回帰確認。"""
    data = load_setting("B")
    results, _ = evaluate_setting(
        data, [("tfidf", "none")], [(0.7, 0.3)], seeds=[42], model_name=None, do_coverage=False,
    )
    got = results["tfidf__none__w70-30"]["per_seed"][0]
    ref_doc = json.load(open(ROOT / "experiments" / "strat_B.json"))
    ref = next(p for p in ref_doc["results"]["baseline"]["per_seed"] if p["seed"] == 42)
    assert got["n_cases"] == ref["n_cases"]
    for m in ["Recall@K_correct", "MAP", "Recall@20"]:
        assert abs(got[m] - ref[m]) < 1e-9, (m, got[m], ref[m])
