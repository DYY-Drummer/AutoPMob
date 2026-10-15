"""kato1014_domain_merge.py のテスト（§8 の分野ラベルの合併）.

旧版（run_domain_merge_ablation.py）は式のラベルを群の ID（"C12" など）に置き換えたので、
説明文の部分文字列で比べる x7 がほとんど 1 にならず、合併の効果を測れていなかった。
新版の x7 は「説明文が、式の群に入るどれかのラベルを（元の規則で）含めば 1」とする。
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from kato1014_domain_merge import make_cluster_bank
from stage2_fast import FastBank
from two_stage_query_conditioned import case_text

LABELS = ["CSTR", "Continuous Stirred Tank Reactor", "Heat Exchanger", "", "Distillation"]


def _toy():
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(30)]
    n_eq = 25
    ev = [set(rng.choice(["a", "b", "c", "T"], size=2, replace=False)) for _ in range(n_eq)]
    ed = [LABELS[i % len(LABELS)] for i in range(n_eq)]
    et = [" ".join(rng.choice(words, size=5)) for _ in range(n_eq)]
    cases = [{"context": "A cstr with a cooling jacket", "input_variables": ["a"], "output_variables": ["T"]},
             {"context": "distillation column dynamics", "input_variables": ["b"], "output_variables": ["c"]}]
    tf = TfidfVectorizer().fit(et)
    X_eq, X_ctx = tf.transform(et), tf.transform([case_text(c) for c in cases])
    return X_ctx, X_eq, cases, ev, ed, rng.random((len(cases), n_eq))


def test_identity_mapping_reproduces_original_x7():
    X_ctx, X_eq, cases, ev, ed, sv = _toy()
    ident = {d: d for d in LABELS if d}
    a = FastBank(X_ctx, X_eq, cases, ev, ed, sv)
    b = make_cluster_bank(ident)(X_ctx, X_eq, cases, ev, ed, sv)
    for ci in range(len(cases)):
        cands = list(range(len(ev)))
        assert np.array_equal(a.features(ci, cands, ("Comp", "Coh", "Dom")),
                              b.features(ci, cands, ("Comp", "Coh", "Dom")))


def test_x7_fires_for_any_label_in_the_same_cluster():
    X_ctx, X_eq, cases, ev, ed, sv = _toy()
    merged = {"CSTR": "C0", "Continuous Stirred Tank Reactor": "C0", "Heat Exchanger": "C1", "Distillation": "C2"}
    bank = make_cluster_bank(merged)(X_ctx, X_eq, cases, ev, ed, sv)
    F = bank.features(0, list(range(len(ev))))
    for j, d in enumerate(ed):
        # 説明文は "cstr" を含む → 群 C0（CSTR と Continuous Stirred Tank Reactor）の式は 1
        assert F[j, 6] == (1.0 if d in ("CSTR", "Continuous Stirred Tank Reactor") else 0.0), d
