"""analyze_split_overlap.py（テスト用のケースが訓練用のケースと正解式を共有する割合）の単体テスト."""
from analyze_split_overlap import ground_truth_ids, shared_with_training, summarize


GT = {"t1": ["s1__a", "s2__b"], "t2": ["s5__k"],
      "x": ["s3__c"],                 # 共有なし
      "y": ["s2__b", "s4__d"],        # 訓練用のケース t1 と式 s2__b を共有
      "z": ["s2__e"],                 # 式は共有しないが、出典 s2 は訓練用の正解式の出典でもある
      "v": ["s1__a"]}


def test_shared_with_training_counts_equations_and_sources_separately():
    out = shared_with_training({"train": ["t1", "t2"], "val": ["v"], "test": ["x", "y", "z"]}, GT)
    assert out == {"n_test": 3, "shared_equation": 1, "shared_source": 2, "cases_shared_equation": ["y"],
                   "cases_shared_source": ["y", "z"]}


def test_shared_with_training_ignores_validation_cases():
    out = shared_with_training({"train": ["t2"], "val": ["t1"], "test": ["y"]}, GT)
    assert out["shared_equation"] == 0 and out["shared_source"] == 0


def test_summarize_pools_the_splits_and_reports_the_range():
    per_seed = {"1": {"n_test": 4, "shared_equation": 1, "shared_source": 2, "cases_shared_equation": ["y"]},
                "2": {"n_test": 6, "shared_equation": 3, "shared_source": 6, "cases_shared_equation": ["y", "p", "q"]}}
    out = summarize(per_seed)
    assert out["n_test_records"] == 10
    assert out["shared_equation"] == 4 and out["shared_equation_share"] == 0.4
    assert out["shared_equation_share_min"] == 0.25 and out["shared_equation_share_max"] == 0.5
    assert out["shared_source_share"] == 0.8


def test_ground_truth_ids_reads_a_list_or_its_string_form():
    assert ground_truth_ids({"correct_model_ids": ["s1__a ", " s2__b"]}) == ["s1__a", "s2__b"]
    assert ground_truth_ids({"correct_model_ids": "['s1__a', 's2__b']"}) == ["s1__a", "s2__b"]
    assert ground_truth_ids({}) == []
