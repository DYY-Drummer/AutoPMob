import json

from analyze_significance import load_per_seed, paired_stats


def test_paired_stats_known_values():
    base = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.52, 0.47, 0.51]
    rer = [0.70, 0.74, 0.69, 0.72, 0.71, 0.75, 0.73, 0.74, 0.66, 0.72]
    s = paired_stats(base, rer)
    assert s["n"] == 10
    # mean(rer)=0.716, mean(base)=0.503 → delta=0.213
    assert abs(s["mean_delta"] - 0.213) < 1e-9
    # 10ペア全てで rer > base のとき Wilcoxon 両側の最小 p = 2/2^10 ≈ 0.00195
    assert abs(s["p_wilcoxon"] - 0.00195) < 1e-4
    assert s["p_ttest"] < 0.001
    assert s["cohen_dz"] > 3.0


def test_load_per_seed(tmp_path):
    doc = {"config": {}, "results": {"baseline": {"per_seed": [
        {"seed": 42, "Recall@K_correct": 0.5, "MAP": 0.6, "Recall@20": 0.8},
        {"seed": 123, "Recall@K_correct": 0.4, "MAP": 0.5, "Recall@20": 0.7},
    ]}}}
    p = tmp_path / "x.json"
    p.write_text(json.dumps(doc))
    assert load_per_seed(p, "baseline", "Recall@K_correct") == [0.5, 0.4]
    assert load_per_seed(p, "baseline", "MAP") == [0.6, 0.5]
