"""
数式グラフ・訓練ケースの可視化とデータ分析。
全体像とデータの特徴を把握し、論文付録用のレポート (dataset_report.txt) を出力する。

出力:
  - dataset_report.txt   : 統計サマリ（論文付録向けフォーマット）
  - figures/             : 各種 PNG（ドメイン分布、次数分布、埋め込みクラスター、変数ネットワーク等）

実行: python analyze_dataset.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
GRAPH_PT = ROOT / "equation_graph.pt"
TRAINING_JSON = ROOT / "training_cases.json"
REPORT_TXT = ROOT / "dataset_report.txt"
FIGURES_DIR = ROOT / "figures"


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------


def load_equations() -> list[dict[str, Any]]:
    if not UNIFIED_JSON.is_file():
        return []
    with open(UNIFIED_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "equations" in raw:
        return raw["equations"]
    return raw.get("papers", raw) if isinstance(raw, dict) else []


def load_graph():
    if not GRAPH_PT.is_file():
        return None
    import torch
    return torch.load(GRAPH_PT, weights_only=False)


def load_training_cases() -> list[dict[str, Any]]:
    if not TRAINING_JSON.is_file():
        return []
    with open(TRAINING_JSON, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 統計計算
# ---------------------------------------------------------------------------


def analyze_equations(equations: list[dict]) -> dict[str, Any]:
    if not equations:
        return {"n_equations": 0}
    domains = Counter(e.get("domain") or "unknown" for e in equations)
    var_freq: Counter[str] = Counter()
    var_to_definition: dict[str, str] = {}  # 変数シンボル -> 代表的な物理意味（最初に出現した定義）
    n_vars_per_eq = []
    source_ids = Counter(e.get("source_id") or "unknown" for e in equations)
    for e in equations:
        vars_dict = e.get("variables") or {}
        if isinstance(vars_dict, dict):
            keys = list(vars_dict.keys())
            n_vars_per_eq.append(len(keys))
            for v in keys:
                var_freq[v] += 1
                if v not in var_to_definition and isinstance(vars_dict.get(v), str):
                    var_to_definition[v] = (vars_dict[v] or "").strip()
    return {
        "n_equations": len(equations),
        "domain_counts": dict(domains.most_common()),
        "variable_frequency": dict(var_freq.most_common(30)),
        "variable_definitions": var_to_definition,
        "n_variables_per_equation": n_vars_per_eq,
        "source_id_counts": dict(source_ids.most_common(20)),
        "unique_variables": len(var_freq),
    }


def analyze_graph(data) -> dict[str, Any] | None:
    if data is None:
        return None
    try:
        import torch
        x = getattr(data, "x", None)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None or x is None:
            return None
        edge_index = edge_index if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index)
        n = x.shape[0] if hasattr(x, "shape") else 0
        if edge_index.dim() == 2 and edge_index.shape[0] == 2:
            e = edge_index.shape[1]
            # 無向なのでエッジ数は実際のリンク数（双方向の片方だけ数えるなら e//2）
        else:
            e = 0
        degree = defaultdict(int)
        for i in range(edge_index.shape[1]):
            u = int(edge_index[0, i].item())
            degree[u] += 1
        deg_list = list(degree.values()) if degree else [0] * n
        return {
            "n_nodes": n,
            "n_edges": e,
            "degree_min": min(deg_list) if deg_list else 0,
            "degree_max": max(deg_list) if deg_list else 0,
            "degree_mean": sum(deg_list) / len(deg_list) if deg_list else 0,
            "degree_histogram": dict(Counter(deg_list)),
            "x": x,
            "edge_index": edge_index,
        }
    except Exception:
        return None


def analyze_training_cases(cases: list[dict]) -> dict[str, Any]:
    if not cases:
        return {"n_cases": 0}
    variant_counts = Counter(c.get("variant_type") or "unknown" for c in cases)
    n_models = [len(c.get("correct_model_ids") or []) for c in cases]
    n_in = [len(c.get("input_variables") or []) for c in cases]
    n_out = [len(c.get("output_variables") or []) for c in cases]
    return {
        "n_cases": len(cases),
        "variant_type_counts": dict(variant_counts),
        "correct_model_ids_per_case": n_models,
        "input_variables_per_case": n_in,
        "output_variables_per_case": n_out,
    }


# ---------------------------------------------------------------------------
# ドメイン合併（論文用に表示ドメイン数を削減）
# ---------------------------------------------------------------------------

# キーワードにマッチしたらその表示ドメインに合併。上から順にマッチさせる。
DOMAIN_MERGE_RULES: list[tuple[list[str], str]] = [
    (["CSTR", "Stirred Tank", "Reactor", "Tubular Reactor", "Plug Flow", "Fed-Batch", "Isothermal CSTR", "Nonisothermal CSTR", "Two CSTRs"], "Reactor / CSTR"),
    (["Kinetics", "Transesterification", "Biodiesel"], "Reaction Kinetics"),
    (["Heat Transfer", "Heat Exchanger", "Heating", "Thermodynamics", "Thermochemistry", "Jacket", "Steam-Heated", "Cooling Coil"], "Heat Transfer / Thermodynamics"),
    (["Control", "PID", "Laplace", "Transfer Function", "State-Space", "Linear Systems", "Signal", "Frequency Response", "Block Diagram", "Stability"], "Control / Dynamics"),
    (["Material Balance", "Mole Balance", "Component continuity", "Overall Material Balance"], "Material Balance"),
    (["Energy Balance"], "Energy Balance"),
    (["Fluid", "Flow", "Tank Draining", "Surge Tank", "Weir", "CFD"], "Fluid / Flow"),
    (["Numerical", "Euler", "Runge-Kutta", "ODE"], "Numerical Methods"),
    (["Electrical", "Circuit"], "Electrical Circuits"),
    (["Mechanical", "Torque", "Electromechanical"], "Mechanical / Electromechanical"),
    (["Membrane", "Blending", "Staged", "Absorber"], "Separation / Mixing"),
    (["Linearization", "Taylor", "Degrees of Freedom", "Dimensionless"], "Modeling / Linearization"),
    (["Fuzzy", "Statistical"], "Other"),  # 論文では Other に統合
    (["General", "Modeling Principles", "Mathematical Modeling", "Conservation Balance"], "General Modeling"),
]

def merge_domain(raw_domain: str) -> str:
    """細かいドメイン名を論文用の表示ドメインに合併する。"""
    if not raw_domain or raw_domain == "unknown":
        return "Unknown"
    d = raw_domain.strip()
    for keywords, display_name in DOMAIN_MERGE_RULES:
        if any(kw.lower() in d.lower() for kw in keywords):
            return display_name
    return "Other"


def get_merged_domain_counts(equations: list[dict]) -> tuple[Counter[str], list[str]]:
    """合併ドメインのカウントと、表示順（カウント降順）のラベルリストを返す。"""
    merged = Counter(merge_domain(e.get("domain")) for e in equations)
    order = [k for k, _ in merged.most_common()]
    return merged, order


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------


def ensure_figures_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_domain_distribution(equations: list[dict], figure_refs: list[str]) -> None:
    if not equations:
        return
    merged_counts, order = get_merged_domain_counts(equations)
    labels = order
    counts = [merged_counts[l] for l in labels]
    n_bars = len(labels)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_h = max(5, min(12, n_bars * 0.45))
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.barh(labels, counts, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_xlabel("Count")
        ax.set_ylabel("Domain (merged)")
        ax.set_title("Equation domain distribution (merged domains)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "domain_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        figure_refs.append("domain_distribution.png  — Equation domain distribution (merged, bar chart).")
    except Exception as e:
        figure_refs.append(f"[domain_distribution.png failed: {e}]")


def plot_degree_histogram(graph_stats: dict | None, figure_refs: list[str]) -> None:
    if not graph_stats or "degree_histogram" not in graph_stats:
        return
    hist = graph_stats["degree_histogram"]
    if not hist:
        return
    degs = sorted(hist.keys())
    counts = [hist[d] for d in degs]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(degs, counts, color="teal", edgecolor="darkgreen", alpha=0.8)
        ax.set_xlabel("Degree (number of neighbors)")
        ax.set_ylabel("Number of nodes")
        ax.set_title("Equation graph: node degree distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "degree_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        figure_refs.append("degree_histogram.png  — Node degree distribution of the equation graph.")
    except Exception as e:
        figure_refs.append(f"[degree_histogram.png failed: {e}]")


def plot_equation_embeddings_2d(
    graph_stats: dict | None,
    equations: list[dict],
    figure_refs: list[str],
) -> None:
    if not graph_stats or "x" not in graph_stats or not equations:
        return
    x = graph_stats["x"]
    try:
        from sklearn.decomposition import PCA
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X = x.numpy() if hasattr(x, "numpy") else x
        if X.shape[0] != len(equations):
            return
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        merged_domains = [merge_domain(e.get("domain")) for e in equations]
        uniq_dom = [k for k, _ in Counter(merged_domains).most_common()]
        cmap = plt.cm.tab20
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, d in enumerate(uniq_dom):
            mask = [j for j in range(len(merged_domains)) if merged_domains[j] == d]
            ax.scatter(
                X2[mask, 0], X2[mask, 1],
                label=d,
                alpha=0.7,
                s=22,
                color=cmap(i % 20),
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Equation embeddings (PCA, colored by merged domain)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, ncols=1)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "equation_embeddings_2d.png", dpi=150, bbox_inches="tight")
        plt.close()
        figure_refs.append("equation_embeddings_2d.png  — PCA of equation embeddings; points colored by merged domain.")
    except Exception as e:
        figure_refs.append(f"[equation_embeddings_2d.png failed: {e}]")


def build_variable_cooccurrence(equations: list[dict]) -> tuple[dict[tuple[str, str], int], Counter[str]]:
    """変数ペアの共起回数、および変数ごとの次数（共起した変数の数）を返す。"""
    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    var_degree: Counter[str] = Counter()
    for e in equations:
        vars_dict = e.get("variables") or {}
        if not isinstance(vars_dict, dict):
            continue
        keys = list(vars_dict.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if a > b:
                    a, b = b, a
                pair_count[(a, b)] += 1
                var_degree[a] += 1
                var_degree[b] += 1
    return dict(pair_count), var_degree


def plot_variable_network(
    equations: list[dict],
    figure_refs: list[str],
    top_n: int = 60,
) -> None:
    """変数共起ネットワーク（上位 top_n 変数）を可視化。"""
    if not equations:
        return
    pair_count, var_degree = build_variable_cooccurrence(equations)
    top_vars = [v for v, _ in var_degree.most_common(top_n)]
    top_set = set(top_vars)
    edges = [(u, v, w) for (u, v), w in pair_count.items() if u in top_set and v in top_set and w >= 1]
    if not edges:
        return
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        pos = nx.spring_layout(G, k=1.5, seed=42, iterations=50)
        fig, ax = plt.subplots(figsize=(12, 10))
        nx.draw_networkx_nodes(
            G, pos, node_color="lightcoral", node_size=80, alpha=0.9, ax=ax
        )
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        labels_short = {n: n[:12] + "…" if len(n) > 12 else n for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels_short, font_size=5, ax=ax)
        ax.set_title(f"Variable co-occurrence network (top {top_n} variables by degree)")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "variable_network.png", dpi=150, bbox_inches="tight")
        plt.close()
        figure_refs.append("variable_network.png  — Variable co-occurrence network (same equation); top variables by degree.")
    except ImportError:
        figure_refs.append("[variable_network.png skipped — install networkx to generate]")
    except Exception as e:
        figure_refs.append(f"[variable_network.png failed: {e}]")


def plot_training_variant_distribution(tc_stats: dict | None, figure_refs: list[str]) -> None:
    if not tc_stats or "variant_type_counts" not in tc_stats:
        return
    vc = tc_stats["variant_type_counts"]
    if not vc:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(vc.keys()), list(vc.values()), color="coral", edgecolor="darkred", alpha=0.8)
        ax.set_xlabel("Variant type")
        ax.set_ylabel("Count")
        ax.set_title("Training cases: variant type distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "training_variant_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        figure_refs.append("training_variant_distribution.png  — Training case variant type (original / context_paraphrased / swap_io / random_io_from_models).")
    except Exception as e:
        figure_refs.append(f"[training_variant_distribution.png failed: {e}]")


def plot_training_model_count_histogram(tc_stats: dict | None, figure_refs: list[str]) -> None:
    if not tc_stats or "correct_model_ids_per_case" not in tc_stats:
        return
    counts = tc_stats["correct_model_ids_per_case"]
    if not counts:
        return
    hist = Counter(counts)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = sorted(hist.keys())
        ys = [hist[x] for x in xs]
        ax.bar(xs, ys, color="seagreen", edgecolor="darkgreen", alpha=0.8)
        ax.set_xlabel("Number of correct model equations per case")
        ax.set_ylabel("Number of training cases")
        ax.set_title("Training cases: correct_model_ids count distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "training_model_count_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        figure_refs.append("training_model_count_histogram.png  — Number of correct model equations per training case.")
    except Exception as e:
        figure_refs.append(f"[training_model_count_histogram.png failed: {e}]")


# ---------------------------------------------------------------------------
# レポート生成（論文付録向け）
# ---------------------------------------------------------------------------


def write_report(
    eq_stats: dict,
    graph_stats: dict | None,
    tc_stats: dict,
    figure_refs: list[str],
) -> None:
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("  Dataset Statistics: Equation Graph and Training Cases")
    lines.append("  (Appendix-style report for manuscript reuse)")
    lines.append("=" * 70)
    lines.append("")

    # Section 1: Equation dataset
    lines.append("1.  Equation Dataset (unified_equations.json)")
    lines.append("-" * 50)
    lines.append(f"   Total number of equations (nodes):  {eq_stats.get('n_equations', 0)}")
    lines.append(f"   Unique variable symbols (across all equations):  {eq_stats.get('unique_variables', 0)}")
    if eq_stats.get("n_variables_per_equation"):
        arr = eq_stats["n_variables_per_equation"]
        lines.append(f"   Variables per equation:  min = {min(arr)}, max = {max(arr)}, mean = {sum(arr)/len(arr):.2f}")
    lines.append("")
    lines.append("   Table 1. Domain distribution (top 20).")
    lines.append("   " + "-" * 40)
    for i, (dom, cnt) in enumerate(list(eq_stats.get("domain_counts", {}).items())[:20], 1):
        lines.append(f"   {i:2d}. {dom[:45]:<45}  {cnt:>6}")
    lines.append("")
    lines.append("   Table 2. Top 15 variable symbols by frequency, with physical meaning.")
    lines.append("   " + "-" * 40)
    var_defs = eq_stats.get("variable_definitions") or {}
    for i, (var, cnt) in enumerate(list(eq_stats.get("variable_frequency", {}).items())[:15], 1):
        meaning = (var_defs.get(var) or "(no definition)")[:60]
        if len(var_defs.get(var) or "") > 60:
            meaning += "..."
        lines.append(f"   {i:2d}. {str(var)[:25]:<25}  {cnt:>5}  |  {meaning}")
    for ref in figure_refs:
        if "domain_distribution" in ref or "variable_network" in ref:
            lines.append(f"   {ref}")
    lines.append("")
    lines.append("")

    # Section 2: Equation graph
    lines.append("2.  Equation Graph (equation_graph.pt)")
    lines.append("-" * 50)
    if graph_stats:
        lines.append(f"   Number of nodes:  {graph_stats.get('n_nodes', 0)}")
        lines.append(f"   Number of edges:  {graph_stats.get('n_edges', 0)}")
        lines.append(f"   Node degree:  min = {graph_stats.get('degree_min', 0)}, max = {graph_stats.get('degree_max', 0)}, mean = {graph_stats.get('degree_mean', 0):.2f}")
    else:
        lines.append("   (Graph file not found or not loaded; run build_graph_data.py to generate equation_graph.pt.)")
    for ref in figure_refs:
        if "degree_histogram" in ref or "embedding" in ref.lower():
            lines.append(f"   {ref}")
    lines.append("")
    lines.append("")

    # Section 3: Training cases
    lines.append("3.  Training Cases (training_cases.json)")
    lines.append("-" * 50)
    lines.append(f"   Total number of training cases:  {tc_stats.get('n_cases', 0)}")
    if tc_stats.get("variant_type_counts"):
        lines.append("   Table 3. Variant type distribution.")
        lines.append("   " + "-" * 40)
        for k, v in tc_stats["variant_type_counts"].items():
            lines.append(f"      {k:<35}  {v:>6}")
    if tc_stats.get("correct_model_ids_per_case"):
        arr = tc_stats["correct_model_ids_per_case"]
        lines.append(f"   Correct model equations per case:  min = {min(arr)}, max = {max(arr)}, mean = {sum(arr)/len(arr):.2f}")
    if tc_stats.get("input_variables_per_case"):
        arr = tc_stats["input_variables_per_case"]
        lines.append(f"   Input variables per case:  min = {min(arr)}, max = {max(arr)}, mean = {sum(arr)/len(arr):.2f}")
    if tc_stats.get("output_variables_per_case"):
        arr = tc_stats["output_variables_per_case"]
        lines.append(f"   Output variables per case:  min = {min(arr)}, max = {max(arr)}, mean = {sum(arr)/len(arr):.2f}")
    for ref in figure_refs:
        if "training" in ref.lower() or "variant" in ref.lower() or "model_count" in ref.lower():
            lines.append(f"   {ref}")
    lines.append("")
    lines.append("")

    # Section 4: Figures summary
    lines.append("4.  Generated figures (figures/)")
    lines.append("-" * 50)
    for ref in figure_refs:
        lines.append(f"   {ref}")
    lines.append("")
    lines.append("=" * 70)

    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {REPORT_TXT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_figures_dir()
    equations = load_equations()
    data = load_graph()
    cases = load_training_cases()

    eq_stats = analyze_equations(equations)
    graph_stats = analyze_graph(data)
    tc_stats = analyze_training_cases(cases)

    figure_refs: list[str] = []

    # Plots (order matches report sections where possible)
    plot_domain_distribution(equations, figure_refs)
    plot_degree_histogram(graph_stats, figure_refs)
    plot_equation_embeddings_2d(graph_stats, equations, figure_refs)
    plot_variable_network(equations, figure_refs)
    plot_training_variant_distribution(tc_stats, figure_refs)
    plot_training_model_count_histogram(tc_stats, figure_refs)

    write_report(eq_stats, graph_stats, tc_stats, figure_refs)
    print("Analysis complete. See dataset_report.txt and figures/.")


if __name__ == "__main__":
    main()
