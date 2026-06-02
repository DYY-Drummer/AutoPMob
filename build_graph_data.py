"""
unified_equations.json から PyTorch Geometric で学習可能なグラフデータ
（equation_graph.pt）を構築する前処理スクリプト。

【二部グラフ版】GNNの強み（タスクに効く構造・局所集約）を活かすため、
式ノードと変数ノードの二部グラフを採用する。

- 式ノード: 0..N_eq-1（各 Equation）、特徴は CodeBERT [CLS] 768次元
- 変数ノード: N_eq..N_eq+N_var-1、特徴はその変数を含む式の CodeBERT 平均
- エッジ: 式 i が変数 v を含むときのみ (i, N_eq+j) と (N_eq+j, i) を双方向に追加
- 式どうしは変数ノードを介した2ホップで情報が伝わる
- 後方互換のため edge_index_eq_eq（式同士の隣接）も保存
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Set, Tuple

import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel


ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
OUTPUT_PT = ROOT / "equation_graph.pt"
CODEBERT_MODEL_NAME = "microsoft/codebert-base"


def load_equations(path: Path) -> List[Dict]:
    if not path.is_file():
        raise FileNotFoundError(f"unified_equations.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        equations = data
    elif isinstance(data, dict) and "equations" in data:
        equations = data["equations"]
    else:
        raise ValueError("unified_equations.json has unexpected format.")
    return equations


def build_node_features(equations: List[Dict]) -> torch.Tensor:
    """
    CodeBERT (microsoft/codebert-base) で各式の [CLS] ベクトルを取得。
    """
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(CODEBERT_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts: List[str] = []
    for eq in equations:
        eq_str = str(eq.get("equation") or "")
        ctx = str(eq.get("context_text") or "")
        text = f"Equation: {eq_str}\nContext: {ctx}"
        texts.append(text)

    batch_size = 16
    cls_embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
            cls_embeddings.append(cls.cpu())

    x_eq = torch.cat(cls_embeddings, dim=0)
    return x_eq


def build_bipartite_graph(equations: List[Dict], x_eq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], torch.Tensor]:
    """
    式-変数二部グラフを構築する。
    戻り値: (x_all, edge_index_bipartite, var_name_to_id, edge_index_eq_eq)
    """
    n_eq = len(equations)
    var_name_to_id: Dict[str, int] = {}
    eq_id_to_vars: List[Set[int]] = []  # eq_id -> set of var_id

    for eq in equations:
        vars_dict = eq.get("variables") or {}
        if not isinstance(vars_dict, dict):
            eq_id_to_vars.append(set())
            continue
        var_ids: Set[int] = set()
        for var_name in vars_dict.keys():
            key = str(var_name).strip()
            if not key:
                continue
            if key not in var_name_to_id:
                var_name_to_id[key] = len(var_name_to_id)
            var_ids.add(var_name_to_id[key])
        eq_id_to_vars.append(var_ids)

    n_var = len(var_name_to_id)
    # 変数ノードの特徴: その変数を含む式の CodeBERT 平均
    var_agg = torch.zeros(n_var, x_eq.shape[1])
    var_deg = torch.zeros(n_var)
    for eq_i, var_ids in enumerate(eq_id_to_vars):
        for v in var_ids:
            var_agg[v] += x_eq[eq_i]
            var_deg[v] += 1
    var_deg = var_deg.clamp(min=1)
    x_var = var_agg / var_deg.unsqueeze(1)
    x_all = torch.cat([x_eq, x_var], dim=0)  # [n_eq + n_var, 768]

    # 二部グラフのエッジ: 式 i が変数 v を含む -> (i, n_eq+v) と (n_eq+v, i)
    edges_bipartite: List[Tuple[int, int]] = []
    for eq_i, var_ids in enumerate(eq_id_to_vars):
        for v in var_ids:
            edges_bipartite.append((eq_i, n_eq + v))
            edges_bipartite.append((n_eq + v, eq_i))

    if not edges_bipartite:
        edge_index_bipartite = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_bipartite = torch.tensor(edges_bipartite, dtype=torch.long).t().contiguous()

    # 後方互換: 式同士の隣接（変数を1つでも共有するペア）
    var_to_eqs: Dict[int, List[int]] = {}
    for eq_i, var_ids in enumerate(eq_id_to_vars):
        for v in var_ids:
            var_to_eqs.setdefault(v, []).append(eq_i)
    edges_eq_eq: Set[Tuple[int, int]] = set()
    for eq_list in var_to_eqs.values():
        for i in range(len(eq_list)):
            for j in range(i + 1, len(eq_list)):
                u, v = eq_list[i], eq_list[j]
                if u != v:
                    edges_eq_eq.add((u, v))
                    edges_eq_eq.add((v, u))
    if not edges_eq_eq:
        edge_index_eq_eq = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_eq_eq = torch.tensor(list(edges_eq_eq), dtype=torch.long).t().contiguous()

    return x_all, edge_index_bipartite, var_name_to_id, edge_index_eq_eq


def build_graph() -> Data:
    equations = load_equations(UNIFIED_JSON)
    print(f"Loaded {len(equations)} equations from {UNIFIED_JSON}")

    x_eq = build_node_features(equations)
    print(f"Equation features x_eq shape: {tuple(x_eq.shape)}")

    x_all, edge_index_bipartite, var_name_to_id, edge_index_eq_eq = build_bipartite_graph(equations, x_eq)
    n_var = len(var_name_to_id)
    n_eq = len(equations)
    print(f"Variables: {n_var}, bipartite edges: {edge_index_bipartite.shape[1]}, eq-eq edges (legacy): {edge_index_eq_eq.shape[1]}")

    data = Data(
        x=x_all,
        edge_index=edge_index_bipartite,
        num_equations=n_eq,
        edge_index_eq_eq=edge_index_eq_eq,
    )
    return data


def main() -> None:
    data = build_graph()
    torch.save(data, OUTPUT_PT)
    print(f"Saved equation graph to {OUTPUT_PT}")


if __name__ == "__main__":
    main()
