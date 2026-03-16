"""
unified_equations.json から PyTorch Geometric で学習可能なグラフデータ
（equation_graph.pt）を構築する前処理スクリプト。

- ノード: 各 Equation（0..N-1 のノードIDを付与）
- エッジ: variables のキー（変数名）に 1つでも共通があれば無向エッジ
- 特徴量 x: CodeBERT (microsoft/codebert-base) の [CLS] ベクトル（768次元）
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
    # unified_equations.json は list か {papers: ...} のどちらかと想定
    if isinstance(data, list):
        equations = data
    elif isinstance(data, dict) and "equations" in data:
        equations = data["equations"]
    else:
        raise ValueError("unified_equations.json has unexpected format.")
    return equations


def build_edges(equations: List[Dict]) -> torch.Tensor:
    """
    variables のキー（変数名）が 1 つでも共通するノード間に無向エッジを張る。
    """
    var_to_nodes: Dict[str, List[int]] = {}
    for idx, eq in enumerate(equations):
        vars_dict = eq.get("variables") or {}
        if not isinstance(vars_dict, dict):
            continue
        for var_name in vars_dict.keys():
            key = str(var_name)
            var_to_nodes.setdefault(key, []).append(idx)

    edges: Set[Tuple[int, int]] = set()
    for nodes in var_to_nodes.values():
        if len(nodes) < 2:
            continue
        # 完全グラフ（すべてのペア）を生成
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if u == v:
                    continue
                edges.add((u, v))
                edges.add((v, u))  # 無向グラフ → 双方向エッジ

    if not edges:
        # エッジが無い場合でも形を保つ（empty tensor）
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index


def build_node_features(equations: List[Dict]) -> torch.Tensor:
    """
    CodeBERT (microsoft/codebert-base) で各ノードの [CLS] ベクトルを取得。
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

    x = torch.cat(cls_embeddings, dim=0)
    return x


def build_graph() -> Data:
    equations = load_equations(UNIFIED_JSON)
    print(f"Loaded {len(equations)} equations from {UNIFIED_JSON}")

    # ノードIDは enumerate のインデックス (0..N-1)
    edge_index = build_edges(equations)
    print(f"edge_index shape: {tuple(edge_index.shape)}")

    x = build_node_features(equations)
    print(f"x shape: {tuple(x.shape)}")

    data = Data(x=x, edge_index=edge_index)
    return data


def main() -> None:
    data = build_graph()
    torch.save(data, OUTPUT_PT)
    print(f"Saved equation graph to {OUTPUT_PT}")


if __name__ == "__main__":
    main()

