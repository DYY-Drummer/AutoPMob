"""論文用に作り直す図を paper/cace/figures に書き出す（修論の図は変えない）.

- fig_dataset：拡張ケースの棒を除いたデータの構成
- fig_split_balance：分割のそろい方（種 42）
- fig_method_comparison、fig_characteristic：凡例に手法の名称（make_numbers.METHOD_NAME、原稿のマクロ \\ours）を出す

そのほかの図は修論の図と同じファイルで、check_manuscript.py が同じかどうかを調べる。
使い方: python3 paper/cace/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import make_numbers as mn  # noqa: E402

METHOD_NAME = mn.METHOD_NAME
FIGURES = HERE / "figures"


def main() -> None:
    import generate_figures_thesis as gft
    gft.dataset(augmented=False, out_dir=FIGURES)
    gft.split_balance(seed=42, out_dir=FIGURES, paper=True)
    gft.method_comparison(method_name=METHOD_NAME, out_dir=FIGURES)
    gft.characteristic(method_name=METHOD_NAME, out_dir=FIGURES)


if __name__ == "__main__":
    main()
