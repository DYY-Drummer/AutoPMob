"""
Semantic Scholar API で学術論文メタデータを一括取得し、
物理モデル構築のデータセット収集用に paper_dataset_links.json に保存する。

クエリA: "continuous stirred tank reactor dynamic mathematical modeling equations" (50件)
クエリB: "biodiesel transesterification kinetics mathematical model" (50件)
fields: title,authors,year,abstract,openAccessPdf,url

使い方:
  pip install requests
  python fetch_papers.py

429 が出る場合: 数分待って再実行するか、Semantic Scholar で API キーを取得し
  export SEMANTIC_SCHOLAR_API_KEY='your-key'
 で指定するとレート制限が緩和されます。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,abstract,openAccessPdf,url"
OUTPUT_FILE = Path(__file__).parent / "paper_dataset_links.json"

QUERIES = [
    ("continuous stirred tank reactor dynamic mathematical modeling equations", 50),
    ("biodiesel transesterification kinetics mathematical model", 50),
]


def fetch_search_results(
    query: str, limit: int = 50, offset: int = 0, api_key: str | None = None
) -> list[dict]:
    """1クエリで limit 件取得（offset でページング）。429 時はリトライ。"""
    params = {
        "query": query,
        "limit": limit,
        "offset": offset,
        "fields": FIELDS,
    }
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    for attempt in range(4):
        resp = requests.get(SEARCH_URL, params=params, headers=headers or None, timeout=30)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  429 rate limit. Waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, list):
            return data
        return []
    print("  Giving up after 429 retries.")
    return []


def normalize_paper(paper: dict, query_label: str) -> dict:
    """openAccessPdf の有無が一目でわかるように構造化する。"""
    open_access = paper.get("openAccessPdf")
    pdf_url = None
    if isinstance(open_access, dict) and open_access.get("url"):
        pdf_url = open_access["url"]
    elif isinstance(open_access, str):
        pdf_url = open_access

    return {
        "paperId": paper.get("paperId"),
        "query": query_label,
        "title": paper.get("title"),
        "authors": paper.get("authors"),
        "year": paper.get("year"),
        "abstract": paper.get("abstract"),
        "url": paper.get("url"),
        "has_open_access_pdf": pdf_url is not None,
        "open_access_pdf_url": pdf_url,
        "openAccessPdf": paper.get("openAccessPdf"),
    }


def main() -> None:
    import os
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    all_papers: list[dict] = []
    seen_ids: set[str] = set()

    for i, (query, limit) in enumerate(QUERIES):
        label = "query_A" if i == 0 else "query_B"
        print(f"Fetching up to {limit} papers for {label}: {query[:50]}...")
        try:
            papers = fetch_search_results(query, limit=limit, offset=0, api_key=api_key)
        except requests.RequestException as e:
            print(f"Error: {e}")
            continue
        for p in papers:
            pid = p.get("paperId")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(normalize_paper(p, label))
        if i < len(QUERIES) - 1:
            time.sleep(1)

    out = {
        "total_count": len(all_papers),
        "papers_with_open_access_pdf": sum(1 for p in all_papers if p.get("has_open_access_pdf")),
        "papers": all_papers,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {out['total_count']} papers to {OUTPUT_FILE}")
    print(f"  Papers with open access PDF: {out['papers_with_open_access_pdf']}")


if __name__ == "__main__":
    main()
