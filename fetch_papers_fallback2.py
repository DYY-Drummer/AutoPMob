"""2 回目の補強収集：まだ 5 件未満の分野に別キーワードで挑む.

対象分野（前回 4 件のみ）:
  伝熱（応用）、化学工学基礎、地盤工学、熱力学（基礎）、
  燃焼工学、物質移動、環境工学
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "pdfs"
META_JSON = ROOT / "docs" / "paper_collection" / "fetched_papers_fallback2_metadata.json"
OPENALEX_URL = "https://api.openalex.org/works"

FILTERS = (
    "open_access.is_oa:true,"
    "publication_year:>2005,"
    "cited_by_count:>5,"
    "type:article|review|book-chapter|preprint|reference-entry|report"
)

# 2 回目補強用：新規キーワード
WEAK2_QUERIES = {
    "伝熱（応用）": [
        "heat exchanger transient dynamics simulation",
        "evaporator condenser thermal performance",
        "boiling condensation heat transfer coefficient",
        "thermal energy storage heat transfer",
    ],
    "化学工学基礎": [
        "unit operation chemical engineering modeling",
        "process flow diagram material energy balance",
        "chemical process simulation Aspen",
        "process integration pinch analysis",
    ],
    "地盤工学": [
        "soil mechanics constitutive model",
        "foundation settlement analysis",
        "earthquake ground motion soil",
        "unsaturated soil hydraulic model",
    ],
    "熱力学（基礎）": [
        "Gibbs free energy chemical potential",
        "Carnot cycle efficiency thermal",
        "second law entropy thermodynamics",
        "statistical thermodynamics partition function",
    ],
    "燃焼工学": [
        "engine combustion model simulation",
        "burner flame ignition stability",
        "soot formation combustion model",
        "spray combustion droplet evaporation",
    ],
    "物質移動": [
        "absorption stripping mass transfer coefficient",
        "membrane diffusion permeation transport",
        "evaporation mass transfer model",
        "interphase mass transfer two phase",
    ],
    "環境工学": [
        "wastewater treatment activated sludge model",
        "air quality dispersion atmospheric model",
        "groundwater contamination transport model",
        "biological treatment kinetic model",
    ],
}


def safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-_.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


def search_openalex(query: str, per_page: int = 20) -> list[dict]:
    enhanced_query = f"{query} mathematical model differential equation"
    params = {
        "search": enhanced_query,
        "filter": FILTERS,
        "per-page": per_page,
        "sort": "relevance_score:desc",
    }
    for attempt in range(3):
        try:
            r = requests.get(OPENALEX_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("results", [])
            elif r.status_code == 429:
                time.sleep(30)
            else:
                return []
        except Exception:
            time.sleep(5)
    return []


def is_relevant(work, query):
    title = (work.get("title") or "").lower()
    abstract = work.get("abstract_inverted_index")
    abstract_text = " ".join(abstract.keys()).lower() if abstract else ""
    text = title + " " + abstract_text
    if not text.strip():
        return False
    query_tokens = [t.lower() for t in query.split() if len(t) > 3]
    if sum(1 for tok in query_tokens if tok in text) < 2:
        return False
    exclude = [
        "deep learning", "neural network", "convolutional",
        "reinforcement learning", "transfer learning",
        "neuromorphic", "voice conversion", "image classification",
        "blockchain", "cryptocurrency", "sentiment analysis",
    ]
    return not any(e in title for e in exclude)


def best_pdf_url(work):
    loc = work.get("best_oa_location") or {}
    if loc.get("pdf_url"):
        return loc["pdf_url"]
    oa = work.get("open_access") or {}
    return oa.get("oa_url")


def download_pdf(url, save_path):
    try:
        r = requests.get(url, timeout=60, stream=True,
                         headers={"User-Agent": "Mozilla/5.0 AutoPMob/1.0"})
        if r.status_code != 200:
            return False
        content_type = r.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not url.endswith(".pdf"):
            return False
        total = 0
        max_size = 30 * 1024 * 1024
        first_bytes = b""
        with save_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not first_bytes:
                    first_bytes = chunk[:8]
                f.write(chunk)
                total += len(chunk)
                if total > max_size:
                    f.close()
                    save_path.unlink()
                    return False
        if total < 10000:
            save_path.unlink()
            return False
        if not first_bytes.startswith(b"%PDF-"):
            save_path.unlink()
            return False
        return True
    except Exception:
        if save_path.exists():
            save_path.unlink()
        return False


def main():
    PDF_DIR.mkdir(exist_ok=True)
    META_JSON.parent.mkdir(parents=True, exist_ok=True)

    all_existing = set()
    for d in PDF_DIR.iterdir():
        if d.is_dir():
            for p in d.glob("*.pdf"):
                all_existing.add(p.stem.lower()[:60])

    all_results = []
    summary = []

    for i, (domain_jp, queries) in enumerate(WEAK2_QUERIES.items(), 1):
        domain_dir = PDF_DIR / safe_filename(domain_jp)
        domain_dir.mkdir(exist_ok=True)
        existing_in_domain = set(p.stem.lower()[:60] for p in domain_dir.glob("*.pdf"))
        existing_count = len(existing_in_domain)

        print(f"\n[{i}/{len(WEAK2_QUERIES)}] {domain_jp}（既存：{existing_count}）")

        all_candidates: dict = {}
        for q in queries:
            print(f"  Query: {q}")
            results = search_openalex(q, per_page=20)
            for w in results:
                oa_id = w.get("id")
                if oa_id and oa_id not in all_candidates and is_relevant(w, q):
                    all_candidates[oa_id] = w
            time.sleep(0.5)

        with_pdf = [(w, best_pdf_url(w)) for w in all_candidates.values()]
        with_pdf = [(w, u) for w, u in with_pdf if u]
        with_pdf.sort(key=lambda x: -(x[0].get("cited_by_count", 0)))
        print(f"  候補（PDF 可）: {len(with_pdf)} 件")

        target = 10
        target_new = max(0, target - existing_count)
        n_downloaded = 0
        for j, (work, pdf_url) in enumerate(with_pdf):
            if n_downloaded >= target_new:
                break
            title_key = safe_filename(work.get("title", "untitled"))[:60].lower()
            if title_key in existing_in_domain or title_key in all_existing:
                continue
            fname = f"FB2_{j+1:02d}_{safe_filename(work.get('title','untitled'))[:60]}_{work.get('publication_year','')}.pdf"
            save_path = domain_dir / fname
            print(f"    DL: {fname[:65]}", end=" ", flush=True)
            if download_pdf(pdf_url, save_path):
                print("✓")
                n_downloaded += 1
                existing_in_domain.add(title_key)
                all_existing.add(title_key)
                all_results.append({
                    "domain": domain_jp,
                    "title": work.get("title"),
                    "year": work.get("publication_year"),
                    "cited_by_count": work.get("cited_by_count"),
                    "doi": work.get("doi"),
                    "pdf_url": pdf_url,
                    "filename": str(save_path.relative_to(ROOT)),
                })
            else:
                print("✗")
            time.sleep(1)

        summary.append({
            "domain": domain_jp,
            "before": existing_count,
            "new": n_downloaded,
            "total": existing_count + n_downloaded,
        })

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "papers": all_results},
                  f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"2 回目補強完了：{len(all_results)} 件追加")
    print(f"{'='*70}\n")
    print(f"{'分野':25s} {'既存':>6s} {'新規':>6s} {'合計':>6s}")
    for s in summary:
        print(f"{s['domain']:25s} {s['before']:>6d} {s['new']:>6d} {s['total']:>6d}")


if __name__ == "__main__":
    main()
