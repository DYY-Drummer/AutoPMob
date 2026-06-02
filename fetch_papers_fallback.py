"""不足分野のための補強収集スクリプト.

fetch_papers_32domains.py の結果で取得数が少ない分野について、
複数の代替クエリで OpenAlex を検索し、追加 PDF を取得する。

既にダウンロード済みの PDF はスキップ（ファイル名重複検出）。

使い方:
  python fetch_papers_fallback.py
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "pdfs"
META_JSON = ROOT / "docs" / "paper_collection" / "fetched_papers_fallback_metadata.json"
OPENALEX_URL = "https://api.openalex.org/works"

FILTERS = (
    "open_access.is_oa:true,"
    "publication_year:>2005,"
    "cited_by_count:>5,"           # さらに緩める（5 以上）
    "type:article|review|book-chapter|preprint|reference-entry|report"
)

# 弱い分野ごとの代替クエリ（複数試して結果を統合）
WEAK_DOMAIN_QUERIES = {
    "HVAC": [
        "building energy simulation thermal model",
        "air conditioning load calculation cooling",
        "ventilation indoor air quality model",
        "heat pump COP energy efficiency",
    ],
    "食品工学": [
        "food drying mass transfer model",
        "food freezing thermal heat process",
        "food sterilization thermal process",
        "extrusion food processing model",
    ],
    "電気化学（基礎）": [
        "electrochemical cell electrode kinetics",
        "Tafel slope corrosion current density",
        "cyclic voltammetry electrode reaction",
        "electrochemistry mathematical model",
    ],
    "輸送現象": [
        "convective heat mass transfer model",
        "porous media flow transport model",
        "boundary layer transport phenomena",
        "momentum transport viscous",
    ],
    "熱力学（応用）": [
        "equation of state Peng Robinson cubic",
        "activity coefficient NRTL UNIQUAC",
        "vapor liquid equilibrium model",
        "phase equilibrium polymer",
    ],
    "薬学": [
        "drug dissolution model Noyes Whitney",
        "pharmacokinetic compartment model",
        "controlled drug release model",
        "drug delivery diffusion model",
    ],
    "振動学": [
        "structural vibration modal analysis",
        "damped oscillator dynamic equation",
        "rotor dynamics vibration model",
        "mechanical resonance frequency model",
    ],
    "空力": [
        "airfoil aerodynamic CFD simulation",
        "compressible flow shock wave",
        "boundary layer aerodynamic equation",
    ],
    "構造力学": [
        "beam bending Euler Bernoulli equation",
        "elastic deformation stress analysis",
        "plate theory bending mechanics",
        "finite element structural analysis",
    ],
    "反応工学（基礎）": [
        "reaction kinetics rate constant model",
        "first order reaction CSTR PFR",
        "catalytic reaction mechanism kinetic",
    ],
    "原子力工学": [
        "nuclear reactor neutronics simulation",
        "fission reactor thermal hydraulics",
        "neutron transport equation diffusion",
    ],
    "半導体物理": [
        "semiconductor device modeling MOSFET",
        "solar cell device physics equation",
        "drift diffusion transport semiconductor",
    ],
    "分離プロセス": [
        "distillation column simulation MESH",
        "absorption column mass transfer",
        "extraction liquid liquid model",
        "membrane separation transport",
    ],
    "伝熱（基礎）": [
        "conduction heat equation Fourier law",
        "convective heat transfer Nusselt",
        "radiation heat transfer view factor",
    ],
}


def safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-_.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


def search_openalex(query: str, per_page: int = 25) -> list[dict]:
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


def is_relevant(work: dict, query: str) -> bool:
    title = (work.get("title") or "").lower()
    abstract = work.get("abstract_inverted_index")
    abstract_text = " ".join(abstract.keys()).lower() if abstract else ""
    text = title + " " + abstract_text
    if not text.strip():
        return False
    query_tokens = [t.lower() for t in query.split() if len(t) > 3]
    hits = sum(1 for tok in query_tokens if tok in text)
    if hits < 2:
        return False
    exclude = [
        "deep learning", "neural network", "convolutional",
        "reinforcement learning", "transfer learning",
        "neuromorphic", "voice conversion", "image classification",
        "social network", "blockchain", "cryptocurrency",
        "natural language processing", "sentiment analysis",
    ]
    for e in exclude:
        if e in title:
            return False
    return True


def best_pdf_url(work: dict):
    loc = work.get("best_oa_location") or {}
    if loc.get("pdf_url"):
        return loc["pdf_url"]
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        return oa["oa_url"]
    return None


def download_pdf(url: str, save_path: Path) -> bool:
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

    # 既にダウンロード済みファイル名（重複検出用）
    all_existing = set()
    for d in PDF_DIR.iterdir():
        if d.is_dir():
            for p in d.glob("*.pdf"):
                all_existing.add(p.stem.lower()[:60])  # 名前のフィンガープリント

    all_results = []
    summary = []

    for i, (domain_jp, queries) in enumerate(WEAK_DOMAIN_QUERIES.items(), 1):
        print(f"\n[{i}/{len(WEAK_DOMAIN_QUERIES)}] {domain_jp}（既存：{(PDF_DIR / safe_filename(domain_jp)).glob('*.pdf') if (PDF_DIR / safe_filename(domain_jp)).exists() else 'なし'}）")
        domain_dir = PDF_DIR / safe_filename(domain_jp)
        domain_dir.mkdir(exist_ok=True)
        existing_in_domain = set(p.stem.lower()[:60] for p in domain_dir.glob("*.pdf"))

        all_candidates: dict[str, dict] = {}  # openalex_id -> work
        for q in queries:
            print(f"  Query: {q}")
            results = search_openalex(q, per_page=20)
            for w in results:
                oa_id = w.get("id")
                if oa_id and oa_id not in all_candidates:
                    if is_relevant(w, q):
                        all_candidates[oa_id] = w
            time.sleep(0.5)

        print(f"  ユニーク候補（関連性フィルタ後）: {len(all_candidates)} 件")

        # PDF URL あるもの
        with_pdf = [(w, best_pdf_url(w)) for w in all_candidates.values()]
        with_pdf = [(w, u) for w, u in with_pdf if u]
        with_pdf.sort(key=lambda x: -(x[0].get("cited_by_count", 0)))
        print(f"  PDF URL 取得可能: {len(with_pdf)} 件")

        # 既存ダウンロード分を除外＆ダウンロード
        n_downloaded = 0
        existing_count = len(existing_in_domain)
        target = 10
        target_new = max(0, target - existing_count)

        for j, (work, pdf_url) in enumerate(with_pdf):
            if n_downloaded >= target_new:
                break
            title_key = safe_filename(work.get("title", "untitled"))[:60].lower()
            if title_key in existing_in_domain or title_key in all_existing:
                continue

            fname = f"FB{j+1:02d}_{safe_filename(work.get('title','untitled'))[:60]}_{work.get('publication_year','')}.pdf"
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
            "existing_before": existing_count,
            "candidates": len(all_candidates),
            "with_pdf": len(with_pdf),
            "newly_downloaded": n_downloaded,
            "total_after": existing_count + n_downloaded,
        })

        time.sleep(0.5)

    # メタデータ保存
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "filters": FILTERS,
            "summary": summary,
            "papers": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"補強収集完了：{len(all_results)} 件の新規 PDF を追加")
    print(f"  メタデータ: {META_JSON}")
    print(f"{'='*70}\n")

    print("=== 分野別補強サマリ ===")
    print(f"{'分野':25s} {'既存':>6s} {'新規':>6s} {'合計':>6s}")
    for s in summary:
        print(f"{s['domain']:25s} {s['existing_before']:>6d} {s['newly_downloaded']:>6d} {s['total_after']:>6d}")


if __name__ == "__main__":
    main()
