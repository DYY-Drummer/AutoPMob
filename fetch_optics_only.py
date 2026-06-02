"""光学分野のみの追加収集（7 件 → 10 件目標）.

専門度の高い古典光学キーワードと、量子光学・非線形光学のキーワードを混ぜる。
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "pdfs"
OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "http://export.arxiv.org/api/query"

DOMAIN_JP = "光学"

# 光学専門の新規クエリ
OPTICS_QUERIES = [
    "thin lens equation imaging formula",
    "interference fringe two slit Young",
    "Maxwell electromagnetic wave equation optics",
    "geometric optics ray tracing system",
    "diffraction grating wavelength equation",
    "Stokes polarization parameter optics",
    "lens aberration design equation",
    "Gaussian beam propagation equation",
    "Fabry Perot interferometer transmission",
    "Snell law refraction index optical",
]

# 引用閾値を 0 まで緩和
FILTERS_OPTICS = (
    "open_access.is_oa:true,"
    "publication_year:>2000,"
    "type:article|review|book-chapter|preprint|reference-entry|report"
    # cited_by_count フィルタなし
)


def safe_filename(s, max_len=80):
    s = re.sub(r"[^\w\-_.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


def search_openalex(query, per_page=25):
    enhanced_query = f"{query} optics light wave"
    params = {
        "search": enhanced_query,
        "filter": FILTERS_OPTICS,
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


def search_arxiv(query, max_results=15):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    for attempt in range(2):
        try:
            r = requests.get(ARXIV_URL, params=params, timeout=30)
            if r.status_code == 200:
                root = ET.fromstring(r.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                results = []
                for entry in root.findall("atom:entry", ns):
                    title_el = entry.find("atom:title", ns)
                    title = title_el.text.strip() if title_el is not None else ""
                    published_el = entry.find("atom:published", ns)
                    year = int(published_el.text[:4]) if published_el is not None else None
                    pdf_url = None
                    for link in entry.findall("atom:link", ns):
                        if link.get("type") == "application/pdf":
                            pdf_url = link.get("href")
                            break
                    if not pdf_url:
                        id_el = entry.find("atom:id", ns)
                        if id_el is not None and "/abs/" in id_el.text:
                            pdf_url = id_el.text.replace("/abs/", "/pdf/") + ".pdf"
                    id_el = entry.find("atom:id", ns)
                    summary_el = entry.find("atom:summary", ns)
                    results.append({
                        "id": id_el.text if id_el is not None else "",
                        "title": title,
                        "publication_year": year,
                        "cited_by_count": 0,
                        "abstract": summary_el.text if summary_el is not None else "",
                        "pdf_url": pdf_url,
                        "source": "arxiv",
                    })
                return results
            elif r.status_code == 429:
                time.sleep(30)
            else:
                return []
        except Exception:
            time.sleep(5)
    return []


def is_relevant(work, query, source="openalex"):
    title = (work.get("title") or "").lower()
    if source == "arxiv":
        abstract = (work.get("abstract") or "").lower()
    else:
        abs_idx = work.get("abstract_inverted_index")
        abstract = " ".join(abs_idx.keys()).lower() if abs_idx else ""
    text = title + " " + abstract
    if not text.strip():
        return False
    # 光学関連語が含まれているか
    optics_keys = ["optic", "light", "wave", "beam", "lens", "refract",
                   "diffract", "polariz", "interfer", "photon", "laser"]
    if not any(k in text for k in optics_keys):
        return False
    exclude = [
        "deep learning", "convolutional", "reinforcement learning",
        "blockchain", "voice", "image classification", "sentiment",
        "fiber communication network",  # 通信寄り
    ]
    if any(e in title for e in exclude):
        return False
    return True


def best_pdf_url(work):
    if work.get("source") == "arxiv":
        return work.get("pdf_url")
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
    all_existing = set()
    for d in PDF_DIR.iterdir():
        if d.is_dir():
            for p in d.glob("*.pdf"):
                all_existing.add(p.stem.lower()[:60])

    domain_dir = PDF_DIR / safe_filename(DOMAIN_JP)
    domain_dir.mkdir(exist_ok=True)
    existing_in_domain = set(p.stem.lower()[:60] for p in domain_dir.glob("*.pdf"))
    existing_count = len(existing_in_domain)
    print(f"光学（既存 {existing_count} 件）")

    all_candidates = {}

    for q in OPTICS_QUERIES:
        print(f"  OpenAlex: {q}")
        results = search_openalex(q, per_page=15)
        for w in results:
            oa_id = w.get("id")
            if oa_id and oa_id not in all_candidates and is_relevant(w, q, "openalex"):
                all_candidates[oa_id] = w
        time.sleep(0.5)

    # arXiv も試す（光学は arXiv に強い分野）
    for q in OPTICS_QUERIES[:5]:
        print(f"  arXiv: {q}")
        results = search_arxiv(q, max_results=10)
        for w in results:
            aid = w.get("id")
            if aid and aid not in all_candidates and is_relevant(w, q, "arxiv"):
                all_candidates[aid] = w
        time.sleep(1)

    with_pdf = [(w, best_pdf_url(w)) for w in all_candidates.values()]
    with_pdf = [(w, u) for w, u in with_pdf if u]
    with_pdf.sort(key=lambda x: -(x[0].get("cited_by_count", 0) or 0))
    print(f"\n候補（PDF 可）: {len(with_pdf)} 件")

    target = 10
    target_new = max(0, target - existing_count)
    n_downloaded = 0
    n_attempted = 0
    new_results = []

    for j, (work, pdf_url) in enumerate(with_pdf):
        if n_downloaded >= target_new:
            break
        n_attempted += 1
        if n_attempted > 30:
            break
        title_key = safe_filename(work.get("title", "untitled"))[:60].lower()
        if title_key in existing_in_domain or title_key in all_existing:
            continue
        fname = f"FB4_{j+1:02d}_{safe_filename(work.get('title','untitled'))[:60]}_{work.get('publication_year','')}.pdf"
        save_path = domain_dir / fname
        print(f"  DL: {fname[:70]}", end=" ", flush=True)
        if download_pdf(pdf_url, save_path):
            print("✓")
            n_downloaded += 1
            existing_in_domain.add(title_key)
            all_existing.add(title_key)
            new_results.append({
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "source": work.get("source", "openalex"),
                "pdf_url": pdf_url,
                "filename": str(save_path.relative_to(ROOT)),
            })
        else:
            print("✗")
        time.sleep(1)

    print(f"\n光学：{existing_count} → {existing_count + n_downloaded}（+{n_downloaded}）")


if __name__ == "__main__":
    main()
