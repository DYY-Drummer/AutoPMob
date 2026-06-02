"""3 回目の補強収集：中位分野（5-7 件）を 8-10 件に底上げ.

各分野に対し、過去のクエリと重複しない 3-5 個の専門クエリを用意。
OpenAlex に加え arXiv も補助的に使用。
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
META_JSON = ROOT / "docs" / "paper_collection" / "fetched_papers_fallback3_metadata.json"
OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "http://export.arxiv.org/api/query"

# 中位分野（5-7 件）用：専門化したクエリ
WEAK3_QUERIES = {
    "電気工学": [
        "circuit analysis state space dynamic equation",
        "power electronics inverter converter model",
        "electromagnetic induction transformer equation",
        "AC DC converter mathematical model",
    ],
    "熱力学（基礎）": [
        "Maxwell relations thermodynamic identity",
        "Carnot heat engine reversible process",
        "thermodynamic property correlation function",
        "Helmholtz free energy state function",
    ],
    "高分子工学": [
        "polymer rheology viscoelastic model",
        "free radical polymerization rate equation",
        "polymer chain growth kinetics",
        "step growth polymerization molecular weight",
    ],
    "電気化学（応用）": [
        "lithium ion battery electrochemistry model",
        "PEM fuel cell electrochemistry equation",
        "electrolyzer hydrogen production model",
        "supercapacitor electrode model",
    ],
    "空力": [
        "wind turbine aerodynamic blade element",
        "rotor helicopter aerodynamic equation",
        "hypersonic flow Mach number model",
        "airplane wing lift drag coefficient",
    ],
    "反応工学（応用）": [
        "tubular reactor design plug flow PFR",
        "fluidized bed reactor model equation",
        "trickle bed reactor multiphase model",
        "membrane reactor reaction equation",
    ],
    "プロセス制御（応用）": [
        "robust H infinity control plant model",
        "fuzzy logic control process model",
        "internal model control IMC equation",
        "Smith predictor dead time compensation",
    ],
    "プロセス制御（基礎）": [
        "stability margin Bode Nyquist analysis",
        "feedback control loop tuning equation",
        "lead lag compensator design model",
        "frequency response analysis controller",
    ],
    "バイオプロセス": [
        "cell culture bioreactor scale up model",
        "enzyme kinetics inhibition Michaelis Menten",
        "biofilm growth mathematical model",
        "metabolic flux analysis network",
    ],
    "音響学": [
        "acoustic wave propagation underwater",
        "noise control absorption coefficient model",
        "room acoustics reverberation time equation",
        "ultrasonic wave attenuation model",
        "acoustic transducer mathematical model",
    ],
    "熱力学（応用）": [
        "thermodynamic cycle Rankine equation",
        "refrigeration cycle efficiency model",
        "Stirling engine thermodynamic model",
        "Brayton cycle gas turbine equation",
        "absorption refrigeration model",
    ],
    "流体力学（応用）": [
        "non-Newtonian fluid rheology equation",
        "porous media Darcy flow model",
        "boundary layer separation flow",
        "compressible flow shock wave",
        "two phase pipe flow model",
    ],
    "光学": [
        "Fresnel coefficient reflection refraction equation",
        "Gaussian beam optics propagation",
        "Snell law refraction equation",
        "polarization wave optics equation",
        "diffraction Fraunhofer Fresnel model",
    ],
}

FILTERS = (
    "open_access.is_oa:true,"
    "publication_year:>2000,"   # 2000 年まで広げる
    "cited_by_count:>3,"        # 被引用 3 以上（さらに緩和）
    "type:article|review|book-chapter|preprint|reference-entry|report"
)


def safe_filename(s, max_len=80):
    s = re.sub(r"[^\w\-_.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


def search_openalex(query, per_page=25):
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


def search_arxiv(query, max_results=15):
    """arXiv API で検索（補助的）"""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    for attempt in range(2):
        try:
            r = requests.get(ARXIV_URL, params=params, timeout=30,
                             headers={"User-Agent": "Mozilla/5.0 AutoPMob/1.0"})
            if r.status_code == 200:
                # Atom XML をパース
                root = ET.fromstring(r.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                results = []
                for entry in root.findall("atom:entry", ns):
                    title_el = entry.find("atom:title", ns)
                    title = title_el.text.strip() if title_el is not None else ""
                    published_el = entry.find("atom:published", ns)
                    year = None
                    if published_el is not None:
                        year = int(published_el.text[:4])
                    pdf_url = None
                    for link in entry.findall("atom:link", ns):
                        if link.get("type") == "application/pdf":
                            pdf_url = link.get("href")
                            break
                    if not pdf_url:
                        id_el = entry.find("atom:id", ns)
                        if id_el is not None and "/abs/" in id_el.text:
                            pdf_url = id_el.text.replace("/abs/", "/pdf/") + ".pdf"
                    summary_el = entry.find("atom:summary", ns)
                    abstract = summary_el.text if summary_el is not None else ""
                    results.append({
                        "id": id_el.text if id_el is not None else "",
                        "title": title,
                        "publication_year": year,
                        "cited_by_count": 0,  # arXiv は被引用情報なし
                        "abstract": abstract,
                        "pdf_url": pdf_url,
                        "source": "arxiv",
                    })
                return results
            elif r.status_code == 429:
                time.sleep(30)
            else:
                return []
        except Exception as e:
            print(f"    arXiv error: {e}")
            time.sleep(5)
    return []


def is_relevant_openalex(work, query):
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


def is_relevant_arxiv(work, query):
    title = (work.get("title") or "").lower()
    abstract = (work.get("abstract") or "").lower()
    text = title + " " + abstract
    if not text.strip():
        return False
    query_tokens = [t.lower() for t in query.split() if len(t) > 3]
    if sum(1 for tok in query_tokens if tok in text) < 2:
        return False
    exclude = [
        "deep learning", "convolutional", "reinforcement learning",
        "blockchain", "sentiment", "voice conversion",
    ]
    return not any(e in title for e in exclude)


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
    META_JSON.parent.mkdir(parents=True, exist_ok=True)

    all_existing = set()
    for d in PDF_DIR.iterdir():
        if d.is_dir():
            for p in d.glob("*.pdf"):
                all_existing.add(p.stem.lower()[:60])

    all_results = []
    summary = []

    for i, (domain_jp, queries) in enumerate(WEAK3_QUERIES.items(), 1):
        domain_dir = PDF_DIR / safe_filename(domain_jp)
        domain_dir.mkdir(exist_ok=True)
        existing_in_domain = set(p.stem.lower()[:60] for p in domain_dir.glob("*.pdf"))
        existing_count = len(existing_in_domain)

        print(f"\n[{i}/{len(WEAK3_QUERIES)}] {domain_jp}（既存：{existing_count}）")

        all_candidates = {}
        for q in queries:
            print(f"  OpenAlex: {q}")
            results = search_openalex(q, per_page=20)
            for w in results:
                oa_id = w.get("id")
                if oa_id and oa_id not in all_candidates and is_relevant_openalex(w, q):
                    all_candidates[oa_id] = w
            time.sleep(0.5)

        # arXiv も補助的に試す（物理・工学系で有効）
        if domain_jp in ["電気工学", "熱力学（基礎）", "熱力学（応用）", "高分子工学",
                          "空力", "音響学", "光学", "流体力学（応用）"]:
            for q in queries[:2]:  # 最初の 2 つのクエリだけで OK
                print(f"  arXiv: {q}")
                results = search_arxiv(q, max_results=15)
                for w in results:
                    arxiv_id = w.get("id")
                    if arxiv_id and arxiv_id not in all_candidates and is_relevant_arxiv(w, q):
                        all_candidates[arxiv_id] = w
                time.sleep(1)

        with_pdf = [(w, best_pdf_url(w)) for w in all_candidates.values()]
        with_pdf = [(w, u) for w, u in with_pdf if u]
        with_pdf.sort(key=lambda x: -(x[0].get("cited_by_count", 0) or 0))
        print(f"  候補（PDF 可）: {len(with_pdf)} 件")

        target = 10
        target_new = max(0, target - existing_count)
        n_downloaded = 0
        n_attempted = 0
        for j, (work, pdf_url) in enumerate(with_pdf):
            if n_downloaded >= target_new:
                break
            n_attempted += 1
            if n_attempted > 30:  # 30 回試して失敗続きなら次の分野へ
                break
            title_key = safe_filename(work.get("title", "untitled"))[:60].lower()
            if title_key in existing_in_domain or title_key in all_existing:
                continue
            fname = f"FB3_{j+1:02d}_{safe_filename(work.get('title','untitled'))[:60]}_{work.get('publication_year','')}.pdf"
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
                    "source": work.get("source", "openalex"),
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
    print(f"3 回目補強完了：{len(all_results)} 件追加")
    print(f"{'='*70}\n")
    print(f"{'分野':25s} {'既存':>6s} {'新規':>6s} {'合計':>6s}")
    for s in summary:
        print(f"{s['domain']:25s} {s['before']:>6d} {s['new']:>6d} {s['total']:>6d}")


if __name__ == "__main__":
    main()
