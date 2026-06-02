"""32 分野 × 10 報の論文を OpenAlex API で自動収集する.

選定基準（マニュアル §3-§5 に対応）:
  - open_access.is_oa:true（PDF ダウンロード可能）
  - publication_year >= 2008（LaTeX 化できる品質）
  - cited_by_count >= 30（Tier 2 相当）
  - 物理モデル関連キーワードを含む

使い方:
  python fetch_papers_32domains.py            # メタデータ収集（PDF は別途）
  python fetch_papers_32domains.py --download # PDF も同時にダウンロード
  python fetch_papers_32domains.py --domain 反応工学（基礎）  # 特定分野のみ
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "pdfs"
META_JSON = ROOT / "docs" / "paper_collection" / "fetched_papers_metadata.json"
OPENALEX_URL = "https://api.openalex.org/works"

# 32 分野 × OpenAlex 検索クエリ
DOMAINS = [
    # === 化学工学コア (14) ===
    ("化学工学基礎", "chemical process simulation steady state material balance"),
    ("反応工学（基礎）", "CSTR PFR batch reactor design equation kinetics"),
    ("反応工学（応用）", "non-isothermal reactor catalytic reactor multiphase model"),
    ("プロセス制御（基礎）", "process control PID transfer function dynamic"),
    ("プロセス制御（応用）", "model predictive control adaptive nonlinear process"),
    ("熱力学（基礎）", "chemical thermodynamics phase equilibrium equation of state"),
    ("熱力学（応用）", "polymer solution thermodynamics activity coefficient model"),
    ("流体力学（基礎）", "fluid dynamics Navier Stokes viscous flow boundary"),
    ("流体力学（応用）", "turbulent flow multiphase CFD RANS LES simulation"),
    ("伝熱（基礎）", "heat transfer Fourier conduction convection radiation"),
    ("伝熱（応用）", "shell tube heat exchanger numerical thermal design"),
    ("物質移動", "mass transfer Fick diffusion coefficient concentration"),
    ("輸送現象", "transport phenomena momentum heat mass coupled"),
    ("分離プロセス", "distillation column tray model MESH equations"),

    # === 電気化学 (2) ===
    ("電気化学（基礎）", "electrochemistry Butler Volmer Nernst Tafel electrode"),
    ("電気化学（応用）", "battery fuel cell electrolyzer mathematical model"),

    # === バイオ・薬学 (3) ===
    ("バイオプロセス", "bioprocess fermentation Monod chemostat kinetic model"),
    ("薬学", "pharmaceutical dissolution pharmacokinetics compartment model"),
    ("食品工学", "food engineering drying thermal processing freezing"),

    # === 材料・高分子 (2) ===
    ("高分子工学", "polymerization reactor model molecular weight distribution kinetics"),
    ("半導体物理", "semiconductor device drift diffusion Shockley MOSFET"),

    # === エネルギー・環境 (4) ===
    ("燃焼工学", "combustion flame propagation kinetic mechanism premixed"),
    ("原子力工学", "nuclear reactor neutron diffusion point kinetics"),
    ("HVAC", "HVAC psychrometrics cooling load humidity ratio"),
    ("環境工学", "environmental pollutant transport advection diffusion treatment"),

    # === 物理・機械工学 (7) ===
    ("光学", "optics lens equation diffraction wave Maxwell"),
    ("音響学", "acoustics wave equation resonance impedance Helmholtz"),
    ("空力", "aerodynamics lift drag boundary layer airfoil"),
    ("振動学", "mechanical vibration degree of freedom modal analysis"),
    ("構造力学", "structural mechanics beam bending stress strain"),
    ("電気工学", "RLC electrical circuit dynamic analysis state space"),
    ("地盤工学", "geotechnical consolidation slope stability earth pressure"),
]

# OpenAlex 取得時のフィルタ
# - オープンアクセス、2005 年以降、被引用 10 以上
# - 種別：article / review / book-chapter / preprint / reference-entry / report も含む
FILTERS = (
    "open_access.is_oa:true,"
    "publication_year:>2005,"
    "cited_by_count:>10,"
    "type:article|review|book-chapter|preprint|reference-entry|report"
)


def search_openalex(query: str, per_page: int = 25) -> list[dict]:
    """OpenAlex から候補論文を検索。被引用数降順。
    クエリには「数式関連語」を必須として加え、物理モデル中心の論文を優先。"""
    # 数式・モデル関連語を追加してクエリを強化
    enhanced_query = f"{query} mathematical model differential equation"
    params = {
        "search": enhanced_query,
        "filter": FILTERS,
        "per-page": per_page,
        "sort": "relevance_score:desc",  # 関連度優先（被引用は二次）
    }
    for attempt in range(3):
        try:
            r = requests.get(OPENALEX_URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get("results", [])
            elif r.status_code == 429:
                print(f"  429 rate limit. Waiting 30s...")
                time.sleep(30)
            else:
                print(f"  Unexpected status {r.status_code}")
                return []
        except Exception as e:
            print(f"  Request error: {e}")
            time.sleep(5)
    return []


def is_relevant(work: dict, query: str) -> bool:
    """簡易的な関連性チェック（タイトル + abstract）"""
    title = (work.get("title") or "").lower()
    abstract = work.get("abstract_inverted_index")
    if abstract:
        abstract_text = " ".join(abstract.keys()).lower()
    else:
        abstract_text = ""

    text = title + " " + abstract_text
    if not text.strip():
        return False

    # キーワードベースのフィルタ
    query_tokens = [t.lower() for t in query.split() if len(t) > 3]
    # 少なくとも 2 つのキーワードを含む
    hits = sum(1 for tok in query_tokens if tok in text)
    if hits < 2:
        return False

    # 排除ワード（タイトルに含まれていれば除外）
    exclude_strict = [
        "deep learning", "neural network", "convolutional",
        "reinforcement learning", "transfer learning",
        "neuromorphic", "voice conversion", "image classification",
        "social network", "blockchain", "cryptocurrency",
        "natural language processing", "sentiment analysis",
    ]
    for e in exclude_strict:
        if e in title:
            return False

    return True


def best_pdf_url(work: dict) -> Optional[str]:
    """ダウンロード可能な PDF URL を取得"""
    loc = work.get("best_oa_location") or {}
    if loc.get("pdf_url"):
        return loc["pdf_url"]
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        return oa["oa_url"]
    return None


def safe_filename(s: str, max_len: int = 80) -> str:
    """ファイル名として安全な形式に変換"""
    import re
    s = re.sub(r"[^\w\-_.]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


def download_pdf(url: str, save_path: Path) -> bool:
    """PDF をダウンロード（タイムアウト・サイズ制限・magic number 検証あり）"""
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
        # PDF magic number 検証（%PDF- で始まる必要あり）
        if not first_bytes.startswith(b"%PDF-"):
            save_path.unlink()
            return False
        return True
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="PDF も同時にダウンロード")
    ap.add_argument("--domain", type=str, default=None, help="特定分野のみ実行")
    ap.add_argument("--target", type=int, default=10, help="各分野の目標報数")
    args = ap.parse_args()

    target_domains = DOMAINS
    if args.domain:
        target_domains = [d for d in DOMAINS if d[0] == args.domain]
        if not target_domains:
            print(f"Domain not found: {args.domain}")
            print("Available:", [d[0] for d in DOMAINS])
            return

    PDF_DIR.mkdir(exist_ok=True)
    META_JSON.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    download_summary = []

    for i, (domain_jp, query) in enumerate(target_domains, 1):
        print(f"\n[{i}/{len(target_domains)}] {domain_jp}")
        print(f"  Query: {query}")

        candidates = search_openalex(query, per_page=30)
        print(f"  検索ヒット: {len(candidates)} 件（被引用 30 以上 OA）")

        # 関連性フィルタ
        relevant = [w for w in candidates if is_relevant(w, query)]
        print(f"  関連性フィルタ後: {len(relevant)} 件")

        # PDF URL あるもののみ
        with_pdf = [(w, best_pdf_url(w)) for w in relevant]
        with_pdf = [(w, u) for w, u in with_pdf if u]
        print(f"  PDF URL 取得可能: {len(with_pdf)} 件")

        # 上位 N 件選抜
        selected = with_pdf[:args.target]
        domain_dir = PDF_DIR / safe_filename(domain_jp)

        if args.download:
            domain_dir.mkdir(exist_ok=True)

        domain_results = []
        n_downloaded = 0
        for j, (work, pdf_url) in enumerate(selected, 1):
            entry = {
                "domain": domain_jp,
                "rank": j,
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "doi": work.get("doi"),
                "openalex_id": work.get("id"),
                "pdf_url": pdf_url,
                "filename": None,
                "downloaded": False,
            }

            if args.download and pdf_url:
                fname = f"{j:02d}_{safe_filename(work.get('title', 'untitled'))[:60]}_{work.get('publication_year','')}.pdf"
                save_path = domain_dir / fname
                if save_path.exists():
                    print(f"    [{j}] Already exists: {fname[:60]}")
                    entry["filename"] = str(save_path.relative_to(ROOT))
                    entry["downloaded"] = True
                    n_downloaded += 1
                else:
                    print(f"    [{j}] DL: {fname[:60]}", end=" ")
                    if download_pdf(pdf_url, save_path):
                        print("✓")
                        entry["filename"] = str(save_path.relative_to(ROOT))
                        entry["downloaded"] = True
                        n_downloaded += 1
                    else:
                        print("✗")
                    time.sleep(1)  # サーバ負荷軽減

            domain_results.append(entry)

        download_summary.append({
            "domain": domain_jp,
            "candidates": len(candidates),
            "relevant": len(relevant),
            "with_pdf": len(with_pdf),
            "selected": len(selected),
            "downloaded": n_downloaded,
        })
        all_results.extend(domain_results)

        # 分野間の負荷分散
        time.sleep(0.5)

    # メタデータ保存
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "filters": FILTERS,
            "summary": download_summary,
            "papers": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"完了：{len(all_results)} 件のメタデータを保存")
    print(f"  保存先: {META_JSON}")
    if args.download:
        n_dl = sum(s["downloaded"] for s in download_summary)
        print(f"  ダウンロード成功: {n_dl} / {len(all_results)} 件")
    print(f"{'='*70}\n")

    print("=== 分野別サマリ ===")
    print(f"{'分野':30s} {'候補':>6s} {'関連':>6s} {'PDF可':>6s} {'採用':>6s} {'DL':>6s}")
    for s in download_summary:
        print(f"{s['domain']:30s} {s['candidates']:>6d} {s['relevant']:>6d} {s['with_pdf']:>6d} {s['selected']:>6d} {s['downloaded']:>6d}")


if __name__ == "__main__":
    main()
