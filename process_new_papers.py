"""新規ダウンロード PDF を一括処理して unified_equations.json に統合する.

ステップ:
1. pdfs/<domain>/*.pdf を走査
2. processed_pdfs.json で重複処理を防止
3. 抽出スクリプトで各 PDF を抽出（Claude or Gemini）
4. unified_equations.json にマージ
5. validate_dataset.py で重複・健全性チェック
6. build_graph_data.py で二部グラフ再構築

使い方:
  python process_new_papers.py --engine claude       # Claude Opus（推奨、ANTHROPIC_API_KEY 必要）
  python process_new_papers.py --engine gemini       # Gemini 2.5 Pro（GEMINI_API_KEY 必要）
  python process_new_papers.py --dry-run             # 処理対象一覧のみ表示
  python process_new_papers.py --domain "反応工学（基礎）"  # 特定分野のみ
  python process_new_papers.py --model claude-opus-4-5-20250805  # モデル指定
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "pdfs"
PROCESSED_JSON = ROOT / "processed_pdfs.json"
UNIFIED_JSON = ROOT / "unified_equations.json"


def load_processed() -> set[str]:
    """既に処理済みの PDF ファイル名を取得（PDF basename ベース）"""
    if PROCESSED_JSON.exists():
        with open(PROCESSED_JSON, encoding="utf-8") as f:
            data = json.load(f)
        # 既存形式は flat list（filename のみ）、新形式は {"processed": [...]} の両方をサポート
        if isinstance(data, list):
            return set(data)
        return set(data.get("processed", []))
    return set()


def save_processed(processed: set[str]):
    """処理済みリストを更新（既存形式 = flat list を維持）"""
    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(processed), f, ensure_ascii=False, indent=2)


def find_new_pdfs(target_domain: str | None = None) -> list[Path]:
    """未処理 PDF を列挙（既存 processed_pdfs.json と互換：filename ベース）"""
    processed = load_processed()
    candidates = []
    if not PDF_DIR.exists():
        return []
    domain_dirs = [d for d in PDF_DIR.iterdir() if d.is_dir()]
    if target_domain:
        domain_dirs = [d for d in domain_dirs if d.name == target_domain]
    for ddir in sorted(domain_dirs):
        for pdf in sorted(ddir.glob("*.pdf")):
            if pdf.name not in processed:
                candidates.append(pdf)
    return candidates


def extract_one(pdf_path: Path, engine: str = "claude", model: str | None = None) -> int:
    """1 PDF を抽出。成功した数式数を返す（失敗時 -1）"""
    print(f"  Extracting: {pdf_path.name}", flush=True)
    script = "extract_equations_claude.py" if engine == "claude" else "extract_equations.py"
    cmd = [sys.executable, script, str(pdf_path)]
    if model and engine == "claude":
        cmd.extend(["--model", model])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            # フルエラーメッセージを表示（最後の 500 文字）
            err_msg = result.stderr.strip()
            if len(err_msg) > 500:
                err_msg = "..." + err_msg[-500:]
            print(f"    FAIL: {err_msg}")
            return -1
        ext_json = ROOT / "extracted_equations.json"
        if not ext_json.exists():
            return -1
        with open(ext_json, encoding="utf-8") as f:
            eqs = json.load(f)
        return len(eqs) if isinstance(eqs, list) else len(eqs.get("equations", []))
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT")
        return -1
    except Exception as e:
        print(f"    ERROR: {e}")
        return -1


def merge_to_unified() -> int:
    """extracted_equations.json を unified_equations.json にマージ"""
    ext_json = ROOT / "extracted_equations.json"
    if not ext_json.exists():
        return 0
    with open(ext_json, encoding="utf-8") as f:
        new_eqs = json.load(f)
    if not isinstance(new_eqs, list):
        new_eqs = new_eqs.get("equations", [])

    with open(UNIFIED_JSON, encoding="utf-8") as f:
        existing = json.load(f)

    existing_keys = {f"{e['source_id']}__{e['eq_id']}" for e in existing}
    added = 0
    for e in new_eqs:
        key = f"{e.get('source_id','')}__{e.get('eq_id','')}"
        if key not in existing_keys:
            existing.append(e)
            existing_keys.add(key)
            added += 1

    with open(UNIFIED_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--domain", type=str, default=None)
    ap.add_argument("--engine", choices=["claude", "gemini"], default="claude",
                    help="抽出エンジン（既定: claude）")
    ap.add_argument("--model", default=None,
                    help="Claude モデル名（例: claude-opus-4-5-20250805）")
    args = ap.parse_args()

    pdfs = find_new_pdfs(args.domain)
    print(f"処理対象 PDF: {len(pdfs)} 件")

    if args.dry_run:
        for p in pdfs:
            print(f"  {p.relative_to(ROOT)}")
        return

    if not pdfs:
        print("処理対象なし。終了。")
        return

    processed = load_processed()
    total_extracted = 0
    total_added = 0
    failed = []
    import time as _time

    # Gemini 5 RPM 対策（subprocess 間で 12 秒空ける）
    last_call_time = 0.0
    min_interval = 13.0 if args.engine == "gemini" else 1.5  # Claude は速い

    for i, pdf in enumerate(pdfs, 1):
        # レート制限：前回呼び出しから最低 N 秒空ける
        elapsed = _time.time() - last_call_time
        if last_call_time > 0 and elapsed < min_interval:
            wait = min_interval - elapsed
            print(f"  [rate limit] sleeping {wait:.1f}s...", flush=True)
            _time.sleep(wait)

        print(f"\n[{i}/{len(pdfs)}] {pdf.relative_to(ROOT)}", flush=True)
        last_call_time = _time.time()
        n = extract_one(pdf, engine=args.engine, model=args.model)
        if n < 0:
            failed.append(str(pdf.relative_to(ROOT)))
            continue
        total_extracted += n
        added = merge_to_unified()
        total_added += added
        print(f"    抽出: {n} 式、新規追加: {added} 式")
        processed.add(pdf.name)
        save_processed(processed)

    print(f"\n{'='*60}")
    print(f"処理完了: {len(pdfs) - len(failed)} / {len(pdfs)} 件成功")
    print(f"抽出式数: {total_extracted}")
    print(f"unified_equations.json への新規追加: {total_added} 式")
    print(f"失敗: {len(failed)} 件")
    if failed:
        for f in failed:
            print(f"  - {f}")
    print(f"{'='*60}")
    print(f"\n次のステップ:")
    print(f"  1. python validate_dataset.py  # 健全性チェック")
    print(f"  2. python build_graph_data.py  # 二部グラフ再構築")


if __name__ == "__main__":
    main()
