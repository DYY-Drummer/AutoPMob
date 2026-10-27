import os
from pathlib import Path

import kato1020_run as R


def test_run_reports_done_when_output_path_is_relative(tmp_path, monkeypatch, capsys):
    # KATO_OUT=experiments/kato1021 のように出力先を相対パスで与えても、進捗の行と終わりの行が出る
    monkeypatch.chdir(R.ROOT)
    out = Path(os.path.relpath(tmp_path, R.ROOT)) / "x.json"
    R.run([(out, ["python3", "-c", f"open({str(out)!r}, 'w').write('{{}}')"])], 1)
    s = capsys.readouterr().out
    assert "done " in s
    assert "ALL JOBS FINISHED（失敗 0）" in s
