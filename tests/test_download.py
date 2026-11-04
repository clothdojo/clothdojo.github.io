import json
import os
from pathlib import Path
import subprocess
import sys


def test_download_allowlists_data_and_keeps_license_files_away_from_code_root(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / "uvx"
    log = tmp_path / "calls.jsonl"
    fake.write_text(f"#!{sys.executable}\n" +
                    "import json, os, sys\n" +
                    "with open(os.environ['CALL_LOG'], 'a') as f: " +
                    "f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", CALL_LOG=str(log))
    subprocess.run(["bash", str(root / "scripts/download.sh"), "fold"], env=env, check=True)
    calls = [json.loads(row) for row in log.read_text().splitlines()]
    assert len(calls) == 3
    asset_call = calls[0]
    patterns = asset_call[asset_call.index("--include") + 1:asset_call.index("--local-dir")]
    assert patterns == ["benchmark/*"]
    assert Path(calls[1][calls[1].index("--local-dir") + 1]) == root / "licenses/assets"
    teacher_patterns = calls[2][calls[2].index("--include") + 1:calls[2].index("--local-dir")]
    assert "fold/*" in teacher_patterns and "flatten/*" not in teacher_patterns
