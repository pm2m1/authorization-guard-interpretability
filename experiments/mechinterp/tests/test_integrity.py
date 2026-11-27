"""Runs the inference-free integrity check (scripts/verify_integrity.py) on the preserved raw results."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def test_preserved_raw_results_integrity():
    proc = subprocess.run([sys.executable, "-I", str(HERE / "scripts/verify_integrity.py")],
                          capture_output=True, text=True, cwd=HERE, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '"status": "PASS"' in proc.stdout
