"""Smoke tests for offline evaluation scripts (no LLM required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_clarification_eval_runs():
    script = ROOT / "evals/offline/run_clarification_eval.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_eval_scripts_exist():
    scripts = [
        "evals/offline/run_clarification_eval.py",
        "evals/offline/run_retrieval_eval.py",
        "evals/offline/run_answer_eval.py",
        "evals/offline/run_judge_eval.py",
    ]
    for rel in scripts:
        assert (ROOT / rel).is_file()
