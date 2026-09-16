"""CLI integration tests.

Every documented command-line entry point must start and parse arguments
without import errors. The legacy scripts under ``scripts/`` are exercised
through a real subprocess (they are not installed entry points); one cheap
executable run (``analysis.bootstrap``) covers the "does it actually run" case.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_ENV = {"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp")}

MODULE_CLIS = [
    "cwsr.predict.query",
    "analysis.inverse",
    "analysis.pareto",
    "analysis.bootstrap",
]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=str(REPO_ROOT),
                          env=CLI_ENV, capture_output=True, text=True, timeout=180)


@pytest.mark.parametrize("module", MODULE_CLIS)
def test_module_cli_exposes_help(module):
    result = run_cli("-m", module, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize("script", ["scripts/run_cwsr.py", "scripts/eval.py"])
def test_legacy_scripts_expose_help(script):
    if not (REPO_ROOT / script).exists():
        pytest.skip("legacy scripts are not packaged (source checkout only)")
    result = run_cli(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_invalid_arguments_are_rejected():
    unknown_option = run_cli("-m", "cwsr.predict.query", "--not-an-option")
    assert unknown_option.returncode == 2
    assert "error:" in unknown_option.stderr

    missing_required = run_cli("-m", "analysis.inverse", "--target", "1.0")
    assert missing_required.returncode == 2

    bad_choice = run_cli("-m", "cwsr.predict.query", "--results", "x.json", "Fe",
                         "--not-real")
    assert bad_choice.returncode == 2


def test_query_cli_predicts_from_a_results_file(refined_results_file):
    result = run_cli("-m", "cwsr.predict.query", "--results",
                     str(refined_results_file), "Fe0.5Ni0.5")
    assert result.returncode == 0, result.stderr
    assert "0.7500" in result.stdout


def test_analysis_bootstrap_cli_runs_end_to_end(tmp_path, linear_npz,
                                               refined_results_file):
    output = tmp_path / "bootstrap.json"
    result = run_cli("-m", "analysis.bootstrap",
                     "--dataset", str(linear_npz),
                     "--results", str(refined_results_file),
                     "--n_bootstrap", "2", "--num_trials", "1", "--jobs", "1",
                     "--seed", "0", "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert "n_successful" not in result.stdout  # summary, not raw JSON
