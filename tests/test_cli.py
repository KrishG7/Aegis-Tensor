"""Tests for Aegis-Tensor CLI."""

import subprocess
import sys


def test_cli_version():
    """Verify that `aegis --version` runs without error and outputs version."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Aegis-Tensor" in result.stdout
    assert "0.1.0" in result.stdout


def test_cli_doctor():
    """Verify that `aegis doctor` runs diagnostics successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis.cli", "doctor"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Environment Diagnostics" in result.stdout
