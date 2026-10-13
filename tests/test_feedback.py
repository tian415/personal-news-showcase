"""Run the browser script's dependency-free interaction regression suite."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_feedback_interactions_export_revoke_hash_and_storage_failures() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for feedback.js behavior tests")
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("test_feedback_browser.mjs"))],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
