"""A download error in the calibration run is unresolved, never a silent skip (issue #48).

No test here touches the network: the scan functions are replaced with fakes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchmarks import calibrate  # noqa: E402


def _boom(*args, **kwargs):
    raise RuntimeError("File reconstruction error: HTTP status 403")


def _unresolved(repo="Qwen/Qwen3-0.6B", kind="safetensors"):
    return {
        "repo": repo,
        "family": "decoder",
        "kind": kind,
        "sha": "abc",
        "files": [],
        "unresolved": "RuntimeError: HTTP status 403",
        "fingerprint": "f1",
    }


def test_download_error_is_unresolved_not_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(calibrate, "scan_repo", _boom)
    record = calibrate.scan_job(None, "safetensors", "Qwen/Qwen3-0.6B", "decoder", "abc", tmp_path, 3.0, False)
    assert record["unresolved"].startswith("RuntimeError")
    assert not record.get("skipped")


def test_gguf_download_error_is_unresolved_not_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(calibrate, "scan_gguf_repo", _boom)
    record = calibrate.scan_job(None, "gguf", "org/model-GGUF", "model.gguf", "abc", tmp_path, 3.0, False)
    assert record["unresolved"].startswith("RuntimeError")
    assert record["file"] == "model.gguf"
    assert not record.get("skipped")


def test_disk_full_still_stops_the_run(monkeypatch, tmp_path):
    def full(*args, **kwargs):
        raise calibrate.DiskFull("not enough space")

    monkeypatch.setattr(calibrate, "scan_repo", full)
    try:
        calibrate.scan_job(None, "safetensors", "a/b", "decoder", "abc", tmp_path, 3.0, False)
    except calibrate.DiskFull:
        return
    raise AssertionError("DiskFull must propagate so the run stops")


def test_unresolved_is_counted_in_summary():
    numbers = calibrate.summary([_unresolved()])
    assert numbers["safetensors"]["unresolved"] == 1
    assert numbers["safetensors"]["models"] == 0


def test_unresolved_is_shown_in_the_report():
    report = calibrate.render_report([_unresolved()], lock={})
    assert "skipped: 0; unresolved: 1" in report
    assert "`Qwen/Qwen3-0.6B`: could not scan (RuntimeError: HTTP status 403)" in report


def test_no_files_at_revision_is_still_a_skip_not_unresolved():
    record = {
        "repo": "x/y",
        "family": "encoder",
        "kind": "safetensors",
        "sha": "a",
        "files": [],
        "skipped": "no .safetensors files at this revision",
    }
    assert calibrate.summary([record])["safetensors"]["unresolved"] == 0
    assert "skipped: 1; unresolved: 0" in calibrate.render_report([record], lock={})


def test_lock_errors_still_count_as_unresolved():
    lock = {"gated/model": {"sha": None, "error": "GatedRepoError: 401"}}
    assert "unresolved: 1" in calibrate.render_report([], lock=lock)


def test_unresolved_records_are_retried_even_under_current_rules():
    old_style = {**_unresolved("a/b"), "unresolved": None, "skipped": "error: RuntimeError: x"}
    assert calibrate.carry_over([_unresolved(), old_style], current="f1") == []


def test_clean_scans_and_real_skips_are_carried_over():
    scanned = {"repo": "a/b", "kind": "safetensors", "files": [{}], "fingerprint": "f1"}
    stale = {"repo": "c/d", "kind": "safetensors", "files": [{}], "fingerprint": "old"}
    skip = {"repo": "e/f", "kind": "safetensors", "files": [], "skipped": "no .safetensors files at this revision"}
    assert calibrate.carry_over([scanned, stale, skip], current="f1") == [scanned, skip]
