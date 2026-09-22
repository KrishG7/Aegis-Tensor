"""Unit and integration tests for Aegis-Tensor SARIF 2.1.0 reporting exporter."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aegis.fuzzer import TrojanScanReport, FuzzIterationReport, LayerActivationStat
from aegis.reporting import (
    SarifReportBuilder,
    export_scan_sarif,
    export_fuzz_sarif,
    RULE_STEGANOGRAPHY,
    RULE_TROJAN,
    SARIF_VERSION,
    SARIF_SCHEMA_URI,
)


def test_sarif_builder_structure():
    """Verify basic SARIF builder initialization and schema attributes."""
    builder = SarifReportBuilder()
    doc = builder.build()

    assert doc["version"] == SARIF_VERSION
    assert doc["$schema"] == SARIF_SCHEMA_URI
    assert len(doc["runs"]) == 1

    driver = doc["runs"][0]["tool"]["driver"]
    assert driver["name"] == "Aegis-Tensor"
    assert len(driver["rules"]) == 2
    rule_ids = [r["id"] for r in driver["rules"]]
    assert "aegis/steganography-detected" in rule_ids
    assert "aegis/trojan-spike-detected" in rule_ids


def test_export_scan_sarif_clean(tmp_path):
    """Verify SARIF export for models with zero suspicious tensors."""
    model_file = tmp_path / "clean_model.safetensors"
    model_file.touch()
    sarif_file = tmp_path / "clean_scan.sarif"

    # Mock clean scan result
    mock_clean_res = MagicMock()
    mock_clean_res.is_suspicious = False
    mock_clean_res.name = "encoder.layer.0.weight"

    doc = export_scan_sarif(model_file, [mock_clean_res], sarif_file)

    assert sarif_file.exists()
    assert doc["version"] == "2.1.0"
    assert len(doc["runs"][0]["results"]) == 0

    # Ensure file content on disk is valid JSON
    loaded = json.loads(sarif_file.read_text(encoding="utf-8"))
    assert loaded["version"] == "2.1.0"


def test_export_scan_sarif_anomalous_tensor(tmp_path):
    """Verify SARIF export for models with steganographic malware tensors."""
    model_file = tmp_path / "trojan_weights.safetensors"
    model_file.touch()
    sarif_file = tmp_path / "anomaly_scan.sarif"

    # Mock suspicious tensor
    mock_susp = MagicMock()
    mock_susp.is_suspicious = True
    mock_susp.name = "hidden_payload_tensor"
    mock_susp.dtype = "F32"
    mock_susp.shape = [1024, 768]
    mock_susp.entropy = 7.9845
    mock_susp.benford_mad = 0.0892
    mock_susp.anomaly_reasons = ["Shannon entropy exceeds threshold (7.98 > 7.92)", "Benford MAD anomaly"]

    doc = export_scan_sarif(model_file, [mock_susp], sarif_file)

    assert sarif_file.exists()
    results = doc["runs"][0]["results"]
    assert len(results) == 1

    res = results[0]
    assert res["ruleId"] == "aegis/steganography-detected"
    assert res["level"] == "error"
    assert "hidden_payload_tensor" in res["message"]["text"]
    assert res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == model_file.as_posix()
    assert res["properties"]["entropy"] == 7.9845
    assert res["properties"]["benfordMad"] == 0.0892


def test_export_fuzz_sarif_clean(tmp_path):
    """Verify SARIF export for benign dynamic fuzzing run."""
    model_file = tmp_path / "resnet.pt"
    model_file.touch()
    sarif_file = tmp_path / "clean_fuzz.sarif"

    report = TrojanScanReport(
        model_name="ResNet50",
        num_fuzz_samples=20,
        baseline_l_inf=1.2,
        peak_fuzzed_l_inf=2.1,
        spike_ratio=1.75,
        suspected_trojan=False,
        suspicious_layers=[],
    )

    doc = export_fuzz_sarif(model_file, report, sarif_file)
    assert sarif_file.exists()
    assert len(doc["runs"][0]["results"]) == 0


def test_export_fuzz_sarif_trojan_detected(tmp_path):
    """Verify SARIF export when Sleeper Agent Trojan activation spike is detected."""
    model_file = tmp_path / "backdoored_bert.pt"
    model_file.touch()
    sarif_file = tmp_path / "trojan_fuzz.sarif"

    report = TrojanScanReport(
        model_name="BackdooredBERT",
        num_fuzz_samples=50,
        baseline_l_inf=1.5,
        peak_fuzzed_l_inf=12.0,
        spike_ratio=8.0,
        suspected_trojan=True,
        suspicious_layers=["bert.encoder.layer.11.output"],
    )

    doc = export_fuzz_sarif(model_file, report, sarif_file)
    assert sarif_file.exists()

    results = doc["runs"][0]["results"]
    assert len(results) == 1

    res = results[0]
    assert res["ruleId"] == "aegis/trojan-spike-detected"
    assert res["level"] == "error"
    assert "BackdooredBERT" in res["message"]["text"]
    assert "8.00x" in res["message"]["text"]
    assert res["properties"]["spikeRatio"] == 8.0
    assert "bert.encoder.layer.11.output" in res["properties"]["suspiciousLayers"]
