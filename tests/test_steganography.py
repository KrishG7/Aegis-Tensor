"""End-to-end validation for synthetic steganography fixtures."""

import pytest

from aegis import CORE_AVAILABLE, scan_safetensors
from tests.generate_stego_model import generate_models


@pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
def test_synthetic_steganography_detection(tmp_path):
    """Clean weights stay clean while both injected tensors are flagged."""
    clean_path, infected_path = generate_models(tmp_path)

    clean_results = scan_safetensors(str(clean_path), 7.92, 0.04)
    assert clean_results
    assert all(not result.is_suspicious for result in clean_results)

    infected_results = scan_safetensors(str(infected_path), 7.92, 0.04)
    suspicious = {
        result.name: result
        for result in infected_results
        if result.is_suspicious
    }
    assert {"layer_1.weight", "layer_2.weight"} <= suspicious.keys()
    assert any("entropy" in reason.lower() for reason in suspicious["layer_1.weight"].anomaly_reasons)
    assert any("benford" in reason.lower() for reason in suspicious["layer_2.weight"].anomaly_reasons)
