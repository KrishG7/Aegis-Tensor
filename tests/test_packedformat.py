"""Find which tensors of a safetensors file are packed, from the repo's own config (issue #32, PR 2).

Nothing here loads a model: only the safetensors header and small JSON files are read.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from aegis.packedformat import detect_packed_formats  # noqa: E402


def _write(path: Path, names: dict[str, str]) -> Path:
    """A real safetensors file with tiny tensors: {name: dtype}."""
    sizes = {"I32": 4, "U8": 1, "F16": 2, "F32": 4}
    header, offset, blobs = {}, 0, []
    for name, dtype in names.items():
        raw = bytes(range(8)) * sizes[dtype]
        header[name] = {"dtype": dtype, "shape": [8], "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))
    return path


GPTQ_NAMES = {
    "model.layers.0.mlp.up_proj.qweight": "I32",
    "model.layers.0.mlp.up_proj.qzeros": "I32",
    "model.layers.0.mlp.up_proj.scales": "F16",
    "model.norm.weight": "F16",
}


def test_gptq_qweight_is_found_from_quantize_config(tmp_path):
    (tmp_path / "quantize_config.json").write_text(json.dumps({"bits": 4, "group_size": 128}))
    found = detect_packed_formats(_write(tmp_path / "m.safetensors", GPTQ_NAMES))
    assert found == {"model.layers.0.mlp.up_proj.qweight": ("gptq", 4)}


def test_awq_is_recognised_by_its_w_bit_key(tmp_path):
    (tmp_path / "quantize_config.json").write_text(json.dumps({"w_bit": 4, "q_group_size": 128, "zero_point": True}))
    found = detect_packed_formats(_write(tmp_path / "m.safetensors", GPTQ_NAMES))
    assert found == {"model.layers.0.mlp.up_proj.qweight": ("awq", 4)}


def test_config_json_quantization_config_is_used_too(tmp_path):
    cfg = {"quantization_config": {"quant_method": "gptq", "bits": 8}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    found = detect_packed_formats(_write(tmp_path / "m.safetensors", GPTQ_NAMES))
    assert found == {"model.layers.0.mlp.up_proj.qweight": ("gptq", 8)}


def test_bitsandbytes_4bit_weight_is_found_by_its_quant_state_siblings(tmp_path):
    names = {
        "layers.0.q_proj.weight": "U8",
        "layers.0.q_proj.weight.absmax": "F32",
        "layers.0.q_proj.weight.quant_map": "F32",
        "layers.0.q_proj.weight.quant_state.bitsandbytes__nf4": "U8",
        "layers.0.norm.weight": "F16",
    }
    found = detect_packed_formats(_write(tmp_path / "m.safetensors", names))
    assert found == {"layers.0.q_proj.weight": ("bnb4", 4)}


def test_a_plain_float_model_has_no_packed_tensors(tmp_path):
    found = detect_packed_formats(_write(tmp_path / "m.safetensors", {"w.weight": "F16", "b.bias": "F16"}))
    assert found == {}


@pytest.mark.parametrize("bits", [3, 5, 16, 0, "4", None])
def test_unsupported_or_malformed_bit_widths_are_not_guessed(tmp_path, bits):
    (tmp_path / "quantize_config.json").write_text(json.dumps({"bits": bits}))
    assert detect_packed_formats(_write(tmp_path / "m.safetensors", GPTQ_NAMES)) == {}


def test_a_broken_config_or_header_never_raises(tmp_path):
    (tmp_path / "quantize_config.json").write_text("{not json")
    (tmp_path / "config.json").write_text("[1, 2]")
    assert detect_packed_formats(_write(tmp_path / "m.safetensors", GPTQ_NAMES)) == {}
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\xff" * 5)
    assert detect_packed_formats(bad) == {}
    assert detect_packed_formats(tmp_path / "missing.safetensors") == {}


def test_without_any_config_qweight_is_not_assumed_to_be_gptq(tmp_path):
    assert detect_packed_formats(_write(tmp_path / "m.safetensors", GPTQ_NAMES)) == {}


def test_calibration_records_the_packed_statistics(tmp_path):
    pytest.importorskip("aegis.aegis_core")
    from benchmarks import calibrate

    rng = np.random.default_rng(1)
    (tmp_path / "quantize_config.json").write_text(json.dumps({"bits": 4}))
    path = tmp_path / "m.safetensors"
    names = {"a.qweight": "I32", "norm.weight": "F32"}
    header, blobs, offset = {}, [], 0
    for name, dtype in names.items():
        raw = rng.integers(0, 256, size=2048, dtype=np.uint8).tobytes()
        raw = raw[: 4 * 512] if dtype in ("I32", "F32") else raw
        header[name] = {"dtype": dtype, "shape": [512], "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))

    records = {t["name"]: t for t in calibrate.scan_file(path)["tensors"]}
    packed = records["a.qweight"]["packed"]
    assert packed["format"] == "gptq_int4"
    assert packed["num_values"] == 512 * 8
    assert len(packed["lane_entropies"]) == 8
    assert "packed" not in records["norm.weight"]
