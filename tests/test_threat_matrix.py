"""Threat matrix: one test per in-scope threat. See docs/THREAT_MATRIX.md.

Each test checks that the threat is (a) detected, (b) pinpointed to a tensor,
header key, global or layer, and (c) where a treatment exists, that the
treatment removes the threat. Malicious pickles are scanned, never loaded,
except in the test that proves an unsafe load is refused.
"""

import json
import os
import pickle
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from aegis import CORE_AVAILABLE, TORCH_AVAILABLE, scan_safetensors
from aegis.cli import app
from aegis.disarm import disarm_file
from aegis.findings import from_safetensors
from aegis.formatscan import scan_safetensors_header
from aegis.picklescan import scan_pickle_file

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

needs_core = pytest.mark.skipif(not CORE_AVAILABLE, reason="aegis_core Rust extension not compiled")
needs_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
runner = CliRunner()


def _weights(seed: int = 0, n: int = 1 << 16) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.02, n).astype(np.float32)


def _write(path: Path, tensors: dict, metadata: dict | None = None) -> Path:
    save_file(tensors, str(path), metadata=metadata)
    return path


def _rewrite_header(path: Path, mutate) -> None:
    """Apply `mutate` to the parsed JSON header and rewrite the file, keeping the data bytes."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    header = mutate(header)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + n :])


def _tensors(path: Path) -> dict:
    return {r.name: r for r in scan_safetensors(str(path))}


# --- Tensor-level threats -----------------------------------------------------


@pytest.mark.claim("C06")
@needs_core
def test_t01_structured_lsb_payload_is_pinpointed_and_disarmed(tmp_path):
    payload = np.unpackbits(np.frombuffer(b"connect 10.0.0.5:4444 /bin/sh " * 200, np.uint8), bitorder="little")
    w = _weights(1, payload.size)
    words = w.view(np.uint32).copy() & ~np.uint32(1)
    words |= payload.astype(np.uint32)
    infected = _write(tmp_path / "m.safetensors", {"layer.weight": words.view(np.float32)})

    before = _tensors(infected)["layer.weight"]
    assert before.is_suspicious
    assert any("structured" in r for r in before.anomaly_reasons)

    cleaned = tmp_path / "disarmed.safetensors"
    disarm_file(infected, cleaned, low_bits=8)
    assert not _tensors(cleaned)["layer.weight"].is_suspicious


@needs_torch
@pytest.mark.claim("C09", "L01")
@needs_core
def test_t02_random_lsb_payload_is_a_documented_blind_spot_and_disarm_removes_it(tmp_path):
    rng = np.random.default_rng(2)
    w = _weights(2)
    random_bits = rng.integers(0, 16, w.size, dtype=np.uint32)  # 4-bit random payload per weight
    words = (w.view(np.uint32) & ~np.uint32(15)) | random_bits
    infected = _write(tmp_path / "m.safetensors", {"layer.weight": words.view(np.float32)})

    assert not _tensors(infected)["layer.weight"].is_suspicious  # static scan cannot see it

    cleaned = tmp_path / "disarmed.safetensors"
    disarm_file(infected, cleaned, low_bits=8)
    recovered = load_file(str(cleaned))["layer.weight"].view(torch.int32).numpy() & 15
    assert np.all(recovered == 0)  # payload gone


@pytest.mark.claim("C05")
@needs_core
def test_t03_overwritten_tensor_is_flagged_and_quarantined(tmp_path):
    blob = np.random.default_rng(3).integers(0, 256, 4 * 65536, dtype=np.uint8).view(np.float32)
    path = _write(tmp_path / "m.safetensors", {"layer.weight": blob, "ok.weight": _weights(3)})
    results = _tensors(path)
    assert results["layer.weight"].is_suspicious
    assert not results["ok.weight"].is_suspicious


@pytest.mark.claim("C05")
@needs_core
def test_t04_partial_overwrite_of_one_percent_is_flagged(tmp_path):
    rng = np.random.default_rng(4)
    arr = _weights(4, 1 << 20).view(np.uint8).reshape(-1, 4)
    idx = rng.choice(arr.shape[0], arr.shape[0] // 100, replace=False)
    arr[idx] = rng.integers(0, 256, (idx.size, 4), dtype=np.uint8)
    path = _write(tmp_path / "m.safetensors", {"layer.weight": arr.reshape(-1).view(np.float32)})
    result = _tensors(path)["layer.weight"]
    assert result.is_suspicious
    assert result.out_of_range_fraction > 0.001


@needs_core
def test_t08_clean_model_has_no_findings_in_any_channel(tmp_path):
    path = _write(
        tmp_path / "clean.safetensors", {f"l{i}.weight": _weights(i) for i in range(4)}, metadata={"format": "pt"}
    )
    assert all(not r.is_suspicious for r in scan_safetensors(str(path)))
    assert not scan_safetensors_header(path).is_suspicious


# --- Container and header channels --------------------------------------------


def test_t05_payload_split_across_small_metadata_values_is_flagged(tmp_path):
    """No single value is over the per-value limit, but together they are over the total limit."""
    path = _write(tmp_path / "m.safetensors", {"w": _weights(5)}, metadata={f"k{i}": "A" * 1000 for i in range(5)})
    header = scan_safetensors_header(path)
    assert [f.channel for f in header.findings] == ["metadata"]
    assert header.findings[0].where == "__metadata__"
    assert "totals 5000 bytes" in header.findings[0].message


@pytest.mark.claim("C07")
def test_t05_payload_in_metadata_is_flagged_and_stripped(tmp_path):
    path = _write(tmp_path / "m.safetensors", {"w": _weights(5)}, metadata={"note": "A" * 200_000})
    header = scan_safetensors_header(path)
    assert header.is_suspicious
    assert header.findings[0].channel == "metadata"
    assert "__metadata__['note']" in header.findings[0].where

    cleaned = tmp_path / "stripped.safetensors"
    disarm_file(path, cleaned, low_bits=8, strip_metadata=True)
    assert not scan_safetensors_header(cleaned).is_suspicious


@pytest.mark.claim("C07")
def test_t06_whitespace_channel_in_header_is_flagged(tmp_path):
    path = _write(tmp_path / "m.safetensors", {"w": _weights(6)})
    _rewrite_header(path, lambda h: h)  # re-serialise to the canonical form
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = raw[8 : 8 + n].rstrip(b" ")
    padded = header + b" " * 5000
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + raw[8 + n :])
    result = scan_safetensors_header(path)
    assert result.is_suspicious
    assert result.findings[0].channel == "header_whitespace"


@pytest.mark.claim("C07")
def test_t07_oversized_tensor_name_is_flagged_for_quarantine(tmp_path):
    long_name = "x" * 600
    path = _write(tmp_path / "m.safetensors", {long_name: _weights(7)})
    result = scan_safetensors_header(path)
    assert result.is_suspicious
    assert result.findings[0].treatment == "quarantine"


@needs_core
def test_t09_truncated_file_is_rejected_not_silently_scanned(tmp_path):
    path = _write(tmp_path / "m.safetensors", {"w": _weights(9)})
    path.write_bytes(path.read_bytes()[:-1000])
    with pytest.raises(OSError, match="incomplete metadata"):
        scan_safetensors(str(path))


# --- Pickle-container threats --------------------------------------------------


class _Exec:
    """Pickles as os.system(...). The payload only writes a marker file."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


@pytest.mark.claim("C02")
def test_t10_os_system_pickle_is_critical_and_pinpointed(tmp_path):
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps({"w": _Exec(tmp_path / "pwned")}))
    result = scan_pickle_file(path)
    assert result.is_critical
    assert any("system" in f.message for f in result.findings)


@pytest.mark.claim("C02")
def test_t11_builtins_eval_pickle_is_critical(tmp_path):
    path = tmp_path / "eval.pkl"
    path.write_bytes(b"\x80\x02cbuiltins\neval\nq\x00(X\x0c\x00\x00\x00print(1)q\x01tq\x02Rq\x03.")
    assert scan_pickle_file(path).is_critical


@pytest.mark.claim("C03")
def test_t12_malformed_pickle_is_reported_not_skipped(tmp_path):
    clean = pickle.dumps({"w": [1.0] * 100})
    path = tmp_path / "broken.pkl"
    path.write_bytes(clean[: len(clean) // 2])
    result = scan_pickle_file(path)
    assert result.is_suspicious


@pytest.mark.claim("C01", "C12")
@needs_torch
def test_t13_unsafe_pickle_is_never_executed_without_trust(tmp_path):
    """The payload writes a marker file. Fuzzing must refuse to load it, so no marker appears."""
    marker = tmp_path / "pwned"
    model_path = tmp_path / "evil.pt"
    model_path.write_bytes(pickle.dumps(_Exec(marker)))
    result = runner.invoke(app, ["fuzz", str(model_path), "-s", "1,10", "-n", "2"])
    assert not marker.exists()
    assert result.exit_code != 0


@needs_torch
def test_t14_torchscript_is_refused_with_explanation(tmp_path):
    model = torch.jit.script(nn.Linear(4, 2))
    path = tmp_path / "m.pt"
    torch.jit.save(model, str(path))
    result = runner.invoke(app, ["fuzz", str(path), "-s", "1,4", "-n", "2"])
    assert result.exit_code != 0
    assert "TorchScript" in result.output


# --- Behavioural threats (sleeper / trigger backdoors) -------------------------


if TORCH_AVAILABLE:

    class _ThresholdBackdoor(nn.Module):
        """Dormant trigger: a large input on feature 0 amplifies the hidden layer by 50x."""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            h = self.fc1(x)
            if (x[:, 0] > 8.0).any():
                h = h * 50.0
            return self.fc2(h)


@needs_torch
def test_t15_trigger_backdoor_is_detected_and_pinpointed_to_its_layer():
    from aegis import DynamicTrojanFuzzer

    torch.manual_seed(0)
    report = DynamicTrojanFuzzer(_ThresholdBackdoor().eval(), spike_threshold=4.0).run_fuzzing(
        input_shape=(2, 10), num_iterations=12, seed=0
    )
    assert report.suspected_trojan
    assert "fc2" in report.suspicious_layers
    assert "fc1" not in report.suspicious_layers  # the gate acts after fc1


@needs_torch
def test_t16_clean_network_is_not_flagged_at_any_input_scale():
    from aegis import DynamicTrojanFuzzer

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).eval()
    report = DynamicTrojanFuzzer(model, spike_threshold=4.0).run_fuzzing(input_shape=(2, 10), num_iterations=40, seed=0)
    assert not report.suspected_trojan
    assert report.spike_ratio < 4.0


# --- Out of scope for a weights-and-container scanner --------------------------
# Data poisoning (needs the training set), adversarial examples at inference,
# membership inference and model extraction (need query access), and LLM prompt
# injection (a runtime property of the deployed system). docs/THREAT_MATRIX.md
# records them as out of scope, so they have no tests here.


@pytest.mark.claim("C05")
@needs_core
def test_t17_nan_and_inf_injection_is_flagged(tmp_path):
    """Non-finite weights break inference (NaN propagation, overflow). Caught by the out-of-range rule."""
    w = _weights(17, 1 << 16)
    idx = np.random.default_rng(17).choice(w.size, w.size // 200, replace=False)
    w[idx[: len(idx) // 2]] = np.nan
    w[idx[len(idx) // 2 :]] = np.inf
    path = _write(tmp_path / "m.safetensors", {"layer.weight": w})
    result = _tensors(path)["layer.weight"]
    assert result.is_suspicious
    assert result.out_of_range_fraction > 0.001


@needs_core
def test_t18_scan_output_names_the_metadata_payload_and_its_treatment(tmp_path):
    path = _write(tmp_path / "m.safetensors", {"w": _weights(18)}, metadata={"note": "B" * 50_000})
    result = runner.invoke(app, ["scan", str(path)])
    assert "__metadata__['note']" in result.output
    assert "--strip-metadata" in result.output
    assert "Recommended Treatment" in result.output


@needs_core
@pytest.mark.claim("C22")
def test_t19_payload_split_into_tiny_tensors_is_flagged(tmp_path):
    rng = np.random.default_rng(19)
    tiny = {f"chunk.{i}": rng.integers(0, 256, 4 * 64, dtype=np.uint8).view(np.float32) for i in range(64)}
    path = _write(tmp_path / "m.safetensors", tiny)
    assert any(result.is_suspicious for result in scan_safetensors(str(path)))


@needs_core
@pytest.mark.claim("L03")
@pytest.mark.xfail(strict=True, reason="known gap L03: small GGUF tensors are not pooled (H1 pools safetensors only)")
def test_t19_payload_split_into_tiny_gguf_tensors_is_flagged(tmp_path):
    from tests.test_model_formats import gguf_bytes

    rng = np.random.default_rng(19)
    tiny = {f"chunk.{i}": rng.integers(0, 256, 4 * 64, dtype=np.uint8).view(np.float32) for i in range(64)}
    path = tmp_path / "m.gguf"
    path.write_bytes(gguf_bytes([("general.alignment", 32)], [(n, [t.size], 0, t.tobytes()) for n, t in tiny.items()]))

    result = CliRunner().invoke(app, ["scan", str(path), "--format", "json"])
    assert result.exit_code == 1, result.output


@needs_core
@pytest.mark.claim("C22")
def test_pooled_finding_names_every_contributing_tensor(tmp_path):
    rng = np.random.default_rng(19)
    tiny = {f"chunk.{i}": rng.integers(0, 256, 4 * 64, dtype=np.uint8).view(np.float32) for i in range(64)}
    path = _write(tmp_path / "m.safetensors", tiny)

    results = scan_safetensors(str(path))
    findings = from_safetensors(str(path), results)
    pooled_finding = next(f for f in findings if f.location.get("tensor", "").startswith("pooled small "))

    assert pooled_finding.evidence["pooled_from"] == sorted(tiny)


def test_pooled_result_provenance_is_in_finding_evidence():
    contributors = ["chunk.0", "chunk.1"]
    result = SimpleNamespace(
        rules=["exponent_lane"],
        anomaly_reasons=["random-looking pooled exponent lane"],
        name="pooled small F32 tensors (2)",
        dtype="F32",
        num_elements=256,
        entropy=7.0,
        exponent_lane_entropy=7.0,
        low_plane_entropy=None,
        out_of_range_fraction=None,
        benford_mad=0.0,
        pooled_from=contributors,
    )

    finding = from_safetensors("model.safetensors", [result])[0]

    assert finding.evidence["pooled_from"] == contributors


@needs_core
def test_small_clean_tensors_pooled_are_not_flagged(tmp_path):
    rng = np.random.default_rng(0)
    tensors = {f"weight.{i:02}": rng.normal(0, 0.02, 64).astype(np.float32) for i in range(64)}
    tensors.update({f"layer_norm.{i}": rng.normal(1.0, 0.02, 64).astype(np.float32) for i in range(3)})
    path = _write(tmp_path / "clean.safetensors", tensors)

    assert not any(result.is_suspicious for result in scan_safetensors(str(path)))


@needs_core
@pytest.mark.claim("C22")
def test_pools_are_per_dtype(tmp_path):
    rng = np.random.default_rng(38)
    tensors = {
        **{f"f16.{i:02}": rng.integers(0, 256, 2 * 64, dtype=np.uint8).view(np.float16) for i in range(40)},
        **{f"f32.{i:02}": rng.integers(0, 256, 4 * 64, dtype=np.uint8).view(np.float32) for i in range(40)},
    }
    path = _write(tmp_path / "mixed.safetensors", tensors)

    pooled = [result for result in scan_safetensors(str(path)) if result.name.startswith("pooled small ")]

    assert len(pooled) == 2
    assert {result.dtype for result in pooled} == {"F16", "F32"}
    assert all(len(result.pooled_from) == 40 for result in pooled)


# --- Packed quantized tensors (issue #32, PR 2): T37 to T40, gap T41 ---------------------------
# A payload replaces the packed values of a GPTQ/AWQ, bitsandbytes 4-bit or FP8 tensor. The rule
# measures the entropy of the unpacked values; thresholds come from docs/calibration.md (P3).


def _write_raw(path: Path, tensors: dict) -> Path:
    """safetensors writer for dtypes numpy cannot save (F8_E4M3): {name: (dtype, shape, bytes)}."""
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header, separators=(",", ":")).encode()  # compact: no whitespace to flag
    head += b" " * ((-len(head)) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))
    return path


def _levels(rng, n: int, centre: float, sigma: float, top: int) -> np.ndarray:
    """Quantized weight levels 0..top: a bell curve, like a real quantized layer."""
    return np.clip(np.round(rng.normal(centre, sigma, n)), 0, top).astype(np.uint32)


def _pack_words(values: np.ndarray, bits: int) -> np.ndarray:
    per_word = 32 // bits
    v = values.reshape(-1, per_word)
    return sum((v[:, lane] << (bits * lane) for lane in range(per_word)), np.zeros(len(v), np.uint32)).astype("<u4")


def _to_e4m3(x: np.ndarray) -> np.ndarray:
    """Round float values to FP8 E4M3 bytes (no ml_dtypes needed)."""
    x = np.clip(x, -448, 448)
    a = np.abs(x)
    e = np.clip(np.floor(np.log2(np.where(a > 0, a, 1))), -6, 8)
    code = np.where(a >= 2.0**-6, (e + 7) * 8 + np.round((a / 2.0**e - 1) * 8), np.round(a * 512))
    return np.clip(code, 0, 126).astype(np.uint8) | ((x < 0).astype(np.uint8) << 7)


N_VALUES = 1 << 17  # 131,072 values per tensor, above the 100,000 the calibration covered


def _packed_model(tmp_path: Path, fmt: str, payload: float = 0.0, seed: int = 37) -> Path:
    """A one-layer model in `fmt`. `payload` is the fraction of its packed bytes replaced with random bytes."""
    rng = np.random.default_rng(seed)
    path = tmp_path / "m.safetensors"
    if fmt in ("gptq_int4", "gptq_int8"):
        bits = 4 if fmt == "gptq_int4" else 8
        values = _levels(rng, N_VALUES, (2**bits - 1) / 2, 2.2 if bits == 4 else 40, 2**bits - 1)
        raw = bytearray(_pack_words(values, bits).tobytes())
        (tmp_path / "quantize_config.json").write_text(json.dumps({"bits": bits}))
        name, dtype = "layer.qweight", "I32"
    elif fmt == "bnb_4bit":
        v = _levels(rng, N_VALUES, 7.5, 2.5, 15).reshape(-1, 2)
        raw = bytearray(((v[:, 0] << 4) | v[:, 1]).astype(np.uint8).tobytes())
        name, dtype = "layer.weight", "U8"
    else:  # fp8_e4m3: weights scaled so the largest is near 448, as FP8 checkpoints are
        raw = bytearray(_to_e4m3(rng.normal(0, 100, N_VALUES)).tobytes())
        name, dtype = "layer.weight", "F8_E4M3"
    if payload:
        n = int(len(raw) * payload)
        raw[:n] = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    tensors = {name: (dtype, [len(raw) // {"I32": 4}.get(dtype, 1)], bytes(raw))}
    if fmt == "bnb_4bit":
        tensors["layer.weight.quant_state.bitsandbytes__nf4"] = ("U8", [8], bytes(8))
    return _write_raw(path, tensors)


def _packed_findings(path: Path) -> tuple[list[str], int]:
    """(packed-rule ids found, exit code) of an `aegis scan` of `path`."""
    result = runner.invoke(app, ["scan", str(path), "--format", "json"])
    found = [f["rule"] for f in json.loads(result.output)["files"][0]["findings"] if f["rule"] == "AEGIS-T007"]
    return found, result.exit_code


PACKED_FORMATS = ["gptq_int4", "gptq_int8", "bnb_4bit", "fp8_e4m3"]


@needs_core
@pytest.mark.claim("C31")
@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_payload_in_packed_tensor_is_flagged(tmp_path, fmt):
    """T37-T40: random bytes replacing a whole packed tensor are flagged by the packed rule."""
    rules, exit_code = _packed_findings(_packed_model(tmp_path, fmt, payload=1.0))
    assert rules == ["AEGIS-T007"]
    assert exit_code == 1


@needs_core
@pytest.mark.claim("C31")
@pytest.mark.parametrize("fmt", PACKED_FORMATS)
def test_clean_quantized_tensor_is_not_flagged(tmp_path, fmt):
    """The same layer with realistic values raises no finding."""
    rules, exit_code = _packed_findings(_packed_model(tmp_path, fmt))
    assert rules == []
    assert exit_code == 0


@needs_core
@pytest.mark.claim("C31")
def test_packed_finding_names_the_tensor_and_its_measured_entropy(tmp_path):
    path = _packed_model(tmp_path, "gptq_int4", payload=1.0)
    result = runner.invoke(app, ["scan", str(path), "--format", "json"])
    finding = next(f for f in json.loads(result.output)["files"][0]["findings"] if f["rule"] == "AEGIS-T007")
    assert finding["location"]["tensor"] == "layer.qweight"
    assert finding["evidence"]["packed_format"] == "gptq_int4"
    assert finding["evidence"]["packed_value_entropy"] > 3.97


@needs_core
@pytest.mark.claim("C31")
def test_without_the_repo_config_a_gptq_tensor_is_not_assumed_packed(tmp_path):
    path = _packed_model(tmp_path, "gptq_int4", payload=1.0)
    (tmp_path / "quantize_config.json").unlink()
    assert _packed_findings(path)[0] == []


@needs_core
@pytest.mark.claim("L04")
@pytest.mark.xfail(strict=True, reason="known gap L04: a small payload in a packed tensor stays under the threshold")
def test_t41_small_payload_in_packed_tensor_is_flagged(tmp_path):
    rules, _ = _packed_findings(_packed_model(tmp_path, "gptq_int4", payload=0.10))
    assert rules == ["AEGIS-T007"]
