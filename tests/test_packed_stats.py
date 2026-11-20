"""Packed-value statistics (issue #32, PR 1): unpacking only, no new findings.

The Rust unit tests cover each unpacker on hand-built words. These tests check the Python
surface: `unpack_stats`, and `scan_safetensors(packed_formats=...)` attaching `packed` to a
tensor without changing which rules fire.
"""

import struct

import numpy as np
import pytest

aegis_core = pytest.importorskip("aegis.aegis_core")


def _write_safetensors(path, tensors):
    """Minimal safetensors writer: {name: (dtype_str, shape, bytes)}."""
    import json

    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def test_unpack_stats_reports_values_lanes_and_planes():
    words = np.array([0x76543210] * 4, dtype="<u4").tobytes()
    stats = aegis_core.unpack_stats("gptq", words, 4)
    assert stats.format == "gptq_int4"
    assert stats.num_values == 32
    assert len(stats.lane_entropies) == 8
    assert len(stats.plane_one_fractions) == 4


def test_unpack_stats_rejects_unknown_format_and_width():
    with pytest.raises(ValueError, match="unknown packed format"):
        aegis_core.unpack_stats("nope", b"\0" * 8, 4)
    with pytest.raises(ValueError, match="only 2, 4 and 8"):
        aegis_core.unpack_stats("gptq", b"\0" * 8, 3)


def test_scan_attaches_packed_stats_only_where_the_format_is_given(tmp_path):
    rng = np.random.default_rng(0)
    qweight = rng.integers(0, 2**32, size=4096, dtype=np.uint64).astype("<u4").tobytes()
    other = rng.standard_normal(4096).astype("<f4").tobytes()
    path = tmp_path / "m.safetensors"
    _write_safetensors(
        path,
        {"layer.qweight": ("I32", [4096], qweight), "layer.norm": ("F32", [4096], other)},
    )

    plain = {t.name: t for t in aegis_core.scan_safetensors(str(path))}
    hinted = {t.name: t for t in aegis_core.scan_safetensors(str(path), packed_formats={"layer.qweight": ("gptq", 4)})}

    assert plain["layer.qweight"].packed is None
    assert hinted["layer.qweight"].packed.format == "gptq_int4"
    assert hinted["layer.qweight"].packed.num_values == 4096 * 8
    assert hinted["layer.norm"].packed is None
    # Statistics only: the findings are identical with and without the hint.
    for name in plain:
        assert plain[name].rules == hinted[name].rules


def test_scan_rejects_an_unknown_packed_format(tmp_path):
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {"w": ("U8", [8], bytes(8))})
    with pytest.raises(ValueError, match="unknown packed format"):
        aegis_core.scan_safetensors(str(path), packed_formats={"w": ("zip", 4)})
