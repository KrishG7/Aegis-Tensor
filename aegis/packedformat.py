"""Find which tensors of a safetensors file hold packed quantized values.

Only the safetensors header and the repo's small JSON config files are read; no model is
loaded and no tensor data is touched. The answer feeds `scan_safetensors(packed_formats=...)`,
which unpacks those tensors and reports statistics (issue #32).

Formats:
  * GPTQ / AWQ: `*.qweight` int32 tensors, with the bit width from `quantize_config.json`
    (`bits`, or `w_bit` for AWQ) or from `quantization_config` in `config.json`.
  * bitsandbytes 4-bit: a `*.weight` tensor that has `*.weight.quant_state.bitsandbytes__*`
    siblings in the same file.
  * FP8 needs no hint: the Rust core recognises the dtype.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

SUPPORTED_BITS = (2, 4, 8)
_BNB_STATE = ".quant_state.bitsandbytes__"


def _header_names(path: Path) -> list[str]:
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            return []
        (length,) = struct.unpack("<Q", prefix)
        if length > 100_000_000:  # a real header is far smaller; do not read an absurd length
            return []
        header = json.loads(handle.read(length))
    return [name for name in header if name != "__metadata__"] if isinstance(header, dict) else []


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _quant_settings(directory: Path) -> tuple[str | None, object]:
    """(method, bits) from quantize_config.json, else config.json's quantization_config."""
    own = _read_json(directory / "quantize_config.json")
    if own:
        if "w_bit" in own:
            return "awq", own["w_bit"]
        return str(own.get("quant_method") or "gptq").lower(), own.get("bits")
    nested = _read_json(directory / "config.json").get("quantization_config")
    if isinstance(nested, dict):
        return str(nested.get("quant_method") or "").lower() or None, nested.get("bits", nested.get("w_bit"))
    return None, None


def detect_packed_formats(path: str | Path) -> dict[str, tuple[str, int]]:
    """{tensor name: (format, bits)} for the packed tensors of one safetensors file.

    Never raises: a missing, malformed or unsupported config simply yields no entry, so a
    scan carries on without packed statistics.
    """
    path = Path(path)
    try:
        names = _header_names(path)
    except (OSError, ValueError):
        return {}

    found: dict[str, tuple[str, int]] = {}
    method, bits = _quant_settings(path.parent)
    # bool is an int subclass; True would otherwise pass for 1-bit, which is not supported anyway.
    if method in ("gptq", "awq") and isinstance(bits, int) and not isinstance(bits, bool) and bits in SUPPORTED_BITS:
        for name in names:
            if name.endswith(".qweight"):
                found[name] = (method, bits)

    states = {name.split(_BNB_STATE)[0] for name in names if _BNB_STATE in name}
    for name in names:
        if name in states and name.endswith(".weight"):
            found[name] = ("bnb4", 4)
    return found
