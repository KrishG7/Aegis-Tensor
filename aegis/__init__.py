"""Aegis-Tensor: Open-source security tool for AI models.

Detects Steganography (hidden malware) and Sleeper Agent Trojans (backdoors)
inside AI models like .safetensors using static Rust analysis and dynamic PyTorch fuzzing.
"""

__version__ = "0.1.0"

# Import compiled Rust core extension if available
try:
    from aegis.aegis_core import (  # type: ignore
        TensorScanResult,
        benford_law_mad,
        mantissa_bit_plane_entropy,
        scan_safetensors,
        shannon_entropy_chunked,
        shannon_entropy_bytes,
    )

    CORE_AVAILABLE = True
except ImportError:
    # Graceful fallback when aegis_core hasn't been compiled yet with maturin
    CORE_AVAILABLE = False
    TensorScanResult = None  # type: ignore
    scan_safetensors = None  # type: ignore
    shannon_entropy_chunked = None  # type: ignore
    shannon_entropy_bytes = None  # type: ignore
    mantissa_bit_plane_entropy = None  # type: ignore
    benford_law_mad = None  # type: ignore

# Dynamic Fuzzer imports
from aegis.fuzzer import (
    DynamicTrojanFuzzer,
    ActivationHookManager,
    TrojanScanReport,
    LayerActivationStat,
    TORCH_AVAILABLE,
)

# SARIF Reporting imports
from aegis.reporting import (
    SarifReportBuilder,
    export_scan_sarif,
    export_fuzz_sarif,
    SARIF_VERSION,
    SARIF_SCHEMA_URI,
)

# Hugging Face Hub Inspector imports
from aegis.hf_inspector import (
    HuggingFaceInspectorError,
    parse_hf_uri,
    inspect_hf_header,
    download_hf_model,
    resolve_hf_model,
)

__all__ = [
    "__version__",
    "CORE_AVAILABLE",
    "TORCH_AVAILABLE",
    "TensorScanResult",
    "scan_safetensors",
    "shannon_entropy_bytes",
    "shannon_entropy_chunked",
    "mantissa_bit_plane_entropy",
    "benford_law_mad",
    "DynamicTrojanFuzzer",
    "ActivationHookManager",
    "TrojanScanReport",
    "LayerActivationStat",
    "SarifReportBuilder",
    "export_scan_sarif",
    "export_fuzz_sarif",
    "SARIF_VERSION",
    "SARIF_SCHEMA_URI",
    "HuggingFaceInspectorError",
    "parse_hf_uri",
    "inspect_hf_header",
    "download_hf_model",
    "resolve_hf_model",
]
