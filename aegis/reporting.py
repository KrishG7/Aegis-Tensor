"""SARIF (Static Analysis Results Interchange Format) 2.1.0 Exporter for Aegis-Tensor.

Enables seamless ingestion of static tensor scans and dynamic Trojan fuzzing reports
into GitHub Advanced Security Code Scanning, SonarQube, and enterprise CI/CD gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aegis import __version__

SARIF_SCHEMA_URI = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
SARIF_VERSION = "2.1.0"

RULE_STEGANOGRAPHY = {
    "id": "aegis/steganography-detected",
    "name": "SteganographicPayloadDetected",
    "shortDescription": {
        "text": "Steganographic malware or encrypted payload detected in model weights."
    },
    "fullDescription": {
        "text": (
            "Tensor weights exhibit anomalous Shannon entropy (exceeding 7.92 bits/byte) "
            "or significant deviation from Benford's Law distribution (Mean Absolute Deviation > 0.04), "
            "indicating covertly embedded encrypted payloads, binary shellcode, or malicious archives."
        )
    },
    "defaultConfiguration": {
        "level": "error"
    },
    "help": {
        "text": (
            "Isolate the flagged model tensor and examine raw payload bytes.\n"
            "Verify weight provenance using cryptographic hashes or re-download weights from trusted sources."
        ),
        "markdown": (
            "### Steganography Detection Remediation\n\n"
            "1. **Isolate Tensor**: Inspect the raw bytes of the anomalous tensor.\n"
            "2. **Verify Provenance**: Compare SHA256 hashes against trusted vendor weights.\n"
            "3. **Quarantine Model**: Prevent deploying this model artifact to production inference clusters."
        )
    },
    "properties": {
        "tags": ["security", "ai-security", "steganography", "malware"],
        "precision": "very-high"
    }
}

RULE_TROJAN = {
    "id": "aegis/trojan-spike-detected",
    "name": "TrojanActivationSpikeDetected",
    "shortDescription": {
        "text": "Sleeper Agent Trojan activation spike detected during dynamic fuzzing."
    },
    "fullDescription": {
        "text": (
            "Intermediate neural network layers exhibited extreme L_infinity norm activation spikes "
            "under adversarial fuzzing inputs. This hyper-localized activation surge is characteristic "
            "of dormant trigger backdoors designed to hijack model outputs under specific input conditions."
        )
    },
    "defaultConfiguration": {
        "level": "error"
    },
    "help": {
        "text": (
            "Conduct latent activation inspection on the flagged layers.\n"
            "Retrain or fine-tune with adversarial trigger pruning (e.g. Fine-Pruning or Neural Cleanse)."
        ),
        "markdown": (
            "### Sleeper Agent Backdoor Remediation\n\n"
            "1. **Inspect Flagged Layers**: Analyze weight matrices in suspicious layers for outlier activations.\n"
            "2. **Prune / Fine-tune**: Apply activation pruning or clean data distillation.\n"
            "3. **Block Deployment**: Reject automated deployment gates until verified clean."
        )
    },
    "properties": {
        "tags": ["security", "ai-security", "backdoor", "trojan"],
        "precision": "very-high"
    }
}


class SarifReportBuilder:
    """Builder class for generating standardized SARIF 2.1.0 log structures."""

    def __init__(self, tool_name: str = "Aegis-Tensor", tool_version: str = __version__):
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.results: List[Dict[str, Any]] = []

    def register_rule(self, rule_def: Dict[str, Any]) -> None:
        """Register a SARIF rule definition if not already present."""
        rule_id = rule_def["id"]
        if rule_id not in self.rules:
            self.rules[rule_id] = rule_def

    def add_static_scan_result(
        self,
        model_path: Path,
        tensor_name: str,
        dtype: str,
        shape: List[int],
        entropy: float,
        benford_mad: float,
        reasons: List[str],
        is_critical: bool = False,
    ) -> None:
        """Add an anomaly result from static safetensors scanner."""
        self.register_rule(RULE_STEGANOGRAPHY)

        severity_level = "error" if (is_critical or entropy >= 7.92) else "warning"
        message_str = (
            f"Tensor '{tensor_name}' ({dtype}, shape {shape}) exhibits anomalous properties: "
            f"{', '.join(reasons)} (Entropy: {entropy:.4f}, Benford MAD: {benford_mad:.4f})."
        )

        result_obj: Dict[str, Any] = {
            "ruleId": RULE_STEGANOGRAPHY["id"],
            "ruleIndex": list(self.rules.keys()).index(RULE_STEGANOGRAPHY["id"]),
            "level": severity_level,
            "message": {
                "text": message_str
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": model_path.as_posix(),
                            "uriBaseId": "%SRCROOT%"
                        },
                        "region": {
                            "startLine": 1,
                            "startColumn": 1
                        }
                    }
                }
            ],
            "properties": {
                "tensorName": tensor_name,
                "dtype": dtype,
                "shape": shape,
                "entropy": entropy,
                "benfordMad": benford_mad,
                "anomalyReasons": reasons
            }
        }
        self.results.append(result_obj)

    def add_trojan_fuzz_result(
        self,
        model_path: Path,
        model_name: str,
        spike_ratio: float,
        baseline_l_inf: float,
        peak_fuzzed_l_inf: float,
        suspicious_layers: List[str],
        num_fuzz_samples: int,
    ) -> None:
        """Add an anomaly result from dynamic Trojan fuzzer."""
        self.register_rule(RULE_TROJAN)

        layers_str = ", ".join(suspicious_layers) if suspicious_layers else "Unknown"
        message_str = (
            f"Model '{model_name}' triggered Trojan backdoor activation spike. "
            f"Spike ratio: {spike_ratio:.2f}x (Peak L_inf: {peak_fuzzed_l_inf:.4f} vs Baseline: {baseline_l_inf:.4f}) "
            f"in layers: {layers_str} across {num_fuzz_samples} fuzz iterations."
        )

        result_obj: Dict[str, Any] = {
            "ruleId": RULE_TROJAN["id"],
            "ruleIndex": list(self.rules.keys()).index(RULE_TROJAN["id"]),
            "level": "error",
            "message": {
                "text": message_str
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": model_path.as_posix(),
                            "uriBaseId": "%SRCROOT%"
                        },
                        "region": {
                            "startLine": 1,
                            "startColumn": 1
                        }
                    }
                }
            ],
            "properties": {
                "modelName": model_name,
                "spikeRatio": spike_ratio,
                "baselineLInf": baseline_l_inf,
                "peakFuzzedLInf": peak_fuzzed_l_inf,
                "suspiciousLayers": suspicious_layers,
                "numFuzzSamples": num_fuzz_samples
            }
        }
        self.results.append(result_obj)

    def build(self) -> Dict[str, Any]:
        """Construct full SARIF 2.1.0 document dictionary."""
        # Ensure default rules are always registered for schema completeness
        self.register_rule(RULE_STEGANOGRAPHY)
        self.register_rule(RULE_TROJAN)

        return {
            "$schema": SARIF_SCHEMA_URI,
            "version": SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self.tool_name,
                            "version": self.tool_version,
                            "informationUri": "https://github.com/KrishG7/Aegis-Tensor",
                            "rules": list(self.rules.values())
                        }
                    },
                    "results": self.results
                }
            ]
        }

    def write_to_file(self, output_path: Union[str, Path]) -> None:
        """Write the constructed SARIF log to a JSON file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        doc = self.build()
        with open(target, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)


def export_scan_sarif(
    model_path: Path,
    results: List[Any],
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """Export static safetensors security scan results to a SARIF 2.1.0 file.
    
    Args:
        model_path: Path to the scanned safetensors model.
        results: List of TensorScanResult objects from Rust core.
        output_path: Destination path for the .sarif JSON file.
        
    Returns:
        Generated SARIF dictionary.
    """
    builder = SarifReportBuilder()
    for res in results:
        if getattr(res, "is_suspicious", False):
            builder.add_static_scan_result(
                model_path=model_path,
                tensor_name=str(res.name),
                dtype=str(res.dtype),
                shape=[int(x) for x in res.shape],
                entropy=float(res.entropy),
                benford_mad=float(res.benford_mad),
                reasons=[str(r) for r in res.anomaly_reasons],
                is_critical=float(res.entropy) >= 7.92,
            )

    builder.write_to_file(output_path)
    return builder.build()


def export_fuzz_sarif(
    model_path: Path,
    report: Any,
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """Export dynamic Trojan fuzzing report to a SARIF 2.1.0 file.
    
    Args:
        model_path: Path to the fuzzed PyTorch model.
        report: TrojanScanReport dataclass instance from DynamicTrojanFuzzer.
        output_path: Destination path for the .sarif JSON file.
        
    Returns:
        Generated SARIF dictionary.
    """
    builder = SarifReportBuilder()
    if getattr(report, "suspected_trojan", False):
        builder.add_trojan_fuzz_result(
            model_path=model_path,
            model_name=str(report.model_name),
            spike_ratio=float(report.spike_ratio),
            baseline_l_inf=float(report.baseline_l_inf),
            peak_fuzzed_l_inf=float(report.peak_fuzzed_l_inf),
            suspicious_layers=[str(l) for l in report.suspicious_layers],
            num_fuzz_samples=int(report.num_fuzz_samples),
        )

    builder.write_to_file(output_path)
    return builder.build()
