# Engineering Design & Decision Rationale: Issue #8

**Subsystem:** CI/CD Automation & Reporting  
**Author:** Amandeep Singh ([@Amandeep-Bajwa](https://github.com/Amandeep-Bajwa))  
**Sprint Milestone:** Milestone 2 (Due October 6, 2026)  
**Status:** Implemented / Production Ready  

---

## 1. Executive Summary & Objective

Modern enterprise MLOps environments require automated, machine-readable security reporting to block backdoored or weaponized neural network weights prior to inference cluster deployment. 

The objective of **Issue #8** is to equip **Aegis-Tensor** with enterprise-grade interoperability by:
1. Developing a standardized **SARIF (Static Analysis Results Interchange Format) 2.1.0** export engine in `aegis/reporting.py`.
2. Seamlessly surfacing `--output-sarif` flags across both static weight scanning (`aegis scan`) and dynamic activation fuzzing (`aegis fuzz`).
3. Architecting an automated GitHub Actions CI/CD matrix across operating systems (Linux, macOS x86_64, macOS arm64) and Python versions (3.9 through 3.14) with native wheel artifact compilation via `maturin-action`.

---

## 2. Architectural Decisions & Rationale

### 2.1 Reporting Standard: SARIF 2.1.0 vs. Custom JSON vs. SonarQube Format

| Format | Ecosystem Interoperability | Native GitHub Code Scanning Support | Schema Rigor | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **SARIF 2.1.0 (OASIS Standard)** *(Selected)* | Universal (GitHub, GitLab, Azure DevOps, VS Code, SonarQube). | **Native.** Ingests directly into the GitHub Security Tab without custom parsers or third-party actions. | Strict JSON schema backed by OASIS and ISO specifications. | **Adopted.** Essential for automated CI/CD security gating. |
| **Custom Raw JSON** | Aegis-Tensor CLI only. | None (requires custom GitHub Action scripting or webhook transformation). | Ad-hoc, unstructured across tool updates. | **Retained as secondary format** via `--output-json` for quick scripting. |
| **SonarQube Generic Issue Format** | Confined strictly to SonarQube. | None. | Vendor-locked XML/JSON dialect. | **Rejected.** Severely limits deployment pipeline flexibility. |

**Rationale:**  
SARIF (OASIS standard ISO/IEC 29117) is the recognized industry standard for security static analysis results. By emitting compliant SARIF 2.1.0 logs, Aegis-Tensor scan results can be directly uploaded to GitHub via `github/codeql-action/upload-sarif`, populating the repository's **Security -> Code Scanning Alerts** tab with annotated file locations, severity indicators, and remediation advice.

---

### 2.2 Rule Ontology & Anomaly Classification

We defined two formal, semantically bounded rules within the SARIF driver:

#### 1. `aegis/steganography-detected`
- **Rule ID:** `aegis/steganography-detected`
- **Name:** `SteganographicPayloadDetected`
- **Severity Mapping:**
  - `error`: Shannon entropy $\ge 7.92\text{ bits/byte}$ (approaching theoretical maximum of 8.0, indicating encrypted shellcode or high-density compressed payloads).
  - `warning`: Benford MAD $> 0.04$ with entropy in borderline ranges (indicating non-natural distribution manipulation).
- **Precision:** `very-high`
- **Remediation Metadata:** Detailed remediation instructions including tensor isolation, cryptographic provenance verification against vendor checkpoints, and quarantine procedures.

#### 2. `aegis/trojan-spike-detected`
- **Rule ID:** `aegis/trojan-spike-detected`
- **Name:** `TrojanActivationSpikeDetected`
- **Severity Mapping:**
  - `error`: Observed $L_\infty$ activation spike multiplier exceeds specified threshold (default $4.0\times$), pointing to localized backdoor trigger circuits.
- **Precision:** `very-high`
- **Remediation Metadata:** Structural layer isolation, fine-pruning protocols, and neural cleansing instructions.

---

### 2.3 Physical Location & URI Base Mapping

GitHub Code Scanning requires valid `physicalLocation` objects to map findings to repository artifacts. For model containers (`.safetensors`, `.pt`, `.pth`), we implemented URI normalization:
- `artifactLocation.uri`: Relative or absolute path to the scanned model weight container.
- `artifactLocation.uriBaseId`: `%SRCROOT%` to enable portable path resolution across local checkouts and cloud CI runners.
- `region`: Anchored to `startLine: 1, startColumn: 1` as tensor binary containers are non-textual artifacts.
- `properties`: Custom property dictionary embedding the tensor name, datatype, tensor shape, computed Shannon entropy, Benford MAD, and trigger spike multipliers.

---

### 2.4 CI/CD Pipeline Matrix & Platform Topology

To guarantee cross-platform reliability, `.github/workflows/ci.yml` was upgraded to an automated matrix topology:

1. **Rust Core Check (`rust-check`)**:
   - **Runners:** `ubuntu-latest`, `macos-14`, `macos-latest`.
   - Validates low-level byte parsing and SIMD-accelerated zero-copy operations across architectures.
   - Executes `cargo test --no-default-features --verbose`.

2. **Python Test Matrix (`python-check`)**:
   - **Platforms:** Linux (`ubuntu-latest`) and macOS (`macos-latest`).
   - **Python Versions:** 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14-dev.
   - Validates that `maturin develop` builds clean PyO3 native extensions on each runtime and runs the full pytest test suite (21 unit and integration tests).

3. **Wheel Artifact Packaging (`wheel-package`)**:
   - Uses `PyO3/maturin-action@v1`.
   - Packages release wheels with PyO3 `abi3-py39` stable ABI compatibility.
   - Uploads compiled wheels as verified CI artifacts.

---

## 3. Verification & Validation Evidence

### 3.1 Automated Test Coverage
- **`tests/test_reporting.py`**:
  - `test_sarif_builder_structure`: Asserts presence of SARIF 2.1.0 schema, tool driver, and rule definitions.
  - `test_export_scan_sarif_clean`: Validates clean models produce 0 result entries with valid schema.
  - `test_export_scan_sarif_anomalous_tensor`: Validates suspicious tensors generate correct `aegis/steganography-detected` errors and property metadata.
  - `test_export_fuzz_sarif_clean`: Validates clean dynamic fuzzing passes with 0 alerts.
  - `test_export_fuzz_sarif_trojan_detected`: Validates backdoor spikes generate `aegis/trojan-spike-detected` error records with layer names and spike ratios.
- **`tests/test_cli.py`**:
  - `test_cli_scan_help`: Asserts `--output-sarif` option is exposed in `aegis scan --help`.
  - `test_cli_fuzz_help`: Asserts `--output-sarif` option is exposed in `aegis fuzz --help`.
  - `test_cli_scan_mocked_sarif_export`: Validates static scanner SARIF log export and schema compliance.
  - `test_cli_fuzz_mocked_execution`: End-to-end execution testing JSON export and SARIF 2.1.0 output files concurrently.

All 21 tests pass with zero failures.
