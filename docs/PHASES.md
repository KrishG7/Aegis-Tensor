# Aegis-Tensor: Development Phases & Roadmap

This document outlines the four-phase development lifecycle for the Aegis-Tensor college capstone project.

---

## Timeline Overview

```
Phase 1: Foundation & Static Core     [██████████] Weeks 1-3   (Complete / Active)
Phase 2: Cryptanalysis & Fuzzing Core [████████░░] Weeks 4-6   (In Progress)
Phase 3: CLI, Reporting & CI/CD       [░░░░░░░░░░] Weeks 7-9   (Upcoming)
Phase 4: Benchmarking & Defense Valid [░░░░░░░░░░] Weeks 10-12 (Upcoming)
```

---

## Phase 1: Foundation & High-Performance Core (Weeks 1 - 3)
**Objective:** Establish cross-language build infrastructure and zero-copy memory-mapped file ingestion.

- [x] Workspace initialization (`pyproject.toml`, `Cargo.toml`, `.gitignore`, `README.md`).
- [x] Stable Rust and Python 3.9–3.14 ABI3 compilation setup via Maturin.
- [x] Zero-copy memory mapping (`memmap2`) and `.safetensors` header deserialization.
- [x] Basic Shannon Entropy ($H(X)$) byte calculation in Rust.
- [x] Rayon multi-threading integration for parallel tensor traversal.
- [x] Environment validation and health check (`aegis doctor`).

**Deliverables:**
- Working `aegis_core` native binary linked to Python.
- Baseline static scanner scanning local `.safetensors` files without RAM exhaustion.

---

## Phase 2: Statistical Cryptanalysis & Dynamic Fuzzing (Weeks 4 - 6)
**Objective:** Implement statistical detection algorithms and dynamic runtime activation monitoring.

- [x] Benford's Law Mean Absolute Deviation (MAD) algorithm on float32 weights.
- [ ] Bit-plane mantissa slice analysis for low-order bit steganography detection.
- [x] PyTorch forward-hook instrumentation pipeline (`ActivationHookManager`).
- [x] Layer-wise $L_\infty$ norm activation tracking under clean baseline inputs.
- [ ] Dynamic perturbation generator (Gaussian noise, boundary triggers, adversarial patches).
- [ ] Calibration of activation spike threshold ratio ($\tau_{\text{spike}}$).

**Deliverables:**
- Full static mathematical engine capable of distinguishing clean weights from injected payloads.
- Dynamic fuzzer identifying anomalous activation surges in PyTorch networks.

---

## Phase 3: CLI, Reporting & MLOps Integration (Weeks 7 - 9)
**Objective:** Deliver an intuitive command-line interface and machine-readable output formats for security pipelines.

- [x] Unified CLI structure (`aegis scan`, `aegis doctor`, `aegis --version`).
- [ ] Implement `aegis fuzz <model_path>` CLI command with model loader arguments.
- [x] Rich formatted terminal interface (interactive tables, colored risk badges, progress spinners).
- [x] Structured JSON report generation for downstream security logging.
- [ ] SARIF / CycloneDX SBOM export for automated GitHub Security gating.
- [ ] Automated testing pipeline (GitHub Actions for Cargo unit tests, Pytest, and wheel packaging).

**Deliverables:**
- Single-command CLI capable of outputting audit reports to terminal and JSON files.
- Automated CI pipeline testing on Linux and macOS.

---

## Phase 4: Attack Benchmarks & Academic Evaluation (Weeks 10 - 12)
**Objective:** Validate detection efficacy against real and synthetic adversarial models; prepare project defense.

- [ ] Build synthetic stego-payload generator (embed encrypted zip/executable bytes into standard weights).
- [ ] Benchmark Aegis-Tensor against clean Hugging Face models (e.g. TinyLlama, ResNet, BERT).
- [ ] Evaluate True Positive Rate (TPR), False Positive Rate (FPR), and scanning throughput (GB/sec).
- [ ] Create interactive demo scripts showcasing live malware detection and Trojan trigger isolation.
- [ ] Prepare comprehensive project report, IEEE-style paper draft, and presentation slides.

**Deliverables:**
- Benchmark suite demonstrating high recall ($>98\%$) and near-zero false positive rate.
- College capstone documentation, presentation deck, and demonstration video.
