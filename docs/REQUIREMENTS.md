# Aegis-Tensor: Project Requirements Document

## 1. Executive Summary
**Aegis-Tensor** is an open-source, dual-engine adversarial security scanner designed to inspect AI model files (specifically `.safetensors` format) before deployment. It identifies two emerging and critical AI supply-chain attack vectors:
1. **Steganographic Malware:** Hidden malicious payloads, C2 configurations, or encrypted executables injected into floating-point weight tensors.
2. **Sleeper Agent Trojans:** Dormant backdoors embedded in model architectures that activate under specific trigger inputs, causing massive anomalous spikes in internal layer activations.

---

## 2. Problem Statement & Threat Model

### 2.1 Threat Vector A: Model Steganography
- **Mechanism:** Attackers exploit the high dimensionality and tolerance to minor noise of deep neural networks. By manipulating the least significant bits (LSB) of float32 mantissas or replacing entire uncritical parameter matrices with high-entropy encrypted blobs, adversaries embed operational malware directly into model weights.
- **Limitation of Existing Tools:** Traditional antivirus and static scanners treat model weights as opaque binary blobs. Existing `.safetensors` validators only verify JSON header syntax and type safety, but perform zero content inspection on the underlying tensor buffer.
- **Aegis-Tensor Defense:** Zero-copy memory-mapped statistical analysis using **Shannon Entropy** and **Benford's Law** to detect deviations from natural weight distributions.

### 2.2 Threat Vector B: Sleeper Agent Trojans & Backdoors
- **Mechanism:** Trojaned neural networks behave indistinguishably from clean models on standard benchmark datasets. However, when presented with a specific trigger (a watermark, phrase, or visual perturbation), dormant sub-networks fire, causing catastrophic output deviation or targeted misclassification.
- **Indicator:** Triggered activations exhibit an extreme explosion in dynamic range, measurable via intermediate layer $L_\infty$ norm spikes.
- **Aegis-Tensor Defense:** Dynamic behavioral fuzzing via PyTorch `forward_hook`s measuring layer-by-layer activation divergence relative to baseline inputs.

---

## 3. Functional Requirements (FR)

### FR-1: High-Performance Static Scanning
- **FR-1.1:** The tool MUST parse `.safetensors` files using zero-copy memory mapping (`memmap2`) without loading entire multi-gigabyte models into system RAM.
- **FR-1.2:** The tool MUST calculate byte-level Shannon Entropy for every individual tensor ($0.0 \le H \le 8.0$).
- **FR-1.3:** The tool MUST evaluate Benford's Law Mean Absolute Deviation (MAD) on the leading digits of non-zero float32 tensors.
- **FR-1.4:** The tool MUST flag tensors exceeding configurable anomaly thresholds (`--entropy-threshold`, `--benford-threshold`).
- **FR-1.5:** Multi-core parallel execution MUST be supported using Rayon for near-instant scanning of multi-billion parameter models.

### FR-2: Dynamic Trojan Fuzzing Engine
- **FR-2.1:** The tool MUST attach non-intrusive forward hooks (`register_forward_hook`) to all non-container modules of a PyTorch model.
- **FR-2.2:** The tool MUST establish a baseline activation profile using standard/clean inputs.
- **FR-2.3:** The tool MUST fuzz the model with generated adversarial, boundary, and perturbed inputs.
- **FR-2.4:** The tool MUST compute the $L_\infty$ norm ($\|x\|_\infty = \max |x_i|$) for every hooked layer per iteration.
- **FR-2.5:** The tool MUST detect activation spikes exceeding the baseline threshold ratio and isolate the specific suspicious layers.

### FR-3: Command-Line Interface & Reporting
- **FR-3.1:** Provide a unified CLI executable (`aegis`) with subcommands: `scan`, `fuzz`, `doctor`, and `version`.
- **FR-3.2:** Display rich formatted terminal tables, progress bars, and colored alert callouts.
- **FR-3.3:** Output structured machine-readable JSON reports for CI/CD security pipelines.
- **FR-3.4:** Provide an environment diagnostic command (`aegis doctor`) verifying Rust core, Python interpreter, and PyTorch status.

---

## 4. Non-Functional Requirements (NFR)

- **Performance & Scalability:** Static scanning of a 7B parameter `.safetensors` model (~14 GB) must complete within 15 seconds on modern multi-core NVMe hardware.
- **Memory Footprint:** Resident memory usage during static scanning must remain under 250 MB regardless of model file size due to zero-copy memory mapping.
- **Cross-Platform Compatibility:** Must run on Linux (x86_64, aarch64) and macOS (Apple Silicon / Intel).
- **Extensibility:** Core scanning routines must be exposed both as a standalone CLI and as an importable Python library (`import aegis`).
- **Reliability:** Graceful error handling for corrupted files, truncated headers, unsupported dtypes, and non-conforming model architectures.

---

## 5. Acceptance Criteria
1. Successfully detects synthetic steganographic injection (entropy $> 7.92$) in test models without false flagging standard weights.
2. Identifies simulated backdoor activation spikes ($\ge 4.0\times$ baseline $L_\infty$ norm) in trojaned neural networks.
3. Passes full static type checking, unit tests, and continuous integration pipeline.
