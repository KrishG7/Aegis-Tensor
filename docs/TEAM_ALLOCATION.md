# Aegis-Tensor: Team Work Allocation & Responsibilities

## Team Roster

| Member | Role | GitHub Username | Status |
| :--- | :--- | :--- | :--- |
| **Krish Gupta** | Lead System Architect & Core Infrastructure | [`KrishG7`](https://github.com/KrishG7) | Active Contributor (Lead) |
| **Himanshu Chhillar** | Static Detection & Cryptanalysis Engineer | [`HimanshuChhillar`](https://github.com/HimanshuChhillar) | Active Contributor |
| **Daksh Dahiya** | Dynamic Fuzzing & Activation Hook Engineer | [`7dxksh7`](https://github.com/7dxksh7) | Active Contributor |
| **Amandeep Singh** | CLI, CI/CD & Evaluation Engineer | [`Amandeep-bajwa`](https://github.com/Amandeep-bajwa) | Active Contributor |
| **Mukul Chauhan** | Project Presentation & Documentation Support | [`Mukul09800`](https://github.com/Mukul09800) | Passive / Advisory |

---

## 1. Detailed Member Work Breakdown

### 1. Krish Gupta (`KrishG7`) — Lead Architect & Core Infrastructure
**Domain:** Cross-language bridge, low-level memory mapping, and architectural governance.

- **Tasks & Responsibilities:**
  1. **Core Systems Engineering:**
     - Implement and maintain zero-copy memory mapping (`memmap2`) inside `src/lib.rs`.
     - Direct byte slice extraction and header parsing from `.safetensors` format.
     - Multi-threaded Rayon work-stealing pool orchestration for parallel tensor scanning.
  2. **PyO3 / Maturin Foreign Function Interface (FFI):**
     - Maintain ABI3 forward-compatible C-extension configuration (`abi3-py39`).
     - Expose native Rust data structures (`TensorScanResult`) safely to Python runtimes.
     - Build orchestration and packaging configuration in `pyproject.toml` and `Cargo.toml`.
  3. **Repository Management & Review:**
     - Architectural oversight, PR reviews, code standards, and version tagging.
     - Lead integration of Rust static core with Python dynamic components.

---

### 2. Himanshu Chhillar (`HimanshuChhillar`) — Static Detection & Statistical Cryptanalysis
**Domain:** Mathematical algorithms, entropy measurement, and steganography detection heuristics.

- **Tasks & Responsibilities:**
  1. **Shannon Entropy Engine:**
     - Optimize byte-level entropy calculation algorithm ($H(X) = -\sum P(x) \log_2 P(x)$).
     - Implement sliding window and chunk-level entropy analysis to detect payloads concentrated in specific weight sub-matrices.
     - Research and implement mantissa/LSB bit-plane entropy checks to catch subtle LSB steganography.
  2. **Benford's Law Anomaly Detection:**
     - Refine leading digit extraction across float32 / float16 / bfloat16 tensors.
     - Implement Chi-squared ($\chi^2$) Goodness-of-Fit and Mean Absolute Deviation (MAD) metrics.
     - Calibrate statistical anomaly thresholds against benign pre-trained weights (ResNet, LLaMA, BERT).
  3. **Steganography Benchmark Generation:**
     - Write synthetic test scripts that inject encrypted and compressed blobs into dummy `.safetensors` files to validate detection accuracy.

---

### 3. Daksh Dahiya (`7dxksh7`) — Dynamic Fuzzing & Activation Engine
**Domain:** PyTorch model introspection, runtime hook pipeline, and Trojan backdoor discovery.

- **Tasks & Responsibilities:**
  1. **PyTorch Forward Hook Architecture:**
     - Develop and enhance `aegis/fuzzer.py` and `ActivationHookManager`.
     - Implement robust hook attachment across diverse neural network architectures (CNNs, Transformers, MLPs).
     - Extract clean intermediate activations without modifying model weights or creating memory leaks.
  2. **$L_\infty$ Norm Spike Analytics:**
     - Compute layer-wise $L_\infty$ norms ($\max |x_i|$) across forward passes.
     - Calculate spike ratios relative to baseline clean inputs and identify anomalous neuron activations.
     - Implement sub-network isolation to pinpoint the exact layer(s) containing dormant Trojan triggers.
  3. **Input Perturbation & Trigger Fuzzing:**
     - Design input generation strategies: boundary values, Gaussian noise, frequency-domain perturbations, and watermark masks.
     - Benchmark fuzzing efficiency and optimize batch execution on GPU (CUDA/MPS) and CPU.

---

### 4. Amandeep Singh (`Amandeep-bajwa`) — CLI, CI/CD & Benchmark Suite
**Domain:** User interface, machine-readable reporting, automated testing, and evaluation metrics.

- **Tasks & Responsibilities:**
  1. **Command Line Interface & Terminal UX:**
     - Build interactive Rich terminal displays in `aegis/cli.py` (live tables, colored status badges, scan spinners).
     - Add dedicated `aegis fuzz` CLI command linking into the dynamic fuzzer engine.
     - Implement structured output options (`--output-json`, SARIF format for GitHub Code Scanning).
  2. **Testing & Continuous Integration:**
     - Write automated unit tests using `pytest` for Python CLI and `cargo test` for Rust core.
     - Set up GitHub Actions CI workflow to build wheels, run linters, and verify cross-platform compatibility.
  3. **Evaluation & Performance Benchmarking:**
     - Benchmark throughput (tensors/sec and GB/sec) across varied model sizes (100M to 7B parameters).
     - Generate evaluation charts and metrics (True Positive Rate, False Positive Rate, latency).
     - Write model pre-download integration utility (vetting Hugging Face repository files before download).

---

### 5. Mukul Chauhan (`Mukul09800`) — Presentation & Academic Documentation (Passive)
**Domain:** Academic deliverables, presentation design, and demonstration assets.

- **Tasks & Responsibilities:**
  - Assist with formatting college project report and final thesis document.
  - Compile slide deck for project presentations and capstone defense.
  - Coordinate demonstration media and walkthrough guides.

---

## 2. Responsibility Matrix (RACI)

| Milestone / Deliverable | Krish (`KrishG7`) | Himanshu (`HimanshuChhillar`) | Daksh (`7dxksh7`) | Amandeep (`Amandeep-bajwa`) | Mukul (`Mukul09800`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rust Memory-Mapped Core** | **Accountable / Responsible** | Consulted | Informed | Informed | Informed |
| **Shannon Entropy & Benford's Law** | Consulted | **Accountable / Responsible** | Informed | Informed | Informed |
| **PyTorch Forward Hook Fuzzer** | Consulted | Informed | **Accountable / Responsible** | Consulted | Informed |
| **CLI & Rich Terminal UX** | Informed | Informed | Consulted | **Accountable / Responsible** | Informed |
| **JSON / SARIF Reporting** | Informed | Consulted | Informed | **Accountable / Responsible** | Informed |
| **CI/CD & GitHub Actions** | Responsible | Informed | Informed | **Accountable / Responsible** | Informed |
| **Benchmark & Validation Suite** | Consulted | Responsible | Responsible | **Accountable** | Informed |
| **College Report & Presentation** | Consulted | Consulted | Consulted | Consulted | **Responsible** |
