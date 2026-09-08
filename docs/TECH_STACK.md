# Aegis-Tensor: Technology Stack Specification

Aegis-Tensor combines a systems-level Rust core with an accessible Python ML ecosystem to achieve both bare-metal performance and seamless deep learning framework interoperability.

---

## 1. System Components & Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                       CLI & User API                        │
│            Typer (CLI)  •  Rich (Terminal UI)               │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌──────────────────────────────┐      ┌──────────────────────────────┐
│        Rust Core Engine      │      │    Dynamic Fuzzing Engine    │
│  • Memory Mapping (memmap2)  │      │  • PyTorch forward_hooks     │
│  • Shannon Entropy (H(X))    │      │  • L_infinity Norm Analysis  │
│  • Benford's Law (MAD)       │      │  • Dynamic Trigger Fuzzer    │
│  • Rayon Multi-threading     │      │  • NumPy Array Ops           │
└──────────────┬───────────────┘      └──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│     PyO3 / Maturin Bridge    │
│  • PyO3 v0.23 (abi3-py39)    │
│  • Zero-copy data exchange   │
└──────────────────────────────┘
```

---

## 2. Core Technologies

### 2.1 Systems & Static Scanning Core (Rust)
| Technology | Version | Purpose | Rationale |
| :--- | :--- | :--- | :--- |
| **Rust** | 2021 Edition (1.80+) | Core systems engine | Zero-cost abstractions, thread safety, and direct hardware memory control. |
| **memmap2** | `0.9.x` | Virtual memory mapping | Allows zero-copy inspection of multi-gigabyte models without allocating memory buffers. |
| **safetensors** | `0.4.x` / `0.5.x` | Model file parsing | Official Hugging Face format parser; extracts tensor headers and offsets zero-copy. |
| **rayon** | `1.10.x` | Data parallelism | Distributes per-tensor statistical calculations across all available CPU cores. |
| **serde / serde_json** | `1.0.x` | Serialization | Formats structured scan reports and metadata into serializable types. |
| **byteorder** | `1.5.x` | Binary decoding | Low-level byte manipulation for float conversions and bit-plane extraction. |

### 2.2 Foreign Function Interface (FFI) & Packaging
| Technology | Version | Purpose | Rationale |
| :--- | :--- | :--- | :--- |
| **PyO3** | `0.23.x` | Rust-Python FFI | Native bindings between Rust structs and Python objects with minimal overhead. |
| **abi3-py39** | Stable ABI | Forward compatibility | Builds a single universal binary compatible across Python 3.9 through 3.14+. |
| **Maturin** | `1.15.x` | Build tool & wheel builder | Seamless build orchestration combining Cargo and pip/pyproject.toml standards. |

### 2.3 Dynamic Fuzzing & Machine Learning (Python)
| Technology | Version | Purpose | Rationale |
| :--- | :--- | :--- | :--- |
| **Python** | `>=3.9, <=3.14` | Runtime & User Interface | High-level language standard for AI engineering and data science ecosystems. |
| **PyTorch** | `>=2.0.0` | Deep learning execution | Provides `module.register_forward_hook` for introspection of neural activations. |
| **NumPy** | `>=1.24.0` | Numerical utilities | Array manipulation, statistical baseline comparison, and metrics calculation. |

### 2.4 Terminal User Experience & Tooling
| Technology | Version | Purpose | Rationale |
| :--- | :--- | :--- | :--- |
| **Typer** | `>=0.9.0` | CLI Framework | Type-hint driven CLI parser with subcommands, validation, and automated help pages. |
| **Rich** | `>=13.0.0` | Terminal rendering | Color-coded status tables, spinners, progress bars, and alerts for security teams. |
| **Pytest** | `>=7.0.0` | Test runner | Unit, integration, and end-to-end testing of CLI commands and fuzzer hooks. |

---

## 3. Hardware Requirements & Performance Targets
- **CPU:** 4+ physical cores recommended (scales linearly via Rayon parallel iterators).
- **RAM:** 2 GB system RAM minimum (memory-mapped I/O uses OS page cache rather than heap allocations).
- **Storage:** NVMe SSD recommended for maximizing disk read throughput during multi-gigabyte scans.
- **Accelerator:** Optional CUDA/MPS device for accelerated forward pass inference during dynamic fuzzing.
