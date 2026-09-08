# Aegis-Tensor: Deep Architectural Specification

Aegis-Tensor provides multi-layer inspection of neural network weights by integrating mathematical cryptanalysis with dynamic runtime introspection.

---

## 1. Mathematical Foundations

### 1.1 Shannon Information Entropy Analysis
Shannon Entropy measures the degree of uncertainty, randomness, or information density contained within a discrete sequence of symbols:

$$H(X) = - \sum_{i=0}^{255} P(x_i) \log_2 P(x_i)$$

Where:
- $X$ is the byte stream extracted from a given tensor.
- $P(x_i)$ is the empirical probability of observing byte value $x_i \in [0, 255]$.
- $H(X) \in [0.0, 8.0]$ bits per byte.

#### Threat Detection Rationale:
- **Clean Weights:** Standard neural network parameters (trained with SGD/Adam and weight decay) conform to smooth Gaussian, Laplacian, or Student's-t distributions. Their byte representation exhibits structured repetition, yielding an entropy typically between **$5.2$ and $7.4$ bits/byte**.
- **Steganographic Injections:** Encrypted payloads (e.g. AES ciphertext), compressed executables (e.g. ZIP/ELF), or pseudorandom keystreams maximize information density, pushing entropy to **$\ge 7.92$ bits/byte** (near-maximal randomness).

```
Entropy Spectrum:
0.0                                6.0                 7.8   8.0
|-----------------------------------|-------------------|------|
  Zero / Constant Tensors            Clean Model Weights  Stego Payload Alert!
```

---

### 1.2 Benford's Law (First-Digit Phenomenon)
Benford's Law states that in naturally occurring numerical datasets spanning multiple orders of magnitude, the first non-zero leading digit $d \in \{1, 2, \dots, 9\}$ follows a logarithmic probability distribution:

$$P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$$

| Digit ($d$) | Theoretical $P(d)$ | Approximate % |
| :---: | :---: | :---: |
| **1** | $\log_{10}(2.000)$ | 30.1% |
| **2** | $\log_{10}(1.500)$ | 17.6% |
| **3** | $\log_{10}(1.333)$ | 12.5% |
| **4** | $\log_{10}(1.250)$ | 9.7% |
| **5** | $\log_{10}(1.200)$ | 7.9% |
| **6** | $\log_{10}(1.167)$ | 6.7% |
| **7** | $\log_{10}(1.143)$ | 5.8% |
| **8** | $\log_{10}(1.125)$ | 5.1% |
| **9** | $\log_{10}(1.111)$ | 4.6% |

#### Mean Absolute Deviation (MAD):
We quantify departure from natural Benford distribution using the MAD metric:

$$\text{MAD} = \frac{1}{9} \sum_{d=1}^9 |O(d) - E(d)|$$

Where $O(d)$ is the observed frequency and $E(d)$ is the expected Benford frequency.
- **$\text{MAD} \le 0.012$:** Close natural conformity.
- **$\text{MAD} \ge 0.035$:** Anomalous non-natural distribution, signaling artificial weight manipulation or synthetic payload interleaving.

---

### 1.3 Dynamic Activation Introspection ($L_\infty$ Norm Spikes)
Sleeper Agent Trojans remain dormant under normal inputs, producing baseline activation values. When triggered by a backdoor input $x_{trigger}$, dedicated internal neurons fire with extreme intensity to force the malicious output.

For layer $l$ producing activation tensor $A^{(l)}$:

$$\|A^{(l)}\|_\infty = \max_{i} |A^{(l)}_i|$$

The **Spike Ratio** $R^{(l)}$ compares the peak activation under fuzzed input to clean baseline input:

$$R^{(l)} = \frac{\|A^{(l)}_{\text{fuzz}}\|_{\infty}}{\max\left(\|A^{(l)}_{\text{baseline}}\|_{\infty}, \epsilon\right)}$$

If $R^{(l)} \ge \tau_{\text{spike}}$ (default $\tau = 4.0$), layer $l$ is marked as anomalous, isolating the specific sub-network harboring the Trojan trigger.

---

## 2. Memory-Mapped Static Scanner Pipeline

```
┌────────────────────────────────────────────────────────┐
│               .safetensors File on Disk               │
│  [8-byte Header Length N] [N-byte JSON] [Raw Tensors]  │
└──────────────────────────┬─────────────────────────────┘
                           │
             OS Virtual Memory Page Table
             (Zero Heap Copy via memmap2)
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                   SafeTensors Parser                   │
│   • Reads JSON metadata: tensor names, shapes, offsets │
│   • Obtains direct byte slices: &[u8]                  │
└──────────────────────────┬─────────────────────────────┘
                           │
            Rayon Parallel Work-Stealing Pool
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
       [Tensor 1]    [Tensor 2]    [Tensor N]
             │             │             │
      ┌──────┴──────┐      │             │
      │ Shannon     │      │             │
      │ Entropy     │      │             │
      ├─────────────┤      │             │
      │ Benford MAD │      │             │
      └──────┬──────┘      │             │
             └─────────────┼─────────────┘
                           │
                           ▼
            Aggregated TensorScanResult Vec
                           │
                           ▼
             PyO3 Native Python Conversion
```

---

## 3. Dynamic Fuzzer Pipeline

```
  Clean Input Baseline ────────┐
                               ▼
  ┌───────────────────────────────────────────────────────┐
  │                 PyTorch Target Model                  │
  │                                                       │
  │  Layer 1 ──► [Hook 1: Compute ||A_1||_inf] ──► ...   │
  │  Layer 2 ──► [Hook 2: Compute ||A_2||_inf] ──► ...   │
  │  Layer L ──► [Hook L: Compute ||A_L||_inf] ──► Output │
  └────────────────────────────┬──────────────────────────┘
                               ▲
  Adversarial Fuzzer ──────────┘
  (Random perturbations,
   boundary noise, patterns)
                               │
                               ▼
  ┌───────────────────────────────────────────────────────┐
  │              Anomaly Detection Engine                 │
  │  • Calculates R^(l) = ||A_fuzz|| / ||A_clean||        │
  │  • Evaluates Spike Threshold Tau                      │
  │  • Flags Suspicious Sub-network Components            │
  └───────────────────────────────────────────────────────┘
```
