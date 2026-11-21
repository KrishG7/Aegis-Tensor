"""Default scan thresholds, shared by the CLI and the calibration benchmark.

They match the defaults of the Rust scan functions (src/lib.rs). Calibration results
record them, so a change here marks those results, and the claims that rest on them, stale.
"""

DEFAULT_THRESHOLDS = {
    "entropy": 7.92,  # pooled whole-tensor entropy, bits per byte
    "exponent": 6.5,  # sign/exponent byte-lane entropy (float32, bfloat16)
    # float16 only: its high byte also holds two mantissa bits. Largest clean float16 value in
    # calibration was 6.85 (Pythia/BLOOM attention biases), random data 7.6-8.0; 7.2 leaves
    # 0.35 bits of margin on each side (docs/calibration.md, P1).
    "exponent_f16": 7.2,
    "plane": 7.5,  # low mantissa bit-plane entropy floor
    "benford": None,  # Benford MAD; off by default
    # Entropy of the unpacked values of a quantized tensor, in bits, per format (packed tensors of
    # at least 100,000 values). Each is the highest clean value in calibration plus a margin; a
    # random payload scores about the value width (4 or 8). The 4-bit margins are thin (docs/calibration.md, P3).
    "packed_entropy": {"gptq_int4": 3.97, "bnb_4bit": 3.97, "gptq_int8": 7.70, "fp8_e4m3": 7.20},
}
