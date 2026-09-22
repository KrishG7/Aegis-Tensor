"""Generate deterministic clean and synthetic steganography test models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


MODEL_SIZE = 256
LSB_PAYLOAD = b"Aegis-Tensor synthetic steganography payload"


def _clean_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        f"layer_{index}.weight": rng.normal(
            loc=0.0, scale=0.02, size=(MODEL_SIZE, MODEL_SIZE)
        ).astype(np.float32)
        for index in range(1, 6)
    }


def _naive_injection(rng: np.random.Generator) -> np.ndarray:
    """Represent encrypted payload bytes as float32 words."""
    payload = rng.integers(
        0, 256, size=MODEL_SIZE * MODEL_SIZE * 4, dtype=np.uint8
    )
    return payload.view(np.float32).reshape(MODEL_SIZE, MODEL_SIZE)


def _lsb_injection() -> np.ndarray:
    """Embed payload bits in the mantissa LSB of 100,000 float32 values."""
    elements = 100_000
    digits = np.arange(elements, dtype=np.uint32) % 9 + 1
    values = digits.astype(np.float32) + np.float32(0.25)
    words = values.view(np.uint32)

    payload_bits = np.unpackbits(
        np.frombuffer(LSB_PAYLOAD, dtype=np.uint8), bitorder="little"
    )
    repeated_bits = np.resize(payload_bits, elements).astype(np.uint32)
    words ^= repeated_bits
    return words.view(np.float32)


def generate_models(output_dir: Path, seed: int = 20260922) -> tuple[Path, Path]:
    """Write clean_model.safetensors and infected_model.safetensors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clean = _clean_weights(rng)
    infected = dict(clean)
    infected["layer_1.weight"] = _naive_injection(rng)
    infected["layer_2.weight"] = _lsb_injection()

    clean_path = output_dir / "clean_model.safetensors"
    infected_path = output_dir / "infected_model.safetensors"
    save_file(clean, str(clean_path), metadata={"generator": "aegis-stego-test"})
    save_file(infected, str(infected_path), metadata={"generator": "aegis-stego-test"})
    return clean_path, infected_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "fixtures",
        help="Directory for generated safetensors files.",
    )
    args = parser.parse_args()
    clean_path, infected_path = generate_models(args.output_dir)
    print(f"Generated {clean_path}")
    print(f"Generated {infected_path}")


if __name__ == "__main__":
    main()
