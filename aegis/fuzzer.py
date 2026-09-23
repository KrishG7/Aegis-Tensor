"""Dynamic Fuzzing Engine for Aegis-Tensor.

Hooks into PyTorch models using forward hooks, measuring intermediate layer
activations under fuzzing to detect Sleeper Agent Trojan triggers via L_infinity norm spikes.
"""

from dataclasses import dataclass, fields, field, is_dataclass
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LayerActivationStat:
    layer_name: str
    l_inf_norm: float
    mean_activation: float
    std_activation: float


@dataclass
class FuzzIterationReport:
    iteration: int
    input_tag: str
    layer_stats: Dict[str, LayerActivationStat]
    max_l_inf: float
    highest_layer: str


@dataclass
class TrojanScanReport:
    model_name: str
    num_fuzz_samples: int
    baseline_l_inf: float
    peak_fuzzed_l_inf: float
    spike_ratio: float
    suspected_trojan: bool
    suspicious_layers: List[str]
    iteration_reports: List[FuzzIterationReport] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input perturbation generators (Issue #6)
# ---------------------------------------------------------------------------

FuzzCase = Tuple[str, Any]  # (input_tag, input_tensor)


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for dynamic fuzzing. Install with `pip install torch`.")


def generate_gaussian_noise(
    shape: Tuple[int, ...],
    sigma: float = 1.0,
    generator: Optional["torch.Generator"] = None,
) -> "torch.Tensor":
    """Zero-mean Gaussian noise with standard deviation ``sigma``.

    Sweeping ``sigma`` across a wide range probes activation behaviour far
    outside the benign input distribution.
    """
    _require_torch()
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return torch.randn(tuple(shape), generator=generator) * sigma


def generate_boundary_inputs(
    shape: Tuple[int, ...],
    magnitude: float = 10.0,
    num_impulses: int = 3,
) -> List[FuzzCase]:
    """Extreme-value inputs: all zeros, saturated +/-magnitude and sparse delta impulses.

    A delta impulse sets a single feature to ``magnitude`` and everything else
    to zero, isolating triggers keyed on one input position.
    """
    _require_torch()
    shape = tuple(shape)
    cases: List[FuzzCase] = [
        ("boundary_zeros", torch.zeros(shape)),
        (f"boundary_pos_{magnitude:g}", torch.full(shape, float(magnitude))),
        (f"boundary_neg_{magnitude:g}", torch.full(shape, -float(magnitude))),
    ]

    per_sample = int(np.prod(shape[1:])) if len(shape) > 1 else int(np.prod(shape))
    if per_sample > 0 and num_impulses > 0:
        # Spread impulses evenly: first, middle and last features, etc.
        positions = sorted({int(p) for p in np.linspace(0, per_sample - 1, num_impulses)})
        for pos in positions:
            x = torch.zeros(shape)
            if len(shape) > 1:
                x.view(shape[0], -1)[:, pos] = float(magnitude)
            else:
                x.view(-1)[pos] = float(magnitude)
            cases.append((f"boundary_delta_{pos}", x))
    return cases


def generate_patch_triggers(
    shape: Tuple[int, ...],
    patch_size: Tuple[int, int] = (8, 8),
    intensity: float = 1.0,
    positions: Tuple[str, ...] = ("bottom_right", "top_left", "center"),
    base_input: Optional["torch.Tensor"] = None,
) -> List[FuzzCase]:
    """BadNets-style triggers: a high-contrast checkerboard patch stamped on an image.

    The patch is written to the last two (spatial) dimensions, so ``shape`` is
    expected to look like (N, C, H, W), (C, H, W) or (H, W).
    """
    _require_torch()
    shape = tuple(shape)
    if len(shape) < 2:
        raise ValueError("Patch triggers need at least 2 spatial dimensions (H, W)")

    height, width = shape[-2], shape[-1]
    ph, pw = min(patch_size[0], height), min(patch_size[1], width)
    # Checkerboard of +intensity / -intensity (maximum local contrast)
    rows = torch.arange(ph).unsqueeze(1)
    cols = torch.arange(pw).unsqueeze(0)
    patch = torch.where((rows + cols) % 2 == 0, float(intensity), -float(intensity))

    anchors = {
        "top_left": (0, 0),
        "top_right": (0, width - pw),
        "bottom_left": (height - ph, 0),
        "bottom_right": (height - ph, width - pw),
        "center": ((height - ph) // 2, (width - pw) // 2),
    }
    cases: List[FuzzCase] = []
    for pos in positions:
        if pos not in anchors:
            raise ValueError(f"Unknown patch position '{pos}'. Choose from {sorted(anchors)}")
        top, left = anchors[pos]
        x = base_input.clone().float() if base_input is not None else torch.zeros(shape)
        x[..., top:top + ph, left:left + pw] = patch
        cases.append((f"patch_{pos}_{ph}x{pw}", x))
    return cases


def generate_token_perturbations(
    vocab_size: int,
    seq_len: int,
    batch_size: int = 1,
    generator: Optional["torch.Generator"] = None,
) -> List[FuzzCase]:
    """Token-ID perturbations for language models.

    Covers extreme repetition of a single token, alternating token pairs,
    vocabulary boundary IDs and rarely used tail-of-vocabulary tokens.
    """
    _require_torch()
    if vocab_size < 2 or seq_len < 1:
        raise ValueError("vocab_size must be >= 2 and seq_len >= 1")
    shape = (batch_size, seq_len)
    rand_tok = int(torch.randint(0, vocab_size, (1,), generator=generator).item())
    tail_start = max(0, int(vocab_size * 0.95))  # last 5% of the vocabulary

    alternating = torch.tensor([rand_tok, vocab_size - 1]).repeat(seq_len // 2 + 1)[:seq_len]
    return [
        (f"token_repeat_{rand_tok}", torch.full(shape, rand_tok, dtype=torch.long)),
        ("token_repeat_min_id", torch.zeros(shape, dtype=torch.long)),
        ("token_repeat_max_id", torch.full(shape, vocab_size - 1, dtype=torch.long)),
        ("token_alternating", alternating.unsqueeze(0).repeat(batch_size, 1)),
        ("token_vocab_tail", torch.randint(tail_start, vocab_size, shape, generator=generator)),
    ]


def build_auto_fuzz_suite(
    input_shape: Optional[Tuple[int, ...]] = None,
    vocab_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0),
    patch_size: Tuple[int, int] = (8, 8),
    seed: Optional[int] = None,
) -> List[FuzzCase]:
    """Build the default fuzzing suite and interleave strategies round-robin.

    Interleaving means even a short run touches every strategy
    (noise, boundary, patch, token) instead of exhausting one first.
    """
    _require_torch()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    groups: List[List[FuzzCase]] = []

    if vocab_size is not None:
        groups.append(generate_token_perturbations(vocab_size, seq_len or 32, generator=gen))
    elif input_shape is not None:
        shape = tuple(input_shape)
        groups.append([(f"gaussian_sigma_{s:g}", generate_gaussian_noise(shape, s, gen)) for s in sigmas])
        groups.append(generate_boundary_inputs(shape))
        if len(shape) >= 3:  # image-like: (N, C, H, W) or (N, H, W)
            groups.append(generate_patch_triggers(shape, patch_size))
    else:
        raise ValueError("Automated fuzzing needs either input_shape or vocab_size")

    suite: List[FuzzCase] = []
    for i in range(max(len(g) for g in groups)):
        for g in groups:
            if i < len(g):
                suite.append(g[i])
    return suite


class ActivationHookManager:
    """Manages PyTorch forward hooks to record activation statistics."""

    def __init__(
        self,
        model: Any,
        module_types: Optional[
            Union[Type["nn.Module"], Tuple[Type["nn.Module"], ...]]
        ] = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for dynamic fuzzing. Install with `pip install torch`.")
        self.model = model
        if isinstance(module_types, type):
            module_types = (module_types,)
        self.module_types = module_types
        self.hooks: List[Any] = []
        self.current_activations: Dict[str, LayerActivationStat] = {}
        self._register_hooks()

    @staticmethod
    def _find_tensor(output: Any) -> Optional[Any]:
        """Find the first tensor in common PyTorch output containers."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, Mapping):
            for value in output.values():
                tensor = ActivationHookManager._find_tensor(value)
                if tensor is not None:
                    return tensor
        elif isinstance(output, (tuple, list)):
            for value in output:
                tensor = ActivationHookManager._find_tensor(value)
                if tensor is not None:
                    return tensor
        elif is_dataclass(output) and not isinstance(output, type):
            for output_field in fields(output):
                tensor = ActivationHookManager._find_tensor(
                    getattr(output, output_field.name)
                )
                if tensor is not None:
                    return tensor
        elif hasattr(output, "last_hidden_state"):
            return ActivationHookManager._find_tensor(output.last_hidden_state)
        return None

    def _hook_fn(self, name: str) -> Callable:
        def hook(module: Any, input_tensors: Any, output_tensor: Any):
            tensor = self._find_tensor(output_tensor)
            if tensor is not None:
                with torch.no_grad():
                    detached = tensor.detach().float().cpu()
                    if detached.numel() == 0:
                        return
                    # L_infinity norm: max absolute activation
                    l_inf = float(torch.max(torch.abs(detached)).item())
                    mean_val = float(torch.mean(detached).item())
                    std_val = float(torch.std(detached).item()) if detached.numel() > 1 else 0.0

                    self.current_activations[name] = LayerActivationStat(
                        layer_name=name,
                        l_inf_norm=l_inf,
                        mean_activation=mean_val,
                        std_activation=std_val,
                    )

        return hook

    def _register_hooks(self):
        """Recursively registers hooks on named modules."""
        for name, module in self.model.named_modules():
            if self.module_types is not None:
                should_hook = isinstance(module, self.module_types)
            else:
                # Hook non-container modules (layers that perform operations)
                children = list(module.children())
                should_hook = not children and len(list(module.parameters())) > 0
            if should_hook:
                h = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(h)

    def clear(self):
        self.current_activations.clear()

    def remove(self):
        """Remove all registered forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.current_activations.clear()


class DynamicTrojanFuzzer:
    """Fuzzes AI models to detect latent Sleeper Agent Trojans via activation spikes."""

    def __init__(self, model: Any, spike_threshold: float = 4.0):
        """
        Args:
            model: The PyTorch model to inspect.
            spike_threshold: Ratio of fuzzed L_infinity to baseline L_infinity to trigger suspicion.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for dynamic fuzzing.")
        self.model = model
        self.model.eval()
        self.spike_threshold = spike_threshold

    def run_fuzzing(
        self,
        sample_inputs_generator: Optional[Callable[[int], Any]] = None,
        baseline_input: Any = None,
        num_iterations: int = 50,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        vocab_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> TrojanScanReport:
        """Run fuzzing over generated sample inputs and compare against baseline activations.

        Manual mode: pass ``sample_inputs_generator`` and ``baseline_input``.
        Automated mode: leave ``sample_inputs_generator`` as None and pass
        ``input_shape`` (float models) or ``vocab_size`` (token models). The
        built-in suite (Gaussian noise, boundary values, patch triggers or
        token perturbations) is cycled for ``num_iterations`` steps.
        """
        auto_suite: Optional[List[FuzzCase]] = None
        if sample_inputs_generator is None:
            auto_suite = build_auto_fuzz_suite(
                input_shape=input_shape, vocab_size=vocab_size, seq_len=seq_len, seed=seed
            )
            if baseline_input is None:
                gen = torch.Generator().manual_seed(seed) if seed is not None else None
                if vocab_size is not None:
                    baseline_input = torch.randint(0, vocab_size, (1, seq_len or 32), generator=gen)
                else:
                    baseline_input = torch.randn(tuple(input_shape), generator=gen)
        elif baseline_input is None:
            raise ValueError("baseline_input is required when a custom sample_inputs_generator is given")

        hook_mgr = ActivationHookManager(self.model)

        try:
            # 1. Measure baseline activations
            hook_mgr.clear()
            with torch.no_grad():
                _ = self.model(baseline_input)
            baseline_stats = dict(hook_mgr.current_activations)
            baseline_max_l_inf = max((s.l_inf_norm for s in baseline_stats.values()), default=1.0)
            baseline_max_l_inf = max(baseline_max_l_inf, 1e-6)

            peak_fuzzed_l_inf = baseline_max_l_inf
            suspicious_layers: set[str] = set()
            iteration_reports: List[FuzzIterationReport] = []

            # 2. Dynamic fuzzing loop
            for i in range(num_iterations):
                if auto_suite is not None:
                    input_tag, test_input = auto_suite[i % len(auto_suite)]
                else:
                    input_tag, test_input = f"fuzz_step_{i + 1}", sample_inputs_generator(i)
                hook_mgr.clear()

                with torch.no_grad():
                    _ = self.model(test_input)

                layer_stats = dict(hook_mgr.current_activations)
                if not layer_stats:
                    continue

                iter_max = 0.0
                iter_top_layer = ""
                for name, stat in layer_stats.items():
                    if stat.l_inf_norm > iter_max:
                        iter_max = stat.l_inf_norm
                        iter_top_layer = name

                    # Check for anomalous activation spike relative to baseline layer stat
                    base_layer_stat = baseline_stats.get(name)
                    base_l_inf = base_layer_stat.l_inf_norm if base_layer_stat else 1.0
                    base_l_inf = max(base_l_inf, 1e-6)

                    if (stat.l_inf_norm / base_l_inf) >= self.spike_threshold:
                        suspicious_layers.add(name)

                if iter_max > peak_fuzzed_l_inf:
                    peak_fuzzed_l_inf = iter_max

                iteration_reports.append(
                    FuzzIterationReport(
                        iteration=i + 1,
                        input_tag=input_tag,
                        layer_stats=layer_stats,
                        max_l_inf=iter_max,
                        highest_layer=iter_top_layer,
                    )
                )

                if progress_callback is not None:
                    progress_callback(i + 1, num_iterations, iter_top_layer, iter_max)

            spike_ratio = peak_fuzzed_l_inf / baseline_max_l_inf
            suspected_trojan = spike_ratio >= self.spike_threshold or len(suspicious_layers) > 0

            return TrojanScanReport(
                model_name=getattr(self.model, "__class__", type(self.model)).__name__,
                num_fuzz_samples=num_iterations,
                baseline_l_inf=baseline_max_l_inf,
                peak_fuzzed_l_inf=peak_fuzzed_l_inf,
                spike_ratio=spike_ratio,
                suspected_trojan=suspected_trojan,
                suspicious_layers=sorted(list(suspicious_layers)),
                iteration_reports=iteration_reports,
            )

        finally:
            hook_mgr.remove()
