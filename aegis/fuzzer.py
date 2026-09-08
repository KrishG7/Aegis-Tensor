"""Dynamic Fuzzing Engine for Aegis-Tensor.

Hooks into PyTorch models using forward hooks, measuring intermediate layer
activations under fuzzing to detect Sleeper Agent Trojan triggers via L_infinity norm spikes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
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


class ActivationHookManager:
    """Manages PyTorch forward hooks to record activation statistics."""

    def __init__(self, model: Any):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for dynamic fuzzing. Install with `pip install torch`.")
        self.model = model
        self.hooks: List[Any] = []
        self.current_activations: Dict[str, LayerActivationStat] = {}
        self._register_hooks()

    def _hook_fn(self, name: str) -> Callable:
        def hook(module: Any, input_tensors: Any, output_tensor: Any):
            # Flatten or extract raw tensor if output is a tuple/dataclass
            tensor = output_tensor
            if isinstance(tensor, (tuple, list)):
                tensor = tensor[0]
            elif hasattr(tensor, "last_hidden_state"):
                tensor = tensor.last_hidden_state

            if hasattr(tensor, "detach") and hasattr(tensor, "abs"):
                with torch.no_grad():
                    detached = tensor.detach().float()
                    # L_infinity norm: max absolute activation
                    l_inf = float(torch.max(torch.abs(detached)).cpu().item())
                    mean_val = float(torch.mean(detached).cpu().item())
                    std_val = float(torch.std(detached).cpu().item()) if detached.numel() > 1 else 0.0

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
            # Hook non-container modules (layers that perform operations)
            children = list(module.children())
            if not children and len(list(module.parameters())) > 0:
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
        sample_inputs_generator: Callable[[int], Any],
        baseline_input: Any,
        num_iterations: int = 50,
    ) -> TrojanScanReport:
        """Run fuzzing over generated sample inputs and compare against baseline activations."""
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
                test_input = sample_inputs_generator(i)
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
                        input_tag=f"fuzz_step_{i + 1}",
                        layer_stats=layer_stats,
                        max_l_inf=iter_max,
                        highest_layer=iter_top_layer,
                    )
                )

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
