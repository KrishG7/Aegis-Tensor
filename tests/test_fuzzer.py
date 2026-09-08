"""Tests for Dynamic Fuzzing Engine in aegis/fuzzer.py."""

import pytest
from aegis import TORCH_AVAILABLE, DynamicTrojanFuzzer, ActivationHookManager

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class SimpleLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the environment")
def test_activation_hook_manager():
    """Verify that forward hooks register, capture activations, and compute L_inf."""
    model = SimpleLinearModel()
    hook_mgr = ActivationHookManager(model)

    test_input = torch.randn(2, 10)
    _ = model(test_input)

    assert len(hook_mgr.current_activations) > 0
    assert "fc1" in hook_mgr.current_activations
    assert "fc2" in hook_mgr.current_activations

    fc1_stat = hook_mgr.current_activations["fc1"]
    assert fc1_stat.l_inf_norm >= 0.0

    # Cleanup
    hook_mgr.remove()
    assert len(hook_mgr.hooks) == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed in the environment")
def test_dynamic_trojan_fuzzer_clean_model():
    """Verify that a clean model does not trigger high spike ratio."""
    model = SimpleLinearModel()
    fuzzer = DynamicTrojanFuzzer(model, spike_threshold=10.0)

    report = fuzzer.run_fuzzing(
        sample_inputs_generator=lambda i: torch.randn(2, 10),
        baseline_input=torch.zeros(2, 10),
        num_iterations=10,
    )

    assert report.model_name == "SimpleLinearModel"
    assert report.num_fuzz_samples == 10
    assert not report.suspected_trojan
