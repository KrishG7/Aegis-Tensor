"""Unit tests for Benchmark Suite and Evaluation Engine."""

import sys
from pathlib import Path

import pytest

# Ensure workspace root in path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from benchmarks.run_benchmark import (
    ConfusionMatrix,
    ThreatConfusionMatrix,
    MemoryTracker,
    create_mock_safetensors_file,
    evaluate_detection_confusion_matrix,
    evaluate_threat_confusion_matrix,
    run_benchmark_suite,
    BenchmarkMetricRecord,
    export_plot,
)


# ---------------------------------------------------------------------------
# Confusion Matrix Tests
# ---------------------------------------------------------------------------

def test_confusion_matrix_perfect_classification():
    cm = ConfusionMatrix(
        true_positives=10,
        false_positives=0,
        true_negatives=90,
        false_negatives=0,
    )
    assert cm.true_positive_rate == 1.0  # Recall: 10 / (10 + 0)
    assert cm.false_positive_rate == 0.0  # FPR: 0 / (0 + 90)
    assert cm.precision == 1.0  # Precision: 10 / (10 + 0)
    assert cm.f1_score == 1.0


def test_confusion_matrix_mixed_classification():
    cm = ConfusionMatrix(
        true_positives=8,
        false_positives=2,
        true_negatives=88,
        false_negatives=2,
    )
    assert cm.true_positive_rate == 0.8  # 8 / 10
    assert pytest.approx(cm.false_positive_rate, rel=1e-3) == 2 / 90
    assert cm.precision == 0.8  # 8 / (8 + 2)
    assert pytest.approx(cm.f1_score, rel=1e-3) == 0.8


def test_confusion_matrix_zero_division_safety():
    cm = ConfusionMatrix(
        true_positives=0,
        false_positives=0,
        true_negatives=0,
        false_negatives=0,
    )
    assert cm.true_positive_rate == 0.0
    assert cm.false_positive_rate == 0.0
    assert cm.precision == 0.0
    assert cm.f1_score == 0.0


def test_threat_confusion_matrix():
    tm = ThreatConfusionMatrix(
        clean_total=40,
        clean_pred_clean=38,
        clean_pred_stego=2,
        clean_pred_trojan=0,
        stego_total=10,
        stego_pred_clean=1,
        stego_pred_stego=9,
        stego_pred_trojan=0,
        trojan_total=5,
        trojan_pred_clean=0,
        trojan_pred_stego=0,
        trojan_pred_trojan=5,
    )
    assert pytest.approx(tm.clean_recall, rel=1e-3) == 38 / 40
    assert pytest.approx(tm.stego_recall, rel=1e-3) == 9 / 10
    assert tm.trojan_recall == 1.0
    assert tm.overall_accuracy == (38 + 9 + 5) / 55


# ---------------------------------------------------------------------------
# MemoryTracker Tests
# ---------------------------------------------------------------------------

def test_memory_tracker_lifecycle():
    with MemoryTracker() as tracker:
        assert tracker.initial_rss_mb >= 0.0
        data = [i for i in range(100_000)]
        tracker.sample()
        assert tracker.peak_rss_mb >= tracker.initial_rss_mb

    assert tracker.peak_rss_mb >= 0.0


# ---------------------------------------------------------------------------
# Synthetic Model Generation & Evaluation Tests
# ---------------------------------------------------------------------------

def test_create_mock_safetensors_and_evaluate(tmp_path):
    model_path = tmp_path / "test_eval.safetensors"
    ground_truth = create_mock_safetensors_file(
        output_path=model_path,
        num_tensors=20,
        tensor_elements=1000,
        inject_anomalies=True,
    )

    assert model_path.exists()
    assert len(ground_truth) == 20
    assert any(ground_truth.values())

    # Evaluate confusion matrix
    cm = evaluate_detection_confusion_matrix(model_path, ground_truth)
    assert isinstance(cm, ConfusionMatrix)
    assert cm.true_positives > 0
    assert cm.f1_score > 0.0

    # Evaluate multi-threat confusion matrix
    tm = evaluate_threat_confusion_matrix(model_path, ground_truth)
    assert isinstance(tm, ThreatConfusionMatrix)
    assert tm.overall_accuracy > 0.0


def test_run_benchmark_suite_quick(tmp_path):
    records, threat_matrix = run_benchmark_suite(
        iterations=1,
        quick=True,
        output_dir=tmp_path,
    )
    assert len(records) == 1
    rec = records[0]
    assert rec.scale == "100M"
    assert rec.num_tensors == 50
    assert rec.mean_latency_sec > 0.0
    assert rec.zero_copy_verified is True
    assert isinstance(threat_matrix, ThreatConfusionMatrix)

    # Test plot generation if matplotlib is installed
    try:
        import matplotlib
        plot_file = tmp_path / "plot.png"
        export_plot(records, plot_file)
        assert plot_file.exists()
    except ImportError:
        pass
