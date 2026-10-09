"""Unit and Integration Tests for Phase 20 Reproducibility Pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.experiments.reproducibility import (
    CircuitCorrectnessCheck,
    ReproducibilityReport,
    ReproducibilityRunner,
    record_environment_metadata,
    verify_optimized_circuit,
)


def test_record_environment_metadata() -> None:
    """Verify that environment metadata captures all required Phase 20 fields."""
    meta = record_environment_metadata(seed=42)

    assert meta["random_seed"] == 42
    assert "python_version" in meta
    assert "torch_version" in meta
    assert "torch_geometric_version" in meta
    assert "qiskit_version" in meta
    assert "git_commit" in meta
    assert "supported_rules" in meta
    assert len(meta["supported_rules"]) >= 5


def test_verify_optimized_circuit_success() -> None:
    """Verify that circuit correctness check validates equivalent transformation."""
    # Circuit with redundant H H
    orig = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
        ],
    )
    # Optimized circuit where H H is eliminated
    opt = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("CX", (0, 1)),
        ],
    )

    check = verify_optimized_circuit(orig, opt, tolerance=1e-7)

    assert isinstance(check, CircuitCorrectnessCheck)
    assert check.original_reconstructed is True
    assert check.optimized_reconstructed is True
    assert check.is_equivalent is True
    assert check.original_gates == 3
    assert check.optimized_gates == 1
    assert check.recalculated_gates == 1
    assert check.original_cnots == 1
    assert check.optimized_cnots == 1
    assert check.metrics_match is True
    assert check.passed_all_checks is True


def test_verify_optimized_circuit_detects_non_equivalent() -> None:
    """Verify that circuit correctness check rejects non-equivalent circuits."""
    orig = CircuitData(
        num_qubits=2,
        gates=[GateOperation("X", (0,))],
    )
    non_eq = CircuitData(
        num_qubits=2,
        gates=[GateOperation("Z", (0,))],
    )

    check = verify_optimized_circuit(orig, non_eq, tolerance=1e-7)

    assert check.is_equivalent is False
    assert check.passed_all_checks is False


def test_end_to_end_reproducibility_pipeline() -> None:
    """Test full automated reproducibility pipeline on a minimal test problem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ReproducibilityRunner(output_dir=tmpdir, seed=42)
        report = runner.run_full_pipeline(
            num_circuits=4,
            training_epochs=2,
            max_optimization_iters=3,
        )

        assert isinstance(report, ReproducibilityReport)
        assert report.overall_success is True
        assert report.dataset_summary["total_observations"] > 0
        assert report.dataset_summary["leakage_free"] is True
        assert len(report.correctness_checks) > 0

        # Check all output files generated on disk
        out_path = Path(tmpdir)
        assert (out_path / "reproducibility_report.json").exists()
        assert (out_path / "splits" / "train.jsonl").exists()
        assert (out_path / "splits" / "split_metadata.json").exists()
        assert (out_path / "checkpoints" / "best_model.pt").exists()
        assert (out_path / "verified_circuits").exists()
