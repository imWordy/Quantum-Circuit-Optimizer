"""Unit and integration tests for Phase 21 Final Demonstration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.experiments.demonstration import (
    DemonstrationReport,
    FinalDemonstrator,
    IterationDecisionTrace,
)
from qco.models.gnn import CircuitGNN


def _create_unseen_circuit() -> CircuitData:
    """Create an unseen test circuit containing multiple opportunities for optimization:

    - Two consecutive CX(0, 1) gates (cancellation)
    - Consecutive RZ(0, 0.5) and RZ(0, -0.5) (rotation combination to 0 -> removal)
    - Redundant H H on qubit 1 (self-inverse cancellation)
    """
    gates = [
        GateOperation("CX", (0, 1)),
        GateOperation("CX", (0, 1)),
        GateOperation("RZ", (0,), params=[0.5]),
        GateOperation("RZ", (0,), params=[-0.5]),
        GateOperation("H", (1,)),
        GateOperation("H", (1,)),
        GateOperation("X", (0,)),
    ]
    return CircuitData(num_qubits=2, gates=gates)


def test_final_demonstrator_pipeline_and_report() -> None:
    """Verify that FinalDemonstrator runs through the entire 5-stage pipeline."""
    unseen = _create_unseen_circuit()
    gnn = CircuitGNN(hidden_dim=32, gnn_type="gcn")
    demonstrator = FinalDemonstrator(model=gnn, max_iterations=10, max_evaluations_per_iter=10)

    report = demonstrator.run_demonstration(unseen, verbose=False)

    assert isinstance(report, DemonstrationReport)
    assert report.is_equivalent is True
    # The redundant pairs should be eliminated, reducing gate count significantly
    assert report.optimized_cost.total_gates < report.original_cost.total_gates
    assert len(report.decision_traces) > 0

    # Verify decision trace details
    first_trace = report.decision_traces[0]
    assert isinstance(first_trace, IterationDecisionTrace)
    assert first_trace.exact_equivalence_verified is True
    assert first_trace.accepted is True
    assert len(first_trace.ranked_candidates) > 0

    # Verify report formatting
    table_str = report.format_summary_table()
    assert "QUANTUM CIRCUIT OPTIMIZATION REPORT" in table_str
    assert "Gate count" in table_str
    assert "CNOT count" in table_str
    assert "Depth" in table_str
    assert "YES" in table_str


def test_final_demonstrator_save() -> None:
    """Verify saving demonstration artifacts to disk."""
    unseen = _create_unseen_circuit()
    gnn = CircuitGNN(hidden_dim=32, gnn_type="gcn")
    demonstrator = FinalDemonstrator(model=gnn, max_iterations=5, max_evaluations_per_iter=5)

    report = demonstrator.run_demonstration(unseen, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        demonstrator.save_demonstration(report, output_dir=tmpdir)
        p = Path(tmpdir)
        assert (p / "demonstration_report.json").exists()
        assert (p / "demonstration_report.txt").exists()
        assert (p / "original_circuit.json").exists()
        assert (p / "optimized_circuit.json").exists()
