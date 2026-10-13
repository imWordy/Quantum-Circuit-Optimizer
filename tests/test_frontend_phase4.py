"""Tests for Frontend Phase 4 Interactive Optimizer Lab & Timeline Playback."""

from __future__ import annotations

import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.experiments.demonstration import FinalDemonstrator, DemonstrationReport, IterationDecisionTrace
from qco.frontend.presets import get_redundant_cancellation_circuit
from qco.models.gnn import CircuitGNN


def test_final_demonstrator_generates_replayable_traces() -> None:
    """Verify FinalDemonstrator produces step traces with attached circuit_after states."""
    circ = get_redundant_cancellation_circuit()
    model = CircuitGNN(hidden_dim=32, gnn_type="gcn")
    demonstrator = FinalDemonstrator(model=model, max_iterations=5, max_evaluations_per_iter=5)

    report = demonstrator.run_demonstration(circ, verbose=False)

    assert isinstance(report, DemonstrationReport)
    assert report.is_equivalent is True
    assert len(report.decision_traces) > 0

    for trace in report.decision_traces:
        assert isinstance(trace, IterationDecisionTrace)
        assert trace.exact_equivalence_verified is True
        assert trace.accepted is True
        # Check that circuit_after is populated for scrubbing playback
        assert trace.circuit_after is not None
        assert isinstance(trace.circuit_after, CircuitData)
        assert trace.gate_count_after == len(trace.circuit_after.gates)


def test_demonstration_report_serialization_for_download() -> None:
    """Verify that demonstration report serializes cleanly to JSON dictionary."""
    circ = get_redundant_cancellation_circuit()
    model = CircuitGNN(hidden_dim=32, gnn_type="gcn")
    demonstrator = FinalDemonstrator(model=model, max_iterations=2, max_evaluations_per_iter=3)

    report = demonstrator.run_demonstration(circ, verbose=False)
    report_dict = report.to_dict()

    assert "circuit_id" in report_dict
    assert "decision_traces" in report_dict
    assert "original_cost" in report_dict
    assert "optimized_cost" in report_dict
    assert report_dict["is_equivalent"] is True
