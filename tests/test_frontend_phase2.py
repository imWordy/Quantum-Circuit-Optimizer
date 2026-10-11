"""Tests for Frontend Phase 2 Circuit Studio & Visualizer."""

from __future__ import annotations

import pytest

from qco.circuits.representation import CircuitData
from qco.frontend.components.circuit_viewer import render_circuit_text
from qco.frontend.presets import (
    PRESET_CIRCUITS,
    get_bell_state_circuit,
    get_ghz_state_circuit,
    get_redundant_cancellation_circuit,
    get_teleportation_circuit,
)
from qco.metrics.circuit_cost import compute_circuit_cost


def test_preset_library_integrity() -> None:
    """Verify all educational preset circuits are valid CircuitData instances."""
    assert len(PRESET_CIRCUITS) >= 5

    for name, circ in PRESET_CIRCUITS.items():
        assert isinstance(circ, CircuitData)
        assert circ.num_qubits >= 2
        assert len(circ.gates) > 0
        cost = compute_circuit_cost(circ)
        assert cost.total_gates == len(circ.gates)
        assert cost.depth > 0


def test_specific_preset_features() -> None:
    """Verify characteristics of known educational presets."""
    bell = get_bell_state_circuit()
    assert bell.num_qubits == 2
    assert len(bell.gates) == 2
    assert bell.gates[0].name == "H"
    assert bell.gates[1].name == "CX"

    ghz = get_ghz_state_circuit()
    assert ghz.num_qubits == 3
    assert len(ghz.gates) == 3

    redundant = get_redundant_cancellation_circuit()
    cost = compute_circuit_cost(redundant)
    assert cost.cnot_count >= 2
    assert cost.total_gates >= 8


def test_render_circuit_text_wire_diagram() -> None:
    """Verify that wire diagram text rendering produces multi-line ASCII output."""
    bell = get_bell_state_circuit()
    diagram = render_circuit_text(bell)
    assert isinstance(diagram, str)
    assert len(diagram) > 0
    # Should contain wire identifiers like q or q_0
    assert "q" in diagram or "H" in diagram
