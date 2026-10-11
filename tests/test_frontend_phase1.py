"""Tests for Frontend Phase 1 Foundation."""

from __future__ import annotations

import pytest

from qco.circuits.representation import CircuitData
from qco.frontend.config import (
    apply_custom_css,
    get_default_sample_circuit,
    load_default_gnn_model,
)
from qco.models.gnn import CircuitGNN


def test_default_sample_circuit_properties() -> None:
    """Verify that default circuit has valid representation and known redundancies."""
    circ = get_default_sample_circuit()
    assert isinstance(circ, CircuitData)
    assert circ.num_qubits == 2
    assert len(circ.gates) == 7

    gate_names = [g.name for g in circ.gates]
    assert "H" in gate_names
    assert "CX" in gate_names
    assert "RZ" in gate_names


def test_load_default_gnn_model() -> None:
    """Verify default GNN model initializes and can run forward passes."""
    model = load_default_gnn_model()
    assert isinstance(model, CircuitGNN)
    assert not model.training


def test_app_importability() -> None:
    """Verify app.py module can be imported cleanly without syntax errors."""
    import importlib.util

    spec = importlib.util.find_spec("qco.frontend.app")
    assert spec is not None
