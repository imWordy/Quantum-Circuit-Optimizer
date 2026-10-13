"""Tests for Frontend Phase 3 GNN Graph & Ranking Inspector."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.frontend.components.graph_viewer import render_graph_network_plotly
from qco.frontend.presets import get_redundant_cancellation_circuit
from qco.models.gnn import CircuitGNN
from qco.models.ranker import CandidateRanker, RankedCandidate


def test_render_graph_network_plotly_generates_figure() -> None:
    """Verify that render_graph_network_plotly builds a valid Plotly Figure."""
    circ = get_redundant_cancellation_circuit()
    fig = render_graph_network_plotly(circ)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # edge trace + node trace
    # Check trace modes
    assert fig.data[0].mode == "lines"
    assert "markers" in fig.data[1].mode
    assert len(fig.data[1].text) == len(circ.gates)


def test_empty_circuit_graph_plotly() -> None:
    """Verify empty circuit handles cleanly without error."""
    empty_circ = CircuitData(num_qubits=2, gates=[])
    fig = render_graph_network_plotly(empty_circ)
    assert isinstance(fig, go.Figure)


def test_candidate_ranking_pool() -> None:
    """Verify that ranker scores and orders candidates for display in the inspector."""
    circ = get_redundant_cancellation_circuit()
    model = CircuitGNN(hidden_dim=32, gnn_type="gcn")
    ranker = CandidateRanker(model=model)

    ranked = ranker.rank_candidates(circ)
    assert len(ranked) > 0

    for i, rc in enumerate(ranked):
        assert isinstance(rc, RankedCandidate)
        assert rc.rank == i + 1
        assert isinstance(rc.predicted_score, float)
        assert hasattr(rc.candidate, "transformation_name")
        assert hasattr(rc.candidate, "transformation_location")

    # Verify descending sort order
    scores = [rc.predicted_score for rc in ranked]
    assert scores == sorted(scores, reverse=True)
