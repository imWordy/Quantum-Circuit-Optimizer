"""Tests for Frontend Phase 5 Multi-Method Benchmark Arena & Plots."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from qco.frontend.components.benchmark_plots import (
    render_efficiency_scatter_plot,
    render_reduction_comparison_bar,
    render_runtime_bar_plot,
)


def _get_dummy_benchmark_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide realistic sample benchmark DataFrames for chart verification."""
    summary_df = pd.DataFrame(
        [
            {
                "method": "RandomOptimizer",
                "num_circuits": 3,
                "avg_gate_reduction": 1.33,
                "avg_cnot_reduction": 0.67,
                "avg_depth_reduction": 1.0,
                "avg_evaluations": 15.0,
                "avg_checks": 15.0,
                "avg_runtime_sec": 0.045,
                "all_equivalent": True,
            },
            {
                "method": "GreedyOptimizer",
                "num_circuits": 3,
                "avg_gate_reduction": 3.0,
                "avg_cnot_reduction": 1.0,
                "avg_depth_reduction": 2.33,
                "avg_evaluations": 28.0,
                "avg_checks": 28.0,
                "avg_runtime_sec": 0.082,
                "all_equivalent": True,
            },
            {
                "method": "GNNGuidedOptimizer",
                "num_circuits": 3,
                "avg_gate_reduction": 3.0,
                "avg_cnot_reduction": 1.0,
                "avg_depth_reduction": 2.33,
                "avg_evaluations": 8.0,
                "avg_checks": 8.0,
                "avg_runtime_sec": 0.038,
                "all_equivalent": True,
            },
        ]
    )

    raw_df = pd.DataFrame(
        [
            {"circuit_id": "c1", "method": "RandomOptimizer", "gate_reduction": 1, "candidate_evaluations": 10, "runtime": 0.03, "iterations": 2},
            {"circuit_id": "c2", "method": "RandomOptimizer", "gate_reduction": 2, "candidate_evaluations": 20, "runtime": 0.06, "iterations": 3},
            {"circuit_id": "c1", "method": "GNNGuidedOptimizer", "gate_reduction": 3, "candidate_evaluations": 6, "runtime": 0.02, "iterations": 2},
            {"circuit_id": "c2", "method": "GNNGuidedOptimizer", "gate_reduction": 3, "candidate_evaluations": 10, "runtime": 0.05, "iterations": 3},
        ]
    )
    return summary_df, raw_df


def test_render_reduction_comparison_bar() -> None:
    """Verify that reduction bar plot renders without error."""
    summary_df, _ = _get_dummy_benchmark_data()
    fig = render_reduction_comparison_bar(summary_df)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # gate reduction bar trace + cnot reduction bar trace
    assert fig.data[0].name == "Avg Gate Reduction"


def test_render_efficiency_scatter_plot() -> None:
    """Verify that efficiency scatter plot correctly maps evaluations vs reductions."""
    _, raw_df = _get_dummy_benchmark_data()
    fig = render_efficiency_scatter_plot(raw_df)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_render_runtime_bar_plot() -> None:
    """Verify that runtime horizontal bar plot renders correctly."""
    summary_df, _ = _get_dummy_benchmark_data()
    fig = render_runtime_bar_plot(summary_df)

    assert isinstance(fig, go.Figure)
    assert fig.data[0].orientation == "h"
    assert len(fig.data[0].x) == 3
