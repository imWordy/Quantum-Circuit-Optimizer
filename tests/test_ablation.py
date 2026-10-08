"""Tests for Ablation Studies Framework — Phase 19."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.experiments.ablation import AblationFramework, AblationStudyResult
from qco.models.gnn import CircuitGNN
from qco.models.mlp_baseline import AggregateFeatureMLP
from qco.transformations.rules import (
    CNOTCancellation,
    IdentityRemoval,
    SelfInverseCancellation,
)


def _create_test_circuit() -> CircuitData:
    """Create a sample circuit with redundant cancellations for ablation."""
    gates = [
        GateOperation("H", (0,)),
        GateOperation("H", (0,)),
        GateOperation("CX", (0, 1)),
        GateOperation("CX", (0, 1)),
        GateOperation("RZ", (1,), params=[0.0]),
        GateOperation("X", (0,)),
        GateOperation("X", (0,)),
    ]
    return CircuitData(num_qubits=2, gates=gates)


def test_structural_vs_aggregate_ablation() -> None:
    """Verify minimum Phase 19 comparison: A (Random), B (Greedy), C (MLP), D (GNN)."""
    circuits = [_create_test_circuit()]
    gnn = CircuitGNN(hidden_dim=32, gnn_type="gcn")
    mlp = AggregateFeatureMLP(hidden_dim=16)

    framework = AblationFramework(
        test_circuits=circuits,
        gnn_model=gnn,
        mlp_model=mlp,
        max_iterations=5,
        max_evaluations_per_iter=5,
    )

    result = framework.run_structural_vs_aggregate_ablation()

    assert isinstance(result, AblationStudyResult)
    assert result.study_name == "structural_vs_aggregate"
    # Should record 4 runs (1 circuit x 4 methods)
    assert len(result.records) == 4

    methods = {r.method for r in result.records}
    expected_methods = {"RandomOptimizer", "GreedyOptimizer", "MLPOptimizer", "GNNGuidedOptimizer"}
    assert methods == expected_methods

    # All final circuits must remain strictly equivalent
    for rec in result.records:
        assert rec.is_equivalent is True

    # Summary table must have 4 rows
    summary = result.summary_df
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 4
    assert set(summary["method"]) == expected_methods


def test_rule_subset_ablation_and_export() -> None:
    """Verify rule subset ablation and disk persistence."""
    circuits = [_create_test_circuit()]
    gnn = CircuitGNN(hidden_dim=32, gnn_type="gcn")

    framework = AblationFramework(
        test_circuits=circuits,
        gnn_model=gnn,
        max_iterations=3,
        max_evaluations_per_iter=4,
    )

    custom_subsets = {
        "self_inverse_only": [SelfInverseCancellation(), IdentityRemoval()],
        "cnot_only": [CNOTCancellation(), IdentityRemoval()],
    }

    results = framework.run_rule_subset_ablation(rule_subsets=custom_subsets)
    assert len(results) == 2
    assert "self_inverse_only" in results
    assert "cnot_only" in results

    # Test export
    with tempfile.TemporaryDirectory() as tmpdir:
        res = results["self_inverse_only"]
        framework.save_ablation_results(res, output_dir=tmpdir)

        csv_file = Path(tmpdir) / f"{res.study_name}_summary.csv"
        json_file = Path(tmpdir) / f"{res.study_name}_data.json"

        assert csv_file.exists()
        assert json_file.exists()

        loaded_df = pd.read_csv(csv_file)
        assert len(loaded_df) == 1
        assert "GNNGuidedOptimizer[self_inverse_only]" in loaded_df["method"].values
