"""Tests for Phase 11: Non-GNN Baselines.

Verifies:
1. Baseline A (RandomOptimizer):
   - Successfully optimizes reducible circuits.
   - Preserves exact quantum equivalence against the initial circuit.
   - Accurately logs candidate evaluations, equivalence checks, and runtime.
2. Baseline B (GreedyOptimizer):
   - Deterministically selects the best immediate valid improvement at each step.
   - Preserves exact quantum equivalence against the initial circuit.
   - Demonstrates monotonically non-increasing gate count.
3. Comparative execution on held-out circuits:
   - Both baselines run on the same circuits.
   - Comprehensive performance records generated without errors.
4. Baseline C (AggregateFeatureMLP):
   - Extracts 6-dim aggregate features from circuits.
   - Forward pass produces valid predictions without NaNs.
"""

import pytest
import torch

from qco.circuits.representation import CircuitData, GateOperation
from qco.models.mlp_baseline import AggregateFeatureMLP, extract_aggregate_features
from qco.optimization.greedy_optimizer import GreedyOptimizer
from qco.optimization.random_optimizer import RandomOptimizer
from qco.verification.equivalence import check_equivalence


def _create_test_circuit() -> CircuitData:
    """Create a circuit with multiple reduction opportunities across qubits."""
    # Reductions:
    # - H H on qubit 0 (cancellation -> -2 gates)
    # - CX CX on qubits (0, 1) (cancellation -> -2 gates)
    # - RX(0.3) RX(0.5) on qubit 1 (combination -> -1 gate)
    # - RY(0.0) on qubit 0 (removal -> -1 gate)
    return CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),
            GateOperation("RX", (1,), (0.3,)),
            GateOperation("RX", (1,), (0.5,)),
            GateOperation("RY", (0,), (0.0,)),
            GateOperation("X", (0,)),
        ],
    )


def test_random_optimizer_execution():
    """Verify Baseline A (RandomOptimizer) optimizes circuit while preserving equivalence."""
    circuit = _create_test_circuit()
    opt = RandomOptimizer(max_iterations=10, seed=42)
    res = opt.optimize(circuit)

    # 1. Verification of equivalence (critical)
    assert res.is_equivalent is True
    equiv_check = check_equivalence(circuit, res.final_circuit)
    assert equiv_check.equivalent is True

    # 2. Gate reduction achieved
    assert res.final_cost.total_gates < res.initial_cost.total_gates
    assert res.gate_reduction > 0

    # 3. Metrics recorded properly
    assert res.candidate_evaluations > 0
    assert res.equivalence_checks > 0
    assert res.iterations > 0
    assert res.runtime > 0.0
    assert len(res.history) > 0


def test_greedy_optimizer_execution():
    """Verify Baseline B (GreedyOptimizer) executes greedy descent while preserving equivalence."""
    circuit = _create_test_circuit()
    opt = GreedyOptimizer(max_iterations=10)
    res = opt.optimize(circuit)

    # 1. Verification of equivalence (critical)
    assert res.is_equivalent is True
    equiv_check = check_equivalence(circuit, res.final_circuit)
    assert equiv_check.equivalent is True

    # 2. Monotonic reduction
    assert res.final_cost.total_gates < res.initial_cost.total_gates
    assert res.gate_reduction > 0

    # 3. Check step history monotonicity
    for step in res.history:
        assert step.gate_count_after < step.gate_count_before

    # 4. Standard fields recorded
    d = res.to_dict()
    assert "optimizer_name" in d
    assert "gate_reduction" in d
    assert "runtime" in d


def test_baselines_on_irreducible_circuit():
    """Verify both baselines handle irreducible circuits gracefully."""
    irreducible = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("RZ", (0,), (0.7,)),
        ],
    )

    rand_opt = RandomOptimizer(max_iterations=5, seed=42)
    res_rand = rand_opt.optimize(irreducible)
    assert res_rand.gate_reduction == 0
    assert res_rand.final_cost.total_gates == 3
    assert res_rand.is_equivalent is True

    greedy_opt = GreedyOptimizer(max_iterations=5)
    res_greedy = greedy_opt.optimize(irreducible)
    assert res_greedy.gate_reduction == 0
    assert res_greedy.final_cost.total_gates == 3
    assert res_greedy.is_equivalent is True


def test_comparative_baseline_run_on_held_out_circuits():
    """Verify both baselines execute across a collection of circuits."""
    circuits = [
        # Circuit 1: CNOT pair and self-inverse
        CircuitData(
            num_qubits=2,
            gates=[
                GateOperation("CX", (0, 1)),
                GateOperation("CX", (0, 1)),
                GateOperation("Z", (0,)),
                GateOperation("Z", (0,)),
            ],
        ),
        # Circuit 2: Rotation combination
        CircuitData(
            num_qubits=2,
            gates=[
                GateOperation("RY", (1,), (0.4,)),
                GateOperation("RY", (1,), (0.6,)),
                GateOperation("H", (0,)),
            ],
        ),
    ]

    rand_opt = RandomOptimizer(max_iterations=10, seed=123)
    greedy_opt = GreedyOptimizer(max_iterations=10)

    for c in circuits:
        res_r = rand_opt.optimize(c)
        res_g = greedy_opt.optimize(c)

        # Both must produce valid equivalent outputs
        assert res_r.is_equivalent is True
        assert res_g.is_equivalent is True
        assert res_r.final_cost.total_gates <= res_r.initial_cost.total_gates
        assert res_g.final_cost.total_gates <= res_g.initial_cost.total_gates


def test_aggregate_feature_mlp_baseline():
    """Verify Baseline C (AggregateFeatureMLP) feature extraction and forward pass."""
    circuit = _create_test_circuit()
    feats = extract_aggregate_features(circuit)

    assert feats.shape == (6,)
    # Qubit count
    assert feats[0].item() == 2.0
    # Total gates
    assert feats[1].item() == 8.0
    # CNOT count
    assert feats[2].item() == 2.0
    # No NaNs
    assert not torch.isnan(feats).any()

    # Pass through MLP
    mlp = AggregateFeatureMLP(input_dim=6, hidden_dim=16, output_dim=1)
    out = mlp(feats.unsqueeze(0))
    assert out.shape == (1, 1)
    assert not torch.isnan(out).any()
