"""Tests for Phase 16: GNN-Guided Iterative Optimizer.

Verifies:
1. Every accepted candidate is independently verified equivalent (Phase 5 authority).
2. Primary cost (gate count) never increases — strictly non-increasing across steps.
3. Optimizer properly terminates upon convergence or max iterations without infinite loops.
4. No repeated state causes an infinite cycle.
5. Final optimized circuit is valid, equivalent to the initial circuit, and reconstructable.
6. Handles irreducible circuits cleanly with 0 reductions.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from qco.circuits.generator import CircuitGenerator
from qco.circuits.representation import CircuitData, GateOperation
from qco.data.dataset import split_dataset
from qco.data.dataset_builder import DatasetBuilder
from qco.data.gnn_dataset import GNNDataset, collate_gnn_batch
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.models.gnn import CircuitGNN
from qco.models.trainer import GNNTrainer
from qco.optimization.gnn_optimizer import GNNGuidedOptimizer
from qco.verification.equivalence import check_equivalence


def _get_trained_gnn():
    """Build and train a small CircuitGNN for optimizer testing."""
    gen = CircuitGenerator(min_qubits=2, max_qubits=3, min_gates=6, max_gates=16, seed=88)
    circuits = gen.generate_batch(20)

    builder = DatasetBuilder()
    builder.process_batch(circuits)

    splits = split_dataset(
        builder.records,
        circuit_store=builder.circuit_store,
        train_ratio=0.70,
        val_ratio=0.30,
        test_ratio=0.0,
        seed=42,
    )

    train_ds = GNNDataset(splits.train_records, splits.circuit_store)
    val_ds = GNNDataset(splits.val_records, splits.circuit_store)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_gnn_batch)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_gnn_batch)

    model = CircuitGNN(hidden_dim=32, num_gnn_layers=2)
    trainer = GNNTrainer(model, lr=0.005)
    trainer.fit(train_loader, val_loader, epochs=5)

    return model


def test_gnn_guided_optimizer_end_to_end():
    """Verify GNN-guided optimizer successfully optimizes reducible circuit while ensuring equivalence."""
    model = _get_trained_gnn()
    opt = GNNGuidedOptimizer(model=model, max_iterations=10, max_evaluations_per_iter=10)

    # Circuit with multiple reduction opportunities:
    # - H H (q0) -> cancels
    # - CX CX (q0, q1) -> cancels
    # - RX(0.3) RX(0.5) (q1) -> combines
    # - Z Z (q0) -> cancels
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),
            GateOperation("RX", (1,), (0.3,)),
            GateOperation("RX", (1,), (0.5,)),
            GateOperation("Z", (0,)),
            GateOperation("Z", (0,)),
        ],
    )

    result = opt.optimize(circuit)

    # 1. Equivalence guarantee: Final circuit MUST be equivalent to original circuit
    assert result.is_equivalent is True
    recheck_equiv = check_equivalence(circuit, result.final_circuit)
    assert recheck_equiv.equivalent is True

    # 2. Primary cost strictly decreases
    assert result.gate_reduction > 0
    assert result.final_cost.total_gates < result.initial_cost.total_gates

    # 3. Monotonicity across every accepted step
    assert len(result.history) > 0
    for step in result.history:
        assert step.accepted is True
        assert step.gate_count_after < step.gate_count_before

    # 4. Standard metrics logged
    assert result.optimizer_name == "GNNGuidedOptimizer"
    assert result.candidate_evaluations > 0
    assert result.equivalence_checks > 0
    assert result.runtime > 0.0
    assert result.iterations > 0


def test_gnn_guided_optimizer_termination_and_cycle_prevention():
    """Verify optimizer terminates and never loops infinitely on irreducible or finished circuits."""
    model = _get_trained_gnn()
    opt = GNNGuidedOptimizer(model=model, max_iterations=5)

    # Irreducible circuit
    irreducible = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("RZ", (1,), (0.75,)),
        ],
    )

    result = opt.optimize(irreducible)
    assert result.converged is True
    assert result.gate_reduction == 0
    assert result.final_cost.total_gates == 3
    assert result.is_equivalent is True
    assert len(result.history) == 0


def test_reconstructable_circuit_and_result_dict():
    """Verify result dictionary serialization and final circuit reconstructability."""
    model = _get_trained_gnn()
    opt = GNNGuidedOptimizer(model=model, max_iterations=5)

    circuit = CircuitData(
        num_qubits=2,
        gates=[GateOperation("X", (0,)), GateOperation("X", (0,))],
    )

    res = opt.optimize(circuit)
    d = res.to_dict()

    assert "initial_gate_count" in d
    assert "final_gate_count" in d
    assert "gate_reduction" in d
    assert d["gate_reduction"] == 2
    assert d["is_equivalent"] is True

    # Check circuit reconstructability (to Qiskit and back)
    qc = res.final_circuit.to_qiskit()
    recovered = CircuitData.from_qiskit(qc)
    assert recovered.num_qubits == res.final_circuit.num_qubits
