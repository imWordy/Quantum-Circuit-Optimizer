"""Tests for Phase 4: Circuit Cost Metrics.

Tests hand-verifiable circuits against computed values:
- Small circuits
- Single-qubit gates
- Multiple CNOTs
- Parallel gates (verifying depth calculation)
- Sequential gates (verifying depth calculation)
- Parameterized gates
- Reduction metrics (absolute, percentage, division by zero)
- Agreement with Qiskit's native circuit depth
"""

import numpy as np
import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.metrics.circuit_cost import (
    CircuitCost,
    CostReduction,
    calculate_depth,
    compute_circuit_cost,
    compute_cost_reduction,
)


def test_empty_circuit():
    """Verify metrics for an empty circuit."""
    c = CircuitData(num_qubits=2, gates=[])
    cost = compute_circuit_cost(c)
    assert cost.total_gates == 0
    assert cost.single_qubit_gates == 0
    assert cost.two_qubit_gates == 0
    assert cost.cnot_count == 0
    assert cost.depth == 0


def test_single_qubit_gates_sequential():
    """Verify sequential single-qubit gates on the same qubit."""
    # 3 sequential gates on qubit 0 -> depth = 3
    c = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("X", (0,)),
            GateOperation("Z", (0,)),
        ],
    )
    cost = compute_circuit_cost(c)
    assert cost.total_gates == 3
    assert cost.single_qubit_gates == 3
    assert cost.two_qubit_gates == 0
    assert cost.cnot_count == 0
    assert cost.depth == 3
    assert c.to_qiskit().depth() == cost.depth


def test_parallel_gates_depth():
    """Verify parallel gates on disjoint qubits execute in depth 1."""
    # 3 parallel gates on qubits 0, 1, 2 -> depth = 1
    c = CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("X", (1,)),
            GateOperation("Y", (2,)),
        ],
    )
    cost = compute_circuit_cost(c)
    assert cost.total_gates == 3
    assert cost.single_qubit_gates == 3
    assert cost.depth == 1
    assert c.to_qiskit().depth() == 1


def test_cnot_and_two_qubit_gates():
    """Verify multiple CNOTs and depth calculation with entangling gates."""
    # Q0: --H--.--
    #          |
    # Q1: -----X--.--
    #             |
    # Q2: --------X--
    # Total gates: 3 (1 single-qubit, 2 CNOTs)
    # Expected depth: 3
    c = CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (1, 2)),
        ],
    )
    cost = compute_circuit_cost(c)
    assert cost.total_gates == 3
    assert cost.single_qubit_gates == 1
    assert cost.two_qubit_gates == 2
    assert cost.cnot_count == 2
    assert cost.depth == 3
    assert c.to_qiskit().depth() == 3


def test_parallel_cnots():
    """Verify two CNOTs on disjoint qubit pairs execute in parallel."""
    # CX(0, 1) and CX(2, 3) -> depth = 1
    c = CircuitData(
        num_qubits=4,
        gates=[
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (2, 3)),
        ],
    )
    cost = compute_circuit_cost(c)
    assert cost.total_gates == 2
    assert cost.two_qubit_gates == 2
    assert cost.cnot_count == 2
    assert cost.depth == 1
    assert c.to_qiskit().depth() == 1


def test_parameterized_gates_cost():
    """Verify rotation gates counted as single-qubit gates."""
    c = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("RX", (0,), (0.5,)),
            GateOperation("RY", (1,), (1.2,)),
            GateOperation("RZ", (0,), (2.1,)),
            GateOperation("CX", (0, 1)),
        ],
    )
    cost = compute_circuit_cost(c)
    assert cost.total_gates == 4
    assert cost.single_qubit_gates == 3
    assert cost.two_qubit_gates == 1
    assert cost.cnot_count == 1
    # Q0 has RX, RZ, CX (depth 3)
    # Q1 has RY, CX (depth 2)
    # Overall depth = 3
    assert cost.depth == 3
    assert c.to_qiskit().depth() == 3


def test_reduction_metrics():
    """Verify absolute and percentage reduction calculations including zero denominator."""
    orig = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),
            GateOperation("H", (0,)),
        ],
    )
    cand = CircuitData(
        num_qubits=2,
        gates=[],
    )

    red = compute_cost_reduction(orig, cand)
    assert red.gate_reduction == 4
    assert red.cnot_reduction == 2
    assert red.gate_reduction_pct == 100.0
    assert red.cnot_reduction_pct == 100.0

    # Test reverse (cost increase)
    red_reverse = compute_cost_reduction(cand, orig)
    assert red_reverse.gate_reduction == -4
    assert red_reverse.cnot_reduction == -2
    assert red_reverse.gate_reduction_pct == -100.0  # safe zero-denominator handling

    # Zero vs Zero
    red_zeros = compute_cost_reduction(cand, cand)
    assert red_zeros.gate_reduction == 0
    assert red_zeros.gate_reduction_pct == 0.0
