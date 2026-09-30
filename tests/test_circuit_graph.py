"""Tests for Phase 10: Circuit Graph Representation.

Tests critical requirements:
1. Distinguish circuits with identical aggregate statistics but different structural order:
   - Circuit 1: H -> CX -> RX
   - Circuit 2: RX -> CX -> H
2. Correct node feature shapes and values:
   - Gate type one-hot encoding
   - Layer / depth indexing
   - Parameter preservation
   - Qubit index preservation
3. Correct edge connectivity:
   - Sequential operations on the same qubit have directed dependency edges
   - Two-qubit gates preserve connections along both qubits
4. Circuit immutability: original CircuitData is not mutated.
5. Empty circuit handling.
"""

import numpy as np
import pytest
import torch

from qco.circuits.representation import CircuitData, GateOperation
from qco.graphs.circuit_graph import (
    GATE_TO_IDX,
    GATE_VOCAB,
    VOCAB_SIZE,
    circuit_to_graph,
    compute_gate_layers,
)
from qco.metrics.circuit_cost import compute_circuit_cost


def test_critical_requirement_structural_distinguishability():
    """Phase 10 Critical Requirement:

    Circuits with IDENTICAL aggregate statistics (1 H, 1 CX, 1 RX, gate count=3, depth=3)
    must produce DISTINCT graph representations when their order differs:
        Circuit A: H -> CX -> RX
        Circuit B: RX -> CX -> H
    """
    param = 0.785
    # Circuit A: H on q0, CX(0,1), RX on q0
    c_a = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("RX", (0,), (param,)),
        ],
    )

    # Circuit B: RX on q0, CX(0,1), H on q0
    c_b = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("RX", (0,), (param,)),
            GateOperation("CX", (0, 1)),
            GateOperation("H", (0,)),
        ],
    )

    # 1. Verify aggregate statistics are strictly identical
    cost_a = compute_circuit_cost(c_a)
    cost_b = compute_circuit_cost(c_b)
    assert cost_a == cost_b
    assert cost_a.total_gates == 3
    assert cost_a.cnot_count == 1
    assert cost_a.depth == 3

    # 2. Convert to graphs
    g_a = circuit_to_graph(c_a)
    g_b = circuit_to_graph(c_b)

    # 3. Assert graph representations are clearly different
    assert not torch.equal(g_a.x, g_b.x), "Node features must differ due to reversed gate types and layers"

    # Node 0 in A is H; Node 0 in B is RX
    h_idx = GATE_TO_IDX["H"]
    rx_idx = GATE_TO_IDX["RX"]
    assert g_a.x[0, h_idx] == 1.0
    assert g_b.x[0, rx_idx] == 1.0


def test_node_features_correctness():
    """Verify detailed contents of the node feature tensor."""
    theta = 1.25
    c = CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("RX", (1,), (theta,)),
            GateOperation("CX", (1, 2)),
            GateOperation("Z", (0,)),
        ],
    )

    g = circuit_to_graph(c)
    assert g.num_nodes == 3
    assert g.x.shape == (3, VOCAB_SIZE + 5)

    # Node 0: RX on q1
    assert g.x[0, GATE_TO_IDX["RX"]] == 1.0
    assert g.x[0, VOCAB_SIZE] == 1.0  # num_qubits acted
    assert np.isclose(g.x[0, VOCAB_SIZE + 1].item(), theta, atol=1e-5)  # param
    assert g.x[0, VOCAB_SIZE + 2].item() == 0.0  # layer 0
    assert g.x[0, VOCAB_SIZE + 3].item() == 1.0  # primary qubit
    assert g.x[0, VOCAB_SIZE + 4].item() == -1.0  # secondary qubit (none)

    # Node 1: CX(1, 2)
    assert g.x[1, GATE_TO_IDX["CX"]] == 1.0
    assert g.x[1, VOCAB_SIZE] == 2.0  # num_qubits acted
    assert g.x[1, VOCAB_SIZE + 1].item() == 0.0  # param
    assert g.x[1, VOCAB_SIZE + 2].item() == 1.0  # layer 1
    assert g.x[1, VOCAB_SIZE + 3].item() == 1.0  # control
    assert g.x[1, VOCAB_SIZE + 4].item() == 2.0  # target


def test_edge_dependencies_and_entangling_connectivity():
    """Verify directed dependencies along sequential qubit wires and two-qubit gates."""
    # Circuit:
    # Q0: --H (node 0) -------- CX (node 2) --
    # Q1: -------- X (node 1) - CX (node 2) --
    c = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("X", (1,)),
            GateOperation("CX", (0, 1)),
        ],
    )

    g = circuit_to_graph(c)
    # Node 2 (CX) should have edges incoming from both Node 0 (along q0) and Node 1 (along q1)
    edge_tuples = list(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    assert (0, 2) in edge_tuples, "Edge 0 -> 2 along qubit 0 missing"
    assert (1, 2) in edge_tuples, "Edge 1 -> 2 along qubit 1 missing"
    assert len(edge_tuples) == 2


def test_circuit_immutability():
    """Verify original CircuitData is not modified during graph construction."""
    c = CircuitData(
        num_qubits=2,
        gates=[GateOperation("H", (0,)), GateOperation("X", (1,))],
    )
    gates_backup = list(c.gates)
    cid_backup = c.circuit_id

    _ = circuit_to_graph(c)

    assert c.gates == gates_backup
    assert c.circuit_id == cid_backup


def test_empty_circuit_graph():
    """Verify empty circuit conversion produces valid 0-node PyG Data."""
    c = CircuitData(num_qubits=2, gates=[])
    g = circuit_to_graph(c)

    assert g.num_nodes == 0
    assert g.x.shape[0] == 0
    assert g.edge_index.shape == (2, 0)
    assert g.edge_attr.shape == (0, 3)
    assert g.num_qubits == 2
    assert g.circuit_id == c.circuit_id
