"""Circuit Graph Representation for QCO — Phase 10.

Converts canonical CircuitData into PyTorch Geometric (PyG) Data objects.

Design:
- Nodes: Each gate is a node.
- Node Features (x):
    - One-hot encoding of gate type from vocabulary (9 dims: I, X, Y, Z, H, RX, RY, RZ, CX)
    - Number of participating qubits (1 dim)
    - Normalized gate parameter / rotation angle in (-pi, pi] (1 dim)
    - Layer / Depth position of the gate in the circuit DAG (1 dim)
    - Primary qubit index (1 dim)
    - Secondary/target qubit index (-1 if single-qubit) (1 dim)
    Total node feature dim = 9 + 1 + 1 + 1 + 1 + 1 = 14 dims.

- Edges:
    - Directed temporal dependency edges along qubits (operation A -> operation B on qubit q).
    - Edge Attributes (edge_attr):
        - Shared qubit index (1 dim)
        - Edge type: sequential qubit wire (type 0) or multi-qubit entangling wire (type 1)
          (one-hot 2 dims).
        Total edge feature dim = 3 dims.

Guarantees:
- Circuits with identical aggregate statistics but different sequential order
  (e.g., H -> CX -> RX vs RX -> CX -> H) produce distinct graph structures/features.
- Source circuit is never mutated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from qco.circuits.representation import CircuitData, GateOperation, SUPPORTED_GATES
from qco.transformations.rules import normalize_angle

# Ordered vocabulary for deterministic one-hot gate encoding
GATE_VOCAB: List[str] = sorted(list(SUPPORTED_GATES.keys()))
GATE_TO_IDX: Dict[str, int] = {gate: i for i, gate in enumerate(GATE_VOCAB)}
VOCAB_SIZE: int = len(GATE_VOCAB)


def compute_gate_layers(circuit: CircuitData) -> List[int]:
    """Compute per-gate DAG layer position (depth).

    Returns:
        List of layer indices (0-indexed) for each gate in circuit.gates.
    """
    qubit_depths = [0] * circuit.num_qubits
    gate_layers: List[int] = []

    for gate in circuit.gates:
        layer = max(qubit_depths[q] for q in gate.qubits)
        gate_layers.append(layer)
        for q in gate.qubits:
            qubit_depths[q] = layer + 1

    return gate_layers


def circuit_to_graph(circuit: CircuitData) -> Data:
    """Convert a canonical CircuitData instance into a PyTorch Geometric Data graph.

    Args:
        circuit: Source CircuitData to represent.

    Returns:
        torch_geometric.data.Data object with:
            - x: Node feature tensor [num_nodes, num_node_features]
            - edge_index: Graph connectivity tensor [2, num_edges]
            - edge_attr: Edge feature tensor [num_edges, num_edge_features]
            - num_qubits: Circuit qubit count
            - circuit_id: Deterministic circuit ID
    """
    num_nodes = len(circuit.gates)

    # Empty circuit handling
    if num_nodes == 0:
        return Data(
            x=torch.zeros((0, VOCAB_SIZE + 5), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 3), dtype=torch.float),
            num_nodes=0,
            num_qubits=circuit.num_qubits,
            circuit_id=circuit.circuit_id,
        )

    gate_layers = compute_gate_layers(circuit)

    # 1. Build Node Features
    node_features: List[List[float]] = []

    for i, gate in enumerate(circuit.gates):
        # One-hot gate type (VOCAB_SIZE dims)
        one_hot_type = [0.0] * VOCAB_SIZE
        if gate.name in GATE_TO_IDX:
            one_hot_type[GATE_TO_IDX[gate.name]] = 1.0

        # Number of qubits acted upon
        num_qubits_acted = float(len(gate.qubits))

        # Normalized parameter (rotation angle)
        param_val = normalize_angle(gate.params[0]) if gate.params else 0.0

        # Layer position
        layer = float(gate_layers[i])

        # Qubit assignments
        primary_qubit = float(gate.qubits[0])
        secondary_qubit = float(gate.qubits[1]) if len(gate.qubits) > 1 else -1.0

        feat = (
            one_hot_type
            + [num_qubits_acted, param_val, layer, primary_qubit, secondary_qubit]
        )
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float)

    # 2. Build Dependency Edges
    # Track the last gate index that acted on each qubit
    last_gate_on_qubit: List[Optional[int]] = [None] * circuit.num_qubits
    edge_sources: List[int] = []
    edge_targets: List[int] = []
    edge_attributes: List[List[float]] = []

    for gate_idx, gate in enumerate(circuit.gates):
        for q in gate.qubits:
            prev_idx = last_gate_on_qubit[q]
            if prev_idx is not None and prev_idx != gate_idx:
                edge_sources.append(prev_idx)
                edge_targets.append(gate_idx)

                # Edge features: [shared_qubit, is_single_wire, is_two_qubit_wire]
                is_two_qubit_wire = 1.0 if (len(circuit.gates[prev_idx].qubits) > 1 or len(gate.qubits) > 1) else 0.0
                is_single_wire = 1.0 - is_two_qubit_wire
                edge_attributes.append([float(q), is_single_wire, is_two_qubit_wire])

            last_gate_on_qubit[q] = gate_idx

    if edge_sources:
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        num_qubits=circuit.num_qubits,
        circuit_id=circuit.circuit_id,
    )

    return data
