"""Tests for Phase 2: Canonical Quantum Circuit Representation.

Tests defined in Phase 2 specification:
- Test A: Round trip (Circuit -> serialize -> deserialize -> reconstruct)
- Test B: Parameter preservation (nontrivial RX/RY/RZ angles)
- Test C: Qubit preservation (different control-target assignments)
- Test D: Behavioral preservation (simulate original and reconstructed circuits)
"""

import json
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qco.circuits.representation import CircuitData, GateOperation, SUPPORTED_GATES
from qco.circuits.serialization import (
    deserialize_from_json,
    load_circuit,
    save_circuit,
    serialize_to_json,
)


def test_supported_gates_vocab():
    """Verify supported gate vocabulary covers required gates."""
    required = {"I", "X", "Y", "Z", "H", "RX", "RY", "RZ", "CX"}
    assert required.issubset(SUPPORTED_GATES.keys())


def test_gate_validation_and_canonicalization():
    """Verify GateOperation input validation and naming."""
    # Canonicalize CNOT to CX
    g = GateOperation("cnot", (0, 1))
    assert g.name == "CX"
    assert g.qubits == (0, 1)

    # Invalid gate
    with pytest.raises(ValueError, match="Unsupported gate"):
        GateOperation("INVALID", (0,))

    # Invalid qubit count
    with pytest.raises(ValueError, match="expects 1 qubits"):
        GateOperation("H", (0, 1))

    # CX with identical control and target
    with pytest.raises(ValueError, match="cannot be identical"):
        GateOperation("CX", (1, 1))

    # Missing parameters for rotation gate
    with pytest.raises(ValueError, match="expects 1 parameters"):
        GateOperation("RX", (0,))


def test_circuit_qubit_bounds_validation():
    """Verify circuit rejects gates operating on invalid qubit indices."""
    with pytest.raises(ValueError, match="outside valid range"):
        CircuitData(num_qubits=2, gates=[GateOperation("H", (2,))])

    with pytest.raises(ValueError, match="at least 1 qubit"):
        CircuitData(num_qubits=0)


def test_test_a_round_trip():
    """Phase 2 Test A: Circuit -> serialize -> deserialize -> reconstruct."""
    original = CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("RX", (2,), (np.pi / 3,)),
            GateOperation("CX", (1, 2)),
            GateOperation("Z", (1,)),
        ],
        metadata={"author": "DL Project", "phase": 2}
    )

    # Serialize
    json_str = serialize_to_json(original)

    # Deserialize
    deserialized = deserialize_from_json(json_str)

    # Reconstruct Qiskit circuit
    reconstructed_qc = deserialized.to_qiskit()
    reconstructed_data = CircuitData.from_qiskit(reconstructed_qc)

    # Assert structural equality
    assert original == deserialized
    assert len(original) == len(deserialized)
    assert original.circuit_id == deserialized.circuit_id
    assert original.circuit_id == reconstructed_data.circuit_id
    assert original.metadata["author"] == deserialized.metadata["author"]


def test_test_b_parameter_preservation():
    """Phase 2 Test B: Parameter preservation with nontrivial angles."""
    angles = [0.12345678, np.pi / 7, 2.71828182]
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("RX", (0,), (angles[0],)),
            GateOperation("RY", (1,), (angles[1],)),
            GateOperation("RZ", (0,), (angles[2],)),
        ]
    )

    qc = circuit.to_qiskit()
    from_q = CircuitData.from_qiskit(qc)

    assert len(from_q) == 3
    assert np.isclose(from_q[0].params[0], angles[0], atol=1e-7)
    assert np.isclose(from_q[1].params[0], angles[1], atol=1e-7)
    assert np.isclose(from_q[2].params[0], angles[2], atol=1e-7)


def test_test_c_qubit_preservation():
    """Phase 2 Test C: Qubit preservation with various control-target assignments."""
    circuit = CircuitData(
        num_qubits=4,
        gates=[
            GateOperation("CX", (0, 3)),
            GateOperation("CX", (3, 0)),
            GateOperation("CX", (1, 2)),
            GateOperation("CX", (2, 1)),
        ]
    )

    serialized = serialize_to_json(circuit)
    deserialized = deserialize_from_json(serialized)
    qc = deserialized.to_qiskit()
    reconstructed = CircuitData.from_qiskit(qc)

    assert reconstructed[0].qubits == (0, 3)
    assert reconstructed[1].qubits == (3, 0)
    assert reconstructed[2].qubits == (1, 2)
    assert reconstructed[3].qubits == (2, 1)


def test_test_d_behavioral_preservation():
    """Phase 2 Test D: Behavioral preservation via unitary equivalence."""
    circuit = CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("RY", (2,), (0.45,)),
            GateOperation("CX", (1, 2)),
            GateOperation("RZ", (0,), (1.23,)),
            GateOperation("X", (2,)),
        ]
    )

    # 1. Round trip through JSON
    json_str = serialize_to_json(circuit)
    deserialized = deserialize_from_json(json_str)

    # 2. Convert both to Qiskit and compare exact Unitary Operators
    u_orig = Operator(circuit.to_qiskit()).data
    u_recon = Operator(deserialized.to_qiskit()).data

    # Check that unitary matrices match within numerical tolerance (1e-7)
    assert np.allclose(u_orig, u_recon, atol=1e-7)


def test_file_save_and_load(tmp_path):
    """Verify saving to and loading from a JSON file on disk."""
    circuit = CircuitData(
        num_qubits=2,
        gates=[GateOperation("H", (0,)), GateOperation("CX", (0, 1))]
    )
    file_path = tmp_path / "test_circuit.json"
    save_circuit(circuit, file_path)

    loaded = load_circuit(file_path)
    assert circuit == loaded
    assert circuit.circuit_id == loaded.circuit_id
