"""Tests for Phase 7: Procedural Candidate Generator.

Verifies:
1. Multi-candidate generation: circuit with multiple applicable transformations generates all proposals.
2. Distinctness: generated candidates are distinct where expected.
3. Immutability: original/parent circuit is never mutated.
4. Traceability: candidate metadata fields (ID, parent ID, rule name, location, preliminary cost, seed) are complete.
5. Determinism: identical input and seed yield identical candidate IDs and circuits.
"""

import copy
import numpy as np
import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.transformations.candidate_generator import CandidateCircuit, CandidateGenerator


def test_candidate_generation_multiple_rules():
    """Verify that multiple applicable transformations across a circuit are detected and generated."""
    # Circuit with:
    # 1. H H on qubit 0 (SelfInverseCancellation)
    # 2. CX(0, 1) CX(0, 1) (CNOTCancellation)
    # 3. RZ(0.2) RZ(0.3) on qubit 1 (RotationCombination)
    # 4. RX(0.0) on qubit 0 (ZeroRotationRemoval)
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),
            GateOperation("RZ", (1,), (0.2,)),
            GateOperation("RZ", (1,), (0.3,)),
            GateOperation("RX", (0,), (0.0,)),
        ],
    )

    original_gates_snapshot = copy.deepcopy(circuit.gates)
    generator = CandidateGenerator(seed=42)
    candidates = generator.generate_candidates(circuit)

    # We expect at least 4 distinct candidate transformations
    assert len(candidates) >= 4
    rule_names = {c.transformation_name for c in candidates}
    assert "SelfInverseCancellation" in rule_names
    assert "CNOTCancellation" in rule_names
    assert "RotationCombination" in rule_names
    assert "ZeroRotationRemoval" in rule_names

    # Check that original circuit was NOT mutated
    assert circuit.gates == original_gates_snapshot


def test_candidate_metadata_traceability():
    """Verify that candidate metadata contains required fields and correct links."""
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("X", (0,)),
            GateOperation("X", (0,)),
        ],
    )
    generator = CandidateGenerator(seed=123)
    candidates = generator.generate_candidates(circuit, seed=123)

    assert len(candidates) == 1
    cand = candidates[0]

    assert cand.parent_circuit_id == circuit.circuit_id
    assert cand.transformation_name == "SelfInverseCancellation"
    assert cand.transformation_location == (0, 1)
    assert cand.seed == 123
    assert cand.verified is False
    assert cand.verification_status == "unverified"
    assert cand.preliminary_cost.total_gates == 0

    # Serialization roundtrip for CandidateCircuit
    cand_dict = cand.to_dict()
    reconstructed = CandidateCircuit.from_dict(cand_dict)
    assert reconstructed.candidate_id == cand.candidate_id
    assert reconstructed.circuit == cand.circuit


def test_candidate_determinism():
    """Verify that candidate generation is completely deterministic."""
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),
        ],
    )
    gen = CandidateGenerator(seed=42)
    cands1 = gen.generate_candidates(circuit)
    cands2 = gen.generate_candidates(circuit)

    assert len(cands1) == len(cands2)
    for c1, c2 in zip(cands1, cands2):
        assert c1.candidate_id == c2.candidate_id
        assert c1.transformation_name == c2.transformation_name
        assert c1.circuit == c2.circuit
