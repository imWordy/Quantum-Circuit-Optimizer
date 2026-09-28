"""Tests for Phase 7: Procedural Candidate Generator.

Verifies:
- All intended candidates are generated from a multi-rule circuit.
- Candidates are distinct where expected.
- Source circuit is NOT mutated.
- Metadata (parent_id, rule name, indices, description) is correct.
- Preliminary cost is consistent with the transformed circuit.
- Deterministic: same circuit produces same number of candidates consistently.
- CandidateGenerator works with specific rule subsets.
"""

import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.metrics.circuit_cost import compute_circuit_cost
from qco.transformations import (
    Candidate,
    CandidateGenerator,
    CNOTCancellation,
    RotationCombination,
    SelfInverseCancellation,
    ZeroRotationRemoval,
    IdentityRemoval,
    get_default_rules,
)


def _make_multi_rule_circuit() -> CircuitData:
    """Create a circuit with multiple applicable rules for testing."""
    # Contains:
    # - adjacent H H -> SelfInverseCancellation
    # - adjacent CX CX -> CNOTCancellation
    # - adjacent RX RX -> RotationCombination
    # - RY(0.0) -> ZeroRotationRemoval
    # - I gate -> IdentityRemoval
    return CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),         # cancels with above
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),       # cancels with above
            GateOperation("RX", (2,), (0.3,)),
            GateOperation("RX", (2,), (0.5,)), # combines with above
            GateOperation("RY", (1,), (0.0,)), # ZeroRotationRemoval
            GateOperation("I", (2,)),          # IdentityRemoval
        ],
    )


def test_all_candidates_generated():
    """Verify the expected number of candidates are generated across all rules."""
    circuit = _make_multi_rule_circuit()
    gen = CandidateGenerator(rules=get_default_rules())
    candidates = gen.generate_candidates(circuit)

    # Expected matches:
    # SelfInverseCancellation: 1 (H H at indices 0,1)
    # CNOTCancellation: 1 (CX CX at indices 2,3)
    # RotationCombination: 1 (RX RX at indices 4,5)
    # ZeroRotationRemoval: 1 (RY(0) at index 6)
    # IdentityRemoval: 1 (I at index 7)
    assert len(candidates) == 5


def test_candidates_have_correct_metadata():
    """Verify each candidate has correct metadata referencing parent and rule."""
    circuit = _make_multi_rule_circuit()
    gen = CandidateGenerator(rules=get_default_rules())
    candidates = gen.generate_candidates(circuit)

    for cand in candidates:
        assert cand.parent_circuit_id == circuit.circuit_id
        assert cand.candidate_id is not None
        assert len(cand.candidate_id) > 0
        assert cand.transformation_name in {
            "SelfInverseCancellation",
            "CNOTCancellation",
            "RotationCombination",
            "ZeroRotationRemoval",
            "IdentityRemoval",
        }
        assert isinstance(cand.transformation_location, str)
        assert len(cand.transformation_location) > 0
        assert isinstance(cand.transformation_indices, list)
        assert cand.verification_status == "unverified"


def test_preliminary_cost_consistency():
    """Verify preliminary cost stored in Candidate matches recomputed cost."""
    circuit = _make_multi_rule_circuit()
    gen = CandidateGenerator(rules=get_default_rules())
    candidates = gen.generate_candidates(circuit)

    for cand in candidates:
        recomputed = compute_circuit_cost(cand.circuit)
        assert cand.preliminary_cost.total_gates == recomputed.total_gates
        assert cand.preliminary_cost.cnot_count == recomputed.cnot_count
        assert cand.preliminary_cost.depth == recomputed.depth


def test_source_circuit_not_mutated():
    """Verify that the original circuit is not mutated after candidate generation."""
    circuit = _make_multi_rule_circuit()
    original_gates = circuit.gates[:]
    original_id = circuit.circuit_id
    original_qubit_count = circuit.num_qubits

    gen = CandidateGenerator(rules=get_default_rules())
    _ = gen.generate_candidates(circuit)

    assert circuit.gates == original_gates
    assert circuit.circuit_id == original_id
    assert circuit.num_qubits == original_qubit_count


def test_candidates_are_distinct():
    """Verify all generated candidates have distinct circuit IDs."""
    circuit = _make_multi_rule_circuit()
    gen = CandidateGenerator(rules=get_default_rules())
    candidates = gen.generate_candidates(circuit)

    circuit_ids = [c.circuit.circuit_id for c in candidates]
    assert len(circuit_ids) == len(set(circuit_ids)), (
        f"Duplicate circuit IDs found: {circuit_ids}"
    )


def test_deterministic_generation():
    """Verify same circuit always produces same number of candidates."""
    circuit = _make_multi_rule_circuit()
    gen = CandidateGenerator(rules=get_default_rules())

    candidates1 = gen.generate_candidates(circuit)
    candidates2 = gen.generate_candidates(circuit)

    assert len(candidates1) == len(candidates2)


def test_rule_subset():
    """Verify CandidateGenerator works correctly with a subset of rules."""
    circuit = _make_multi_rule_circuit()

    # Only SelfInverse and CNOT cancellation
    gen_sub = CandidateGenerator(rules=[SelfInverseCancellation(), CNOTCancellation()])
    candidates = gen_sub.generate_candidates(circuit)
    assert len(candidates) == 2

    names = {c.transformation_name for c in candidates}
    assert names == {"SelfInverseCancellation", "CNOTCancellation"}


def test_no_candidates_on_trivial_circuit():
    """Verify an irreducible circuit produces zero candidates."""
    # H then CX: no rule applies
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
        ],
    )
    gen = CandidateGenerator(rules=get_default_rules())
    candidates = gen.generate_candidates(circuit)
    assert len(candidates) == 0


def test_batch_generation():
    """Verify batch generation returns candidates keyed by circuit_id."""
    c1 = _make_multi_rule_circuit()
    c2 = CircuitData(
        num_qubits=2,
        gates=[GateOperation("X", (0,)), GateOperation("X", (0,))],
    )

    gen = CandidateGenerator(rules=get_default_rules())
    result = gen.generate_candidates_from_batch([c1, c2])

    assert c1.circuit_id in result
    assert c2.circuit_id in result
    assert len(result[c1.circuit_id]) == 5
    # X X -> SelfInverseCancellation
    assert len(result[c2.circuit_id]) == 1
