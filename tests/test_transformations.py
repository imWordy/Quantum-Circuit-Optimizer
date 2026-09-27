"""Tests for Phase 6: Valid Transformation Library.

Tests every rule:
1. Construct a valid example.
2. Apply the rule.
3. Check circuit validity.
4. Check exact equivalence with Phase 5 verifier.
5. Check expected cost change with Phase 4 metrics.
6. Test negative cases where the rule must not apply.
"""

import numpy as np
import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.metrics.circuit_cost import compute_cost_reduction
from qco.transformations.rules import (
    CNOTCancellation,
    IdentityRemoval,
    RotationCombination,
    SelfInverseCancellation,
    ZeroRotationRemoval,
    normalize_angle,
)
from qco.transformations.validator import RuleValidator
from qco.verification.equivalence import check_equivalence


def test_angle_normalization():
    """Verify angle normalization wraps cleanly to (-pi, pi]."""
    assert np.isclose(normalize_angle(0.0), 0.0)
    assert np.isclose(normalize_angle(2 * np.pi), 0.0)
    assert np.isclose(normalize_angle(np.pi), np.pi)
    assert np.isclose(normalize_angle(-np.pi), np.pi)
    assert np.isclose(normalize_angle(3 * np.pi), np.pi)
    assert np.isclose(normalize_angle(np.pi / 2), np.pi / 2)


def test_self_inverse_cancellation():
    """Verify SelfInverseCancellation for X, Y, Z, H."""
    rule = SelfInverseCancellation()
    validator = RuleValidator()

    # 1. Valid example: circuit with adjacent H H on qubit 0 and non-adjacent gates
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("X", (1,)),
            GateOperation("H", (0,)),
            GateOperation("H", (0,)),
            GateOperation("Z", (1,)),
        ],
    )

    matches = rule.find_matches(circuit)
    assert len(matches) == 1
    assert matches[0].indices == (1, 2)
    assert matches[0].qubits == (0,)

    # 2. Apply rule
    transformed = rule.apply(circuit, matches[0])
    assert len(transformed) == 2

    # 3. Check validity & equivalence via RuleValidator
    res = validator.validate_transformation(rule, circuit, matches[0])
    assert res.valid is True
    assert res.actual_gate_reduction == 2
    assert res.actual_cnot_reduction == 0

    # 4. Negative case: non-adjacent H gates (interrupted by X on same qubit)
    c_negative = CircuitData(
        num_qubits=1,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("X", (0,)),
            GateOperation("H", (0,)),
        ],
    )
    assert len(rule.find_matches(c_negative)) == 0


def test_cnot_cancellation():
    """Verify CNOTCancellation for adjacent identical CNOTs."""
    rule = CNOTCancellation()
    validator = RuleValidator()

    # 1. Valid example
    circuit = CircuitData(
        num_qubits=3,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (0, 1)),
            GateOperation("Z", (2,)),
        ],
    )

    matches = rule.find_matches(circuit)
    assert len(matches) == 1
    assert matches[0].indices == (1, 2)

    # 2. Validate
    res = validator.validate_transformation(rule, circuit, matches[0])
    assert res.valid is True
    assert res.actual_gate_reduction == 2
    assert res.actual_cnot_reduction == 2

    # 3. Negative case: opposite control/target CX(0, 1) and CX(1, 0)
    c_opposite = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("CX", (0, 1)),
            GateOperation("CX", (1, 0)),
        ],
    )
    assert len(rule.find_matches(c_opposite)) == 0


def test_rotation_combination():
    """Verify RotationCombination for RX, RY, RZ."""
    rule = RotationCombination()
    validator = RuleValidator()

    # 1. General sum: RX(0.3) RX(0.5) -> RX(0.8)
    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("RX", (0,), (0.3,)),
            GateOperation("RX", (0,), (0.5,)),
            GateOperation("CX", (0, 1)),
        ],
    )

    matches = rule.find_matches(circuit)
    assert len(matches) == 1
    assert matches[0].indices == (0, 1)

    transformed = rule.apply(circuit, matches[0])
    assert len(transformed) == 2
    assert transformed.gates[0].name == "RX"
    assert np.isclose(transformed.gates[0].params[0], 0.8)

    res = validator.validate_transformation(rule, circuit, matches[0])
    assert res.valid is True
    assert res.actual_gate_reduction == 1

    # 2. Angle cancels to 0: RZ(pi/2) RZ(-pi/2) -> empty
    c_zero_sum = CircuitData(
        num_qubits=1,
        gates=[
            GateOperation("RZ", (0,), (np.pi / 2,)),
            GateOperation("RZ", (0,), (-np.pi / 2,)),
        ],
    )
    matches_zero = rule.find_matches(c_zero_sum)
    assert len(matches_zero) == 1
    assert matches_zero[0].expected_gate_reduction == 2

    res_zero = validator.validate_transformation(rule, c_zero_sum, matches_zero[0])
    assert res_zero.valid is True
    assert res_zero.actual_gate_reduction == 2

    # 3. Negative case: different rotation axes (RX and RZ)
    c_diff_axis = CircuitData(
        num_qubits=1,
        gates=[
            GateOperation("RX", (0,), (0.5,)),
            GateOperation("RZ", (0,), (0.5,)),
        ],
    )
    assert len(rule.find_matches(c_diff_axis)) == 0


def test_zero_rotation_removal():
    """Verify ZeroRotationRemoval eliminates R_axis(0)."""
    rule = ZeroRotationRemoval()
    validator = RuleValidator()

    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("RY", (1,), (0.0,)),
            GateOperation("CX", (0, 1)),
        ],
    )

    matches = rule.find_matches(circuit)
    assert len(matches) == 1
    assert matches[0].indices == (1,)

    res = validator.validate_transformation(rule, circuit, matches[0])
    assert res.valid is True
    assert res.actual_gate_reduction == 1

    # Negative case: non-zero rotation
    c_nonzero = CircuitData(num_qubits=1, gates=[GateOperation("RY", (0,), (0.2,))])
    assert len(rule.find_matches(c_nonzero)) == 0


def test_identity_removal():
    """Verify IdentityRemoval eliminates I gates."""
    rule = IdentityRemoval()
    validator = RuleValidator()

    circuit = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("I", (0,)),
            GateOperation("X", (1,)),
        ],
    )

    matches = rule.find_matches(circuit)
    assert len(matches) == 1
    assert matches[0].indices == (0,)

    res = validator.validate_transformation(rule, circuit, matches[0])
    assert res.valid is True
    assert res.actual_gate_reduction == 1
