"""Tests for Phase 5: Exact Quantum Equivalence Verification.

Verifies:
1. Known equivalent pairs:
   - Self-inverse pairs (X X, H H, CX CX == Identity)
   - Angle combinations (RX(a) RX(b) == RX(a+b))
2. Known non-equivalent pairs:
   - Different gate types (X vs Z, H vs X)
   - Different qubits / targets
   - Different rotation angles
3. Structurally different but behaviorally equivalent pairs:
   - H Z H == X
   - H X H == Z
   - CX(0,1) with H on both qubits == CX(1,0) (CNOT direction reversal)
4. Global phase handling:
   - Global phase vs no global phase flags
5. Numerical tolerance boundary cases:
   - Perturbations below tolerance pass, perturbations above tolerance fail
6. Qubit mismatch handling:
   - Detects mismatching qubit counts safely
"""

import numpy as np
import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.verification.equivalence import check_equivalence, check_unitary_equivalence
from qco.verification.simulator import compute_unitary, simulate_statevector


def test_known_equivalent_self_inverses():
    """Verify that self-inverse operations evaluate as equivalent to empty / identity."""
    # X X on qubit 0 vs empty circuit
    c_xx = CircuitData(num_qubits=2, gates=[GateOperation("X", (0,)), GateOperation("X", (0,))])
    c_empty = CircuitData(num_qubits=2, gates=[])

    res = check_equivalence(c_xx, c_empty)
    assert res.equivalent is True
    assert res.matrix_distance < 1e-7

    # H H on qubit 1 vs empty circuit
    c_hh = CircuitData(num_qubits=2, gates=[GateOperation("H", (1,)), GateOperation("H", (1,))])
    res_hh = check_equivalence(c_hh, c_empty)
    assert res_hh.equivalent is True

    # CX(0, 1) CX(0, 1) vs empty circuit
    c_cxcx = CircuitData(num_qubits=2, gates=[GateOperation("CX", (0, 1)), GateOperation("CX", (0, 1))])
    res_cx = check_equivalence(c_cxcx, c_empty)
    assert res_cx.equivalent is True


def test_known_non_equivalent_pairs():
    """Verify non-equivalent circuits are correctly rejected."""
    c_x = CircuitData(num_qubits=1, gates=[GateOperation("X", (0,))])
    c_z = CircuitData(num_qubits=1, gates=[GateOperation("Z", (0,))])
    res = check_equivalence(c_x, c_z)
    assert res.equivalent is False
    assert res.matrix_distance >= 1.0

    # Different angles
    c_rx1 = CircuitData(num_qubits=1, gates=[GateOperation("RX", (0,), (0.2,))])
    c_rx2 = CircuitData(num_qubits=1, gates=[GateOperation("RX", (0,), (0.5,))])
    assert check_equivalence(c_rx1, c_rx2).equivalent is False

    # Different CNOT control/target
    c_cx01 = CircuitData(num_qubits=2, gates=[GateOperation("CX", (0, 1))])
    c_cx10 = CircuitData(num_qubits=2, gates=[GateOperation("CX", (1, 0))])
    assert check_equivalence(c_cx01, c_cx10).equivalent is False


def test_structurally_different_equivalent_pairs():
    """Verify structurally different identities that evaluate to the same unitary."""
    # Identity 1: H Z H == X
    c_hzh = CircuitData(
        num_qubits=1,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("Z", (0,)),
            GateOperation("H", (0,)),
        ],
    )
    c_x = CircuitData(num_qubits=1, gates=[GateOperation("X", (0,))])
    res1 = check_equivalence(c_hzh, c_x)
    assert res1.equivalent is True

    # Identity 2: H X H == Z
    c_hxh = CircuitData(
        num_qubits=1,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("X", (0,)),
            GateOperation("H", (0,)),
        ],
    )
    c_z = CircuitData(num_qubits=1, gates=[GateOperation("Z", (0,))])
    res2 = check_equivalence(c_hxh, c_z)
    assert res2.equivalent is True

    # Identity 3: CNOT reversal identity
    # H(0) H(1) CX(0, 1) H(0) H(1) == CX(1, 0)
    c_cnot_rev = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("H", (1,)),
            GateOperation("CX", (0, 1)),
            GateOperation("H", (0,)),
            GateOperation("H", (1,)),
        ],
    )
    c_target_cx = CircuitData(num_qubits=2, gates=[GateOperation("CX", (1, 0))])
    res3 = check_equivalence(c_cnot_rev, c_target_cx)
    assert res3.equivalent is True


def test_parameterized_rotation_combination():
    """Verify RX(a) RX(b) == RX(a+b)."""
    a = 0.456
    b = 0.789
    c_seq = CircuitData(
        num_qubits=1,
        gates=[
            GateOperation("RX", (0,), (a,)),
            GateOperation("RX", (0,), (b,)),
        ],
    )
    c_comb = CircuitData(
        num_qubits=1,
        gates=[
            GateOperation("RX", (0,), (a + b,)),
        ],
    )
    res = check_equivalence(c_seq, c_comb)
    assert res.equivalent is True


def test_global_phase_handling():
    """Verify handling of global phase differences."""
    # Circuit 1: X (unitary has det -1)
    # Circuit 2: RX(pi) = -i X -> differs by global phase e^(-i pi/2) = -i
    c1 = CircuitData(num_qubits=1, gates=[GateOperation("X", (0,))])
    c2 = CircuitData(num_qubits=1, gates=[GateOperation("RX", (0,), (np.pi,))])

    # Up to global phase: EQUIVALENT
    res_with_phase = check_equivalence(c1, c2, ignore_global_phase=True)
    assert res_with_phase.equivalent is True
    assert res_with_phase.phase_difference is not None
    assert np.isclose(abs(res_with_phase.phase_difference), np.pi / 2, atol=1e-5)

    # Exact without global phase allowance: NOT EQUIVALENT
    res_without_phase = check_equivalence(c1, c2, ignore_global_phase=False)
    assert res_without_phase.equivalent is False


def test_numerical_tolerance_boundaries():
    """Verify precision around tolerance threshold."""
    tol = 1e-4
    c_base = CircuitData(num_qubits=1, gates=[GateOperation("RX", (0,), (1.0,))])

    # Below tolerance perturbation
    c_close = CircuitData(num_qubits=1, gates=[GateOperation("RX", (0,), (1.0 + 1e-5,))])
    assert check_equivalence(c_base, c_close, tolerance=tol).equivalent is True

    # Above tolerance perturbation
    c_far = CircuitData(num_qubits=1, gates=[GateOperation("RX", (0,), (1.0 + 1e-2,))])
    assert check_equivalence(c_base, c_far, tolerance=tol).equivalent is False


def test_qubit_count_mismatch():
    """Verify circuits with different qubit counts are immediately reported non-equivalent."""
    c1 = CircuitData(num_qubits=2, gates=[GateOperation("X", (0,))])
    c2 = CircuitData(num_qubits=3, gates=[GateOperation("X", (0,))])
    res = check_equivalence(c1, c2)
    assert res.equivalent is False
    assert "Qubit counts do not match" in res.reason


def test_statevector_simulation():
    """Verify statevector simulation matches theoretical expectations."""
    # Bell state circuit
    c = CircuitData(
        num_qubits=2,
        gates=[
            GateOperation("H", (0,)),
            GateOperation("CX", (0, 1)),
        ],
    )
    sv = simulate_statevector(c)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    assert np.allclose(sv, expected, atol=1e-7)
