"""Tests for Frontend Phase 6: Equivalence Verification Authority Hub."""

import numpy as np
import pytest

from qco.circuits.representation import CircuitData, GateOperation
from qco.frontend.components.unitary_viewer import (
    create_product_distance_heatmap,
    create_unitary_heatmap,
)
from qco.verification.equivalence import check_unitary_equivalence
from qco.verification.simulator import compute_unitary


@pytest.fixture
def sample_circuit() -> CircuitData:
    """Provide a 2-qubit Bell state circuit."""
    return CircuitData(
        circuit_id="test_bell_circ",
        num_qubits=2,
        gates=[
            GateOperation(name="h", qubits=[0], params=[]),
            GateOperation(name="cx", qubits=[0, 1], params=[]),
        ],
    )


def test_create_unitary_heatmap_real_imag(sample_circuit: CircuitData):
    """Test generating real and imaginary unitary heatmaps."""
    unitary = compute_unitary(sample_circuit)
    assert unitary.shape == (4, 4)

    fig = create_unitary_heatmap(unitary, title="Test Bell Unitary", mode="real_imag")
    assert fig is not None
    assert len(fig.data) == 2  # Real trace and Imaginary trace


def test_create_unitary_heatmap_magnitude_phase(sample_circuit: CircuitData):
    """Test generating magnitude and phase unitary heatmaps."""
    unitary = compute_unitary(sample_circuit)

    fig = create_unitary_heatmap(unitary, title="Test Bell Unitary", mode="magnitude_phase")
    assert fig is not None
    assert len(fig.data) == 2  # Mag and Phase traces


def test_create_product_distance_heatmap():
    """Test creating residual difference heatmap between identity and perturbed matrix."""
    u_a = np.eye(4, dtype=complex)
    u_b = np.eye(4, dtype=complex)
    u_b[0, 0] = -1.0  # Introduce deviation

    fig = create_product_distance_heatmap(u_a, u_b, title="Test Residual")
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].z.shape == (4, 4)


def test_verification_math_with_global_phase(sample_circuit: CircuitData):
    """Verify that a global phase addition is recognized as equivalent when global phase is ignored."""
    u_a = compute_unitary(sample_circuit)
    phase_shift = np.pi / 3.0
    u_b = u_a * np.exp(1j * phase_shift)

    # With ignore_global_phase=True
    is_equiv, phi, dist = check_unitary_equivalence(u_a, u_b, tolerance=1e-7, ignore_global_phase=True)
    assert is_equiv is True
    assert phi is not None
    # Check that estimated phase angle is close to -pi/3 or pi/3 modulo 2pi
    assert dist <= 1e-7

    # With ignore_global_phase=False (strict matrix equality)
    is_equiv_strict, _, dist_strict = check_unitary_equivalence(u_a, u_b, tolerance=1e-7, ignore_global_phase=False)
    assert is_equiv_strict is False
    assert dist_strict > 0.1


def test_verification_math_perturbation_rejection(sample_circuit: CircuitData):
    """Verify that adding an unexpected gate causes equivalence rejection."""
    u_a = compute_unitary(sample_circuit)

    # Perturbed circuit: Add an extra X gate
    perturbed_circuit = CircuitData(
        circuit_id="test_perturbed",
        num_qubits=2,
        gates=list(sample_circuit.gates) + [GateOperation(name="x", qubits=[0], params=[])],
    )
    u_b = compute_unitary(perturbed_circuit)

    is_equiv, _, dist = check_unitary_equivalence(u_a, u_b, tolerance=1e-7, ignore_global_phase=True)
    assert is_equiv is False
    assert dist > 1e-2
