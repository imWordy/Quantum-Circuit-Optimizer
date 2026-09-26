"""Exact Quantum Equivalence Verification for QCO.

Independent correctness authority for determining whether two quantum circuits
are equivalent up to global phase and within numerical tolerance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qco.circuits.representation import CircuitData
from qco.utils.config import load_config
from qco.verification.simulator import compute_unitary


@dataclass(frozen=True)
class EquivalenceResult:
    """Structured report returned by equivalence verification."""

    equivalent: bool
    tolerance: float
    comparison_method: str
    global_phase_handling: str
    phase_difference: Optional[float] = None
    matrix_distance: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


def check_unitary_equivalence(
    u_a: np.ndarray,
    u_b: np.ndarray,
    tolerance: float = 1e-7,
    ignore_global_phase: bool = True,
) -> Tuple[bool, Optional[float], float]:
    r"""Check if two unitary matrices are equal up to a global phase factor.

    For unitaries U_A and U_B, they represent the same physical operation iff:
        U_A = e^(i * phi) * U_B
    This is true if U_A * U_B^\dagger is proportional to the identity matrix:
        U_A U_B^\dagger = e^(i * phi) * I

    Args:
        u_a: First unitary matrix.
        u_b: Second unitary matrix.
        tolerance: Maximum allowed absolute error / distance.
        ignore_global_phase: Whether to allow global phase factor e^(i * phi).

    Returns:
        Tuple of (equivalent, global_phase_diff, matrix_distance).
    """
    if u_a.shape != u_b.shape:
        return False, None, float("inf")

    n = u_a.shape[0]

    if not ignore_global_phase:
        diff = np.max(np.abs(u_a - u_b))
        return bool(diff <= tolerance), 0.0, float(diff)

    # U = U_A @ U_B^dagger
    u_prod = u_a @ u_b.conj().T

    # Find the largest magnitude element on the diagonal to determine candidate phase factor
    # In an ideal identity multiple, all diagonal elements equal e^(i * phi)
    diag_elements = np.diag(u_prod)
    idx = int(np.argmax(np.abs(diag_elements)))
    cand_phase_elem = diag_elements[idx]

    # If the candidate element is practically 0, unitaries are definitely not equivalent
    if np.abs(cand_phase_elem) < 0.5:
        dist = float(np.max(np.abs(u_prod - np.eye(n))))
        return False, None, dist

    # Normalize to unit complex number: e^(i * phi)
    phase_factor = cand_phase_elem / np.abs(cand_phase_elem)
    phase_angle = float(np.angle(phase_factor))

    # Target matrix should be phase_factor * Identity
    expected_prod = phase_factor * np.eye(n, dtype=complex)
    matrix_distance = float(np.max(np.abs(u_prod - expected_prod)))

    is_equiv = bool(matrix_distance <= tolerance)
    return is_equiv, phase_angle, matrix_distance


def check_equivalence(
    circuit_a: Union[CircuitData, QuantumCircuit],
    circuit_b: Union[CircuitData, QuantumCircuit],
    tolerance: Optional[float] = None,
    ignore_global_phase: bool = True,
    method: str = "unitary",
) -> EquivalenceResult:
    """Verify exact equivalence of two quantum circuits.

    Args:
        circuit_a: First circuit.
        circuit_b: Second circuit.
        tolerance: Numerical tolerance. Defaults to config verification.tolerance (1e-7).
        ignore_global_phase: Whether global phase differences are allowed (default True).
        method: Comparison method ('unitary').

    Returns:
        EquivalenceResult detailing equivalence status, tolerance, and metrics.
    """
    if tolerance is None:
        try:
            cfg = load_config()
            tolerance = float(cfg.get("verification", {}).get("tolerance", 1e-7))
        except Exception:
            tolerance = 1e-7

    num_qubits_a = circuit_a.num_qubits
    num_qubits_b = circuit_b.num_qubits

    if num_qubits_a != num_qubits_b:
        return EquivalenceResult(
            equivalent=False,
            tolerance=tolerance,
            comparison_method=method,
            global_phase_handling="up_to_global_phase" if ignore_global_phase else "exact",
            reason=f"Qubit counts do not match: {num_qubits_a} vs {num_qubits_b}",
        )

    if method != "unitary":
        raise ValueError(f"Unsupported verification method '{method}'. Allowed: ['unitary']")

    u_a = compute_unitary(circuit_a)
    u_b = compute_unitary(circuit_b)

    is_equiv, phase_diff, dist = check_unitary_equivalence(
        u_a=u_a,
        u_b=u_b,
        tolerance=tolerance,
        ignore_global_phase=ignore_global_phase,
    )

    reason = None if is_equiv else f"Unitaries differ by distance {dist:.2e} > tolerance {tolerance:.2e}"

    return EquivalenceResult(
        equivalent=is_equiv,
        tolerance=tolerance,
        comparison_method=method,
        global_phase_handling="up_to_global_phase" if ignore_global_phase else "exact",
        phase_difference=phase_diff,
        matrix_distance=dist,
        reason=reason,
    )
