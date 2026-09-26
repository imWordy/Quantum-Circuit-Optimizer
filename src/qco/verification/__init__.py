"""Verification subpackage exports."""

from qco.verification.equivalence import (
    EquivalenceResult,
    check_equivalence,
    check_unitary_equivalence,
)
from qco.verification.simulator import (
    compute_unitary,
    simulate_statevector,
)

__all__ = [
    "EquivalenceResult",
    "check_equivalence",
    "check_unitary_equivalence",
    "compute_unitary",
    "simulate_statevector",
]
