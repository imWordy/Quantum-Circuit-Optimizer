"""Curated educational quantum circuit presets for students."""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from qco.circuits.representation import CircuitData, GateOperation


def get_bell_state_circuit() -> CircuitData:
    """Create maximally entangled Bell state preparation (|00> + |11>) / sqrt(2)."""
    gates = [
        GateOperation("H", (0,)),
        GateOperation("CX", (0, 1)),
    ]
    return CircuitData(
        num_qubits=2,
        gates=gates,
        metadata={"name": "Bell State Preparation", "description": "Standard 2-qubit maximally entangled Bell state"},
    )


def get_ghz_state_circuit() -> CircuitData:
    """Create 3-qubit Greenberger-Horne-Zeilinger (GHZ) state (|000> + |111>) / sqrt(2)."""
    gates = [
        GateOperation("H", (0,)),
        GateOperation("CX", (0, 1)),
        GateOperation("CX", (1, 2)),
    ]
    return CircuitData(
        num_qubits=3,
        gates=gates,
        metadata={"name": "3-Qubit GHZ State", "description": "3-qubit maximally entangled GHZ state"},
    )


def get_teleportation_circuit() -> CircuitData:
    """Create standard Quantum Teleportation protocol circuit with unoptimized cleanup gates."""
    gates = [
        # State prep on qubit 0 (arbitrary rotation)
        GateOperation("RY", (0,), params=[1.047197]),  # pi/3
        # Entangle qubit 1 and 2 (Bell pair)
        GateOperation("H", (1,)),
        GateOperation("CX", (1, 2)),
        # Teleportation measurement basis rotation
        GateOperation("CX", (0, 1)),
        GateOperation("H", (0,)),
        # Redundant identity insertions illustrating optimization opportunity
        GateOperation("H", (0,)),
        GateOperation("H", (0,)),
        GateOperation("CX", (1, 2)),
        GateOperation("CX", (1, 2)),
    ]
    return CircuitData(
        num_qubits=3,
        gates=gates,
        metadata={"name": "Quantum Teleportation (Redundant)", "description": "Teleportation protocol with redundant cancellation sites"},
    )


def get_redundant_cancellation_circuit() -> CircuitData:
    """Circuit loaded with self-inverse, CNOT, and rotation cancellations."""
    gates = [
        GateOperation("H", (0,)),
        GateOperation("H", (0,)),                 # H H -> I
        GateOperation("CX", (0, 1)),
        GateOperation("CX", (0, 1)),              # CX CX -> I
        GateOperation("RZ", (0,), params=[0.5]),
        GateOperation("RZ", (0,), params=[-0.5]), # RZ(0.5) RZ(-0.5) -> RZ(0) -> I
        GateOperation("X", (1,)),
        GateOperation("X", (1,)),                 # X X -> I
        GateOperation("Y", (0,)),
        GateOperation("Z", (1,)),
    ]
    return CircuitData(
        num_qubits=2,
        gates=gates,
        metadata={"name": "Redundant Cancellation Pattern", "description": "Dense clusters of H, CX, RZ, and Pauli self-inverse pairs"},
    )


def get_qft_3qubit_circuit() -> CircuitData:
    """3-qubit Quantum Fourier Transform (QFT) with simulated inverse pairs for cancellation."""
    gates = [
        GateOperation("H", (0,)),
        GateOperation("RZ", (0,), params=[np.pi / 2]),
        GateOperation("CX", (1, 0)),
        GateOperation("RZ", (0,), params=[-np.pi / 2]),
        GateOperation("CX", (1, 0)),
        GateOperation("H", (1,)),
        GateOperation("CX", (2, 1)),
        GateOperation("CX", (2, 1)),              # Redundant CNOT pair
        GateOperation("H", (2,)),
    ]
    return CircuitData(
        num_qubits=3,
        gates=gates,
        metadata={"name": "3-Qubit QFT (with Redundancies)", "description": "Quantum Fourier Transform structure with synthetic cancellation pairs"},
    )


PRESET_CIRCUITS: Dict[str, CircuitData] = {
    "Redundant Cancellations (Benchmark)": get_redundant_cancellation_circuit(),
    "Bell State Preparation (2 Qubits)": get_bell_state_circuit(),
    "GHZ State Preparation (3 Qubits)": get_ghz_state_circuit(),
    "Quantum Teleportation Protocol": get_teleportation_circuit(),
    "3-Qubit Quantum Fourier Transform": get_qft_3qubit_circuit(),
}
