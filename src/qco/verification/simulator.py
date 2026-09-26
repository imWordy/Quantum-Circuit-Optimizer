"""Quantum Simulation and Unitary Calculation Utilities for QCO."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

from qco.circuits.representation import CircuitData


def compute_unitary(circuit: Union[CircuitData, QuantumCircuit]) -> np.ndarray:
    """Compute exact unitary matrix representation of a quantum circuit.

    Args:
        circuit: CircuitData or Qiskit QuantumCircuit.

    Returns:
        Complex numpy ndarray of shape (2^n, 2^n).
    """
    qc = circuit.to_qiskit() if isinstance(circuit, CircuitData) else circuit
    # Using Qiskit Operator calculates full unitary
    op = Operator(qc)
    return op.data


def simulate_statevector(
    circuit: Union[CircuitData, QuantumCircuit],
    initial_state: Optional[Union[np.ndarray, Statevector]] = None,
) -> np.ndarray:
    """Simulate output statevector of a quantum circuit.

    Args:
        circuit: CircuitData or Qiskit QuantumCircuit.
        initial_state: Optional initial statevector. Defaults to |0...0>.

    Returns:
        Complex numpy 1D array of shape (2^n,).
    """
    qc = circuit.to_qiskit() if isinstance(circuit, CircuitData) else circuit
    if initial_state is not None:
        sv = Statevector(initial_state).evolve(qc)
    else:
        sv = Statevector.from_int(0, 2**qc.num_qubits).evolve(qc)
    return sv.data
