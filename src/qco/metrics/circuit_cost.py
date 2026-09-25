"""Deterministic Circuit Cost Metrics for QCO.

Calculates:
- Total gate count
- Single-qubit gate count
- Two-qubit gate count
- CNOT count
- Circuit depth (exact DAG-based layer depth)
- Cost reductions (absolute and percentage, safe against division by zero)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from qco.circuits.representation import CircuitData, GateOperation


@dataclass(frozen=True)
class CircuitCost:
    """Immutable container holding circuit cost metrics."""

    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    cnot_count: int
    depth: int

    def to_dict(self) -> Dict[str, int]:
        """Convert metrics to dictionary."""
        return asdict(self)


def calculate_depth(circuit: CircuitData) -> int:
    """Calculate circuit depth deterministically based on qubit operation dependencies.

    The depth is the length of the critical path of sequential quantum operations.
    Two gates on disjoint qubits can execute in parallel in the same time step.
    """
    if len(circuit) == 0:
        return 0

    # Track current depth at each qubit
    qubit_depths = [0] * circuit.num_qubits

    for gate in circuit.gates:
        # The layer for this gate is max(depth of all participating qubits) + 1
        current_layer = max(qubit_depths[q] for q in gate.qubits) + 1
        for q in gate.qubits:
            qubit_depths[q] = current_layer

    return max(qubit_depths) if qubit_depths else 0


def compute_circuit_cost(circuit: CircuitData) -> CircuitCost:
    """Calculate all standard cost metrics for a canonical circuit.

    Args:
        circuit: CircuitData instance to evaluate.

    Returns:
        CircuitCost instance with computed metrics.
    """
    total_gates = len(circuit)
    single_qubit_gates = 0
    two_qubit_gates = 0
    cnot_count = 0

    for gate in circuit.gates:
        num_q = len(gate.qubits)
        if num_q == 1:
            single_qubit_gates += 1
        elif num_q == 2:
            two_qubit_gates += 1
            if gate.name in ("CX", "CNOT"):
                cnot_count += 1
        else:
            raise ValueError(f"Gates with {num_q} qubits are not supported: {gate.name}")

    depth = calculate_depth(circuit)

    return CircuitCost(
        total_gates=total_gates,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        cnot_count=cnot_count,
        depth=depth,
    )


@dataclass(frozen=True)
class CostReduction:
    """Container holding absolute and percentage reductions between original and candidate costs."""

    # Absolute reductions: Original - Candidate
    gate_reduction: int
    cnot_reduction: int
    depth_reduction: int
    single_qubit_reduction: int
    two_qubit_reduction: int

    # Percentage reductions: (Original - Candidate) / Original * 100
    gate_reduction_pct: float
    cnot_reduction_pct: float
    depth_reduction_pct: float
    single_qubit_reduction_pct: float
    two_qubit_reduction_pct: float

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """Convert reduction metrics to dictionary."""
        return asdict(self)


def _safe_pct_reduction(orig: int, cand: int) -> float:
    """Compute percentage reduction safely handling zero denominator."""
    if orig == 0:
        return 0.0 if cand == 0 else -100.0
    return float((orig - cand) / orig * 100.0)


def compute_cost_reduction(
    original: Union[CircuitCost, CircuitData],
    candidate: Union[CircuitCost, CircuitData],
) -> CostReduction:
    """Compute absolute and percentage reductions between original and candidate circuit.

    Args:
        original: Original CircuitCost or CircuitData.
        candidate: Candidate CircuitCost or CircuitData.

    Returns:
        CostReduction containing reductions across all metrics.
    """
    orig_cost = compute_circuit_cost(original) if isinstance(original, CircuitData) else original
    cand_cost = compute_circuit_cost(candidate) if isinstance(candidate, CircuitData) else candidate

    gate_red = orig_cost.total_gates - cand_cost.total_gates
    cnot_red = orig_cost.cnot_count - cand_cost.cnot_count
    depth_red = orig_cost.depth - cand_cost.depth
    single_red = orig_cost.single_qubit_gates - cand_cost.single_qubit_gates
    two_red = orig_cost.two_qubit_gates - cand_cost.two_qubit_gates

    return CostReduction(
        gate_reduction=gate_red,
        cnot_reduction=cnot_red,
        depth_reduction=depth_red,
        single_qubit_reduction=single_red,
        two_qubit_reduction=two_red,
        gate_reduction_pct=_safe_pct_reduction(orig_cost.total_gates, cand_cost.total_gates),
        cnot_reduction_pct=_safe_pct_reduction(orig_cost.cnot_count, cand_cost.cnot_count),
        depth_reduction_pct=_safe_pct_reduction(orig_cost.depth, cand_cost.depth),
        single_qubit_reduction_pct=_safe_pct_reduction(
            orig_cost.single_qubit_gates, cand_cost.single_qubit_gates
        ),
        two_qubit_reduction_pct=_safe_pct_reduction(
            orig_cost.two_qubit_gates, cand_cost.two_qubit_gates
        ),
    )
