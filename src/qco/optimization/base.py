"""Base classes and result containers for quantum circuit optimizers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from qco.circuits.representation import CircuitData
from qco.metrics.circuit_cost import CircuitCost, compute_circuit_cost


@dataclass
class StepRecord:
    """Trace of a single iteration step during optimization."""

    iteration: int
    candidate_id: str
    transformation_name: str
    transformation_location: str
    accepted: bool
    gate_count_before: int
    gate_count_after: int
    cnot_count_before: int
    cnot_count_after: int
    depth_before: int
    depth_after: int
    reason: str


@dataclass
class OptimizationResult:
    """Standardized result returned by all optimizers."""

    optimizer_name: str
    initial_circuit_id: str
    final_circuit_id: str
    initial_circuit: CircuitData
    final_circuit: CircuitData
    initial_cost: CircuitCost
    final_cost: CircuitCost
    gate_reduction: int
    cnot_reduction: int
    depth_reduction: int
    candidate_evaluations: int
    equivalence_checks: int
    iterations: int
    runtime: float  # runtime in seconds
    converged: bool
    is_equivalent: bool
    history: List[StepRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary (without embedding full CircuitData)."""
        return {
            "optimizer_name": self.optimizer_name,
            "initial_circuit_id": self.initial_circuit_id,
            "final_circuit_id": self.final_circuit_id,
            "initial_gate_count": self.initial_cost.total_gates,
            "final_gate_count": self.final_cost.total_gates,
            "initial_cnot_count": self.initial_cost.cnot_count,
            "final_cnot_count": self.final_cost.cnot_count,
            "initial_depth": self.initial_cost.depth,
            "final_depth": self.final_cost.depth,
            "gate_reduction": self.gate_reduction,
            "cnot_reduction": self.cnot_reduction,
            "depth_reduction": self.depth_reduction,
            "candidate_evaluations": self.candidate_evaluations,
            "equivalence_checks": self.equivalence_checks,
            "iterations": self.iterations,
            "runtime": self.runtime,
            "converged": self.converged,
            "is_equivalent": self.is_equivalent,
        }


class BaseOptimizer(ABC):
    """Abstract interface for quantum circuit optimizers."""

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-7,
        seed: Optional[int] = 42,
    ) -> None:
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the optimizer."""
        pass

    @abstractmethod
    def optimize(self, circuit: CircuitData) -> OptimizationResult:
        """Run iterative optimization loop on the given circuit."""
        pass
