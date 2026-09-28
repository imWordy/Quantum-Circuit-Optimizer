"""Data schemas for QCO ground-truth dataset — Phase 8."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ObservationRecord:
    """One verified observation in the ground-truth dataset.

    Each record stores all metrics necessary for GNN training.
    Circuits are stored by reference (circuit_id) to avoid duplication.
    """

    # Circuit references
    parent_circuit_id: str
    candidate_id: str

    # Transformation info
    transformation_type: str
    transformation_location: str

    # Parent circuit costs
    parent_gate_count: int
    parent_cnot_count: int
    parent_depth: int

    # Candidate circuit costs (preliminary values, from candidate generation)
    candidate_gate_count: int
    candidate_cnot_count: int
    candidate_depth: int

    # Actual improvements (independently verified)
    gate_reduction: int
    cnot_reduction: int
    depth_reduction: int

    # Equivalence flag (from Phase 5 independent verifier)
    equivalent: bool

    # Optional: candidate circuit ID for reconstruction
    candidate_circuit_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservationRecord:
        """Construct from dictionary."""
        return cls(**data)
